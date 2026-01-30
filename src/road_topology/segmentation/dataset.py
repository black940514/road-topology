"""PyTorch Dataset for road topology segmentation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from road_topology.core.logging import get_logger
from road_topology.core.types import CLASS_NAMES, NUM_CLASSES

logger = get_logger(__name__)


class RoadTopologyDataset(Dataset):
    """Dataset for road topology segmentation.

    Loads image + mask pairs from directory structure:
    ```
    root/
        images/
            image_001.jpg
            image_002.jpg
        masks/
            image_001.png
            image_002.png
        confidence/  (optional)
            image_001.npy
    ```

    Args:
        root: Dataset root directory.
        split: Data split (train, val, test).
        transforms: Albumentations transforms.
        use_confidence_weights: Whether to load confidence maps.
    """

    CLASS_NAMES = CLASS_NAMES
    NUM_CLASSES = NUM_CLASSES

    def __init__(
        self,
        root: Path | str,
        split: str = "train",
        transforms: Any = None,
        use_confidence_weights: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.use_confidence_weights = use_confidence_weights

        # Setup paths
        self.images_dir = self.root / split / "images"
        self.masks_dir = self.root / split / "masks"
        self.confidence_dir = self.root / split / "confidence"

        # Find all images
        self.image_paths = sorted(self.images_dir.glob("*.jpg")) + \
                          sorted(self.images_dir.glob("*.png"))

        if not self.image_paths:
            logger.warning(f"No images found in {self.images_dir}")
        else:
            logger.info(f"Loaded {len(self.image_paths)} images for {split} split")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a sample.

        Returns:
            Dictionary with:
            - image: (C, H, W) tensor
            - mask: (H, W) tensor with class indices
            - confidence: (H, W) tensor (if use_confidence_weights)
        """
        img_path = self.image_paths[idx]

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask_path = self.masks_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            mask_path = self.masks_dir / f"{img_path.stem}_mask.png"

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")

        # Load confidence if requested
        confidence = None
        if self.use_confidence_weights:
            conf_path = self.confidence_dir / f"{img_path.stem}.npy"
            if conf_path.exists():
                confidence = np.load(conf_path)

        # Apply transforms
        if self.transforms is not None:
            if confidence is not None:
                transformed = self.transforms(
                    image=image, mask=mask, confidence=confidence
                )
                confidence = transformed["confidence"]
            else:
                transformed = self.transforms(image=image, mask=mask)

            image = transformed["image"]
            mask = transformed["mask"]

        # Convert to tensors
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask.astype(np.int64))
        elif isinstance(mask, torch.Tensor):
            mask = mask.long()

        result = {
            "image": image,
            "mask": mask,
            "image_path": str(img_path),
        }

        if confidence is not None:
            result["confidence"] = torch.from_numpy(confidence).float()

        return result

    @classmethod
    def create_splits(
        cls,
        root: Path,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        transforms_train: Any = None,
        transforms_val: Any = None,
    ) -> tuple["RoadTopologyDataset", "RoadTopologyDataset", "RoadTopologyDataset"]:
        """Create train/val/test splits from a flat directory.

        Expects:
        ```
        root/
            images/
            masks/
        ```

        Creates:
        ```
        root/
            train/images/, train/masks/
            val/images/, val/masks/
            test/images/, test/masks/
        ```

        Args:
            root: Dataset root.
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            seed: Random seed.
            transforms_train: Training transforms.
            transforms_val: Validation transforms.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).
        """
        root = Path(root)
        images_dir = root / "images"
        masks_dir = root / "masks"

        # Get all images
        all_images = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

        if not all_images:
            raise ValueError(f"No images found in {images_dir}")

        # Shuffle
        np.random.seed(seed)
        indices = np.random.permutation(len(all_images))

        # Split indices
        n_train = int(len(all_images) * train_ratio)
        n_val = int(len(all_images) * val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        # Create split directories and move files
        for split, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            split_img_dir = root / split / "images"
            split_mask_dir = root / split / "masks"
            split_img_dir.mkdir(parents=True, exist_ok=True)
            split_mask_dir.mkdir(parents=True, exist_ok=True)

            for idx in split_idx:
                img_path = all_images[idx]
                mask_path = masks_dir / f"{img_path.stem}.png"

                # Create symlinks (or copy)
                dst_img = split_img_dir / img_path.name
                dst_mask = split_mask_dir / mask_path.name

                if not dst_img.exists():
                    dst_img.symlink_to(img_path.resolve())
                if mask_path.exists() and not dst_mask.exists():
                    dst_mask.symlink_to(mask_path.resolve())

        logger.info(
            f"Created splits: train={n_train}, val={n_val}, test={len(test_idx)}"
        )

        return (
            cls(root, "train", transforms_train),
            cls(root, "val", transforms_val),
            cls(root, "test", transforms_val),
        )


class PseudoLabelDataset(RoadTopologyDataset):
    """Dataset specifically for pseudo-labeled data.

    Always loads confidence weights and supports confidence-weighted sampling.
    """

    def __init__(
        self,
        root: Path | str,
        split: str = "train",
        transforms: Any = None,
        min_confidence: float = 0.0,
    ) -> None:
        super().__init__(root, split, transforms, use_confidence_weights=True)
        self.min_confidence = min_confidence

    def filter_by_confidence(self, threshold: float) -> "PseudoLabelDataset":
        """Filter samples by mean confidence.

        Args:
            threshold: Minimum mean confidence to keep.

        Returns:
            New dataset with filtered samples.
        """
        filtered_paths = []

        for img_path in self.image_paths:
            conf_path = self.confidence_dir / f"{img_path.stem}.npy"
            if conf_path.exists():
                conf = np.load(conf_path)
                if conf.mean() >= threshold:
                    filtered_paths.append(img_path)

        new_dataset = PseudoLabelDataset(
            self.root,
            self.split,
            self.transforms,
            self.min_confidence,
        )
        new_dataset.image_paths = filtered_paths

        logger.info(
            f"Filtered {len(self.image_paths)} -> {len(filtered_paths)} samples "
            f"with confidence >= {threshold}"
        )

        return new_dataset


class LaneInstanceDataset(RoadTopologyDataset):
    """Dataset for lane instance segmentation.

    Extends RoadTopologyDataset to load both semantic and instance masks.

    Directory structure:
    ```
    root/
        {split}/
            images/
                image_001.jpg
            semantic_masks/  (or masks/)
                image_001.png
            instance_masks/
                image_001.png
    ```

    Instance masks should contain unique IDs for each lane instance:
    - 0: background
    - 1, 2, 3, ...: individual lane instances

    Args:
        root: Dataset root directory.
        split: Data split (train, val, test).
        transforms: Albumentations transforms.
        semantic_dir_name: Name of semantic mask directory (default: "semantic_masks").
        instance_dir_name: Name of instance mask directory (default: "instance_masks").
    """

    def __init__(
        self,
        root: Path | str,
        split: str = "train",
        transforms: Any = None,
        semantic_dir_name: str = "semantic_masks",
        instance_dir_name: str = "instance_masks",
    ) -> None:
        # Initialize parent without loading masks yet
        super().__init__(root, split, transforms, use_confidence_weights=False)

        # Override mask directories
        self.semantic_masks_dir = self.root / split / semantic_dir_name
        self.instance_masks_dir = self.root / split / instance_dir_name

        # Fallback to "masks" for semantic if semantic_masks doesn't exist
        if not self.semantic_masks_dir.exists():
            fallback_dir = self.root / split / "masks"
            if fallback_dir.exists():
                logger.info(f"Using fallback semantic masks directory: {fallback_dir}")
                self.semantic_masks_dir = fallback_dir
            else:
                logger.warning(f"Semantic masks directory not found: {self.semantic_masks_dir}")

        # Check instance masks directory
        if not self.instance_masks_dir.exists():
            raise FileNotFoundError(
                f"Instance masks directory not found: {self.instance_masks_dir}\n"
                f"For lane instance segmentation, you must provide instance_masks/ directory.\n"
                f"Expected structure:\n"
                f"  {self.root}/{split}/\n"
                f"    images/\n"
                f"    {semantic_dir_name}/  (or masks/)\n"
                f"    {instance_dir_name}/"
            )

        logger.info(
            f"LaneInstanceDataset initialized with {len(self.image_paths)} images",
            semantic_dir=str(self.semantic_masks_dir),
            instance_dir=str(self.instance_masks_dir),
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a sample with both semantic and instance masks.

        Returns:
            Dictionary with:
            - image: (C, H, W) tensor
            - semantic_mask: (H, W) tensor with class indices
            - instance_mask: (H, W) tensor with instance IDs
            - image_path: str
        """
        img_path = self.image_paths[idx]

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load semantic mask
        semantic_mask_path = self.semantic_masks_dir / f"{img_path.stem}.png"
        if not semantic_mask_path.exists():
            semantic_mask_path = self.semantic_masks_dir / f"{img_path.stem}_mask.png"

        semantic_mask = cv2.imread(str(semantic_mask_path), cv2.IMREAD_GRAYSCALE)
        if semantic_mask is None:
            raise ValueError(f"Failed to load semantic mask: {semantic_mask_path}")

        # Load instance mask
        instance_mask_path = self.instance_masks_dir / f"{img_path.stem}.png"
        if not instance_mask_path.exists():
            instance_mask_path = self.instance_masks_dir / f"{img_path.stem}_instance.png"

        # Try loading as grayscale first (for single-channel encoded IDs)
        instance_mask = cv2.imread(str(instance_mask_path), cv2.IMREAD_GRAYSCALE)

        if instance_mask is None:
            # Try loading as color and convert to instance IDs
            instance_mask_rgb = cv2.imread(str(instance_mask_path))
            if instance_mask_rgb is None:
                raise ValueError(f"Failed to load instance mask: {instance_mask_path}")

            # Convert RGB to instance IDs (assumes each unique color is an instance)
            instance_mask = self._rgb_to_instance_ids(instance_mask_rgb)
            logger.debug(f"Converted RGB instance mask to {instance_mask.max()} instances")

        # Apply transforms
        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                masks=[semantic_mask, instance_mask],
            )
            image = transformed["image"]
            semantic_mask = transformed["masks"][0]
            instance_mask = transformed["masks"][1]

        # Convert to tensors
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if isinstance(semantic_mask, np.ndarray):
            semantic_mask = torch.from_numpy(semantic_mask.astype(np.int64))
        elif isinstance(semantic_mask, torch.Tensor):
            semantic_mask = semantic_mask.long()

        if isinstance(instance_mask, np.ndarray):
            instance_mask = torch.from_numpy(instance_mask.astype(np.int64))
        elif isinstance(instance_mask, torch.Tensor):
            instance_mask = instance_mask.long()

        return {
            "image": image,
            "semantic_mask": semantic_mask,
            "instance_mask": instance_mask,
            "image_path": str(img_path),
        }

    @staticmethod
    def _rgb_to_instance_ids(rgb_mask: np.ndarray) -> np.ndarray:
        """Convert RGB instance mask to integer instance IDs.

        Args:
            rgb_mask: (H, W, 3) RGB mask where each unique color is an instance.

        Returns:
            (H, W) mask with integer instance IDs.
        """
        h, w = rgb_mask.shape[:2]
        instance_mask = np.zeros((h, w), dtype=np.int32)

        # Flatten to (H*W, 3)
        rgb_flat = rgb_mask.reshape(-1, 3)

        # Find unique colors
        unique_colors = np.unique(rgb_flat, axis=0)

        # Assign IDs (0 is typically background/black)
        for instance_id, color in enumerate(unique_colors):
            if np.all(color == 0):  # Skip black (background)
                continue

            mask = np.all(rgb_mask == color, axis=2)
            instance_mask[mask] = instance_id

        return instance_mask
