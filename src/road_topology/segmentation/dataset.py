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
