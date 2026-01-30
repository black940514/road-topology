"""SAM (Segment Anything Model) wrapper for road segmentation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from road_topology.core.config import SAMConfig
from road_topology.core.device import get_device, empty_cache
from road_topology.core.exceptions import ModelLoadError, SegmentationError
from road_topology.core.logging import get_logger
from road_topology.core.types import BoundingBox, CLASS_NAMES

logger = get_logger(__name__)

ROAD = CLASS_NAMES.index("road")


class SAMSegmenter:
    """SAM model wrapper for segmentation.

    Supports point prompts, box prompts, and YOLO detection prompts.
    The YOLO bbox prompting method is the PRIMARY approach.

    Args:
        model_type: SAM variant (vit_h, vit_l, vit_b).
        checkpoint: Path to model weights. Auto-downloads if None.
        device: Device to use (cuda, cpu).
    """

    MODEL_URLS = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }

    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint: Path | None = None,
        device: str | None = None,
        config: SAMConfig | None = None,
    ) -> None:
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.device = str(get_device(device or "auto"))
        self.config = config or SAMConfig()

        self._model = None
        self._predictor = None
        self._image_set = False

    @property
    def predictor(self):
        """Lazy load SAM predictor."""
        if self._predictor is None:
            self._load_model()
        return self._predictor

    def _load_model(self) -> None:
        """Load SAM model."""
        try:
            from segment_anything import sam_model_registry, SamPredictor

            # Determine checkpoint path
            if self.checkpoint is None:
                # Use default location
                checkpoint_dir = Path("./models/sam")
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Map model type to filename
                filenames = {
                    "vit_h": "sam_vit_h_4b8939.pth",
                    "vit_l": "sam_vit_l_0b3195.pth",
                    "vit_b": "sam_vit_b_01ec64.pth",
                }
                self.checkpoint = checkpoint_dir / filenames.get(self.model_type, filenames["vit_h"])

            if not Path(self.checkpoint).exists():
                raise ModelLoadError(
                    f"SAM checkpoint not found: {self.checkpoint}. "
                    "Run: python scripts/download_models.py download --model sam_vit_h",
                    model_path=str(self.checkpoint),
                )

            logger.info("Loading SAM model", model_type=self.model_type, device=self.device)

            # Load checkpoint to CPU first to ensure CUDA/MPS compatibility
            state_dict = torch.load(str(self.checkpoint), map_location="cpu", weights_only=True)
            self._model = sam_model_registry[self.model_type]()  # Create model without checkpoint
            self._model.load_state_dict(state_dict)
            self._model.to(device=self.device)
            self._predictor = SamPredictor(self._model)

            logger.info("SAM model loaded successfully")

        except ImportError:
            raise ModelLoadError(
                "segment-anything not installed. Install with: pip install segment-anything"
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load SAM model: {e}")

    def set_image(self, image: np.ndarray) -> None:
        """Set image for segmentation.

        Args:
            image: RGB image as numpy array (H, W, 3).
        """
        # Convert BGR to RGB if needed (check blue channel dominance)
        if image.shape[2] == 3:
            # Assume input is RGB
            self.predictor.set_image(image)
        self._image_set = True

    def segment_with_points(
        self,
        image: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Segment with point prompts.

        Args:
            image: RGB image (H, W, 3).
            points: Point coordinates (N, 2).
            labels: Point labels (N,) - 1=foreground, 0=background.

        Returns:
            Binary mask (H, W).
        """
        self.set_image(image)

        masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )

        # Return best mask
        best_idx = np.argmax(scores)
        return masks[best_idx].astype(np.uint8)

    def segment_with_box(
        self,
        image: np.ndarray,
        box: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Segment with bounding box prompt.

        Args:
            image: RGB image (H, W, 3).
            box: Bounding box (x1, y1, x2, y2).

        Returns:
            Binary mask (H, W).
        """
        self.set_image(image)

        box_array = np.array(box)

        masks, scores, _ = self.predictor.predict(
            box=box_array,
            multimask_output=True,
        )

        best_idx = np.argmax(scores)
        return masks[best_idx].astype(np.uint8)

    def segment_with_yolo_detections(
        self,
        image: np.ndarray,
        detections: list[BoundingBox],
        bbox_expansion: float = 0.1,
        class_label: int = ROAD,
    ) -> np.ndarray:
        """PRIMARY METHOD: Segment using YOLO vehicle detections.

        For each vehicle detection:
        1. Expand bounding box by expansion ratio
        2. Use expanded bbox as SAM prompt
        3. Assume segmented region is road (vehicles drive on roads)
        4. Union all masks

        Args:
            image: RGB image (H, W, 3).
            detections: Vehicle bounding boxes from YOLO.
            bbox_expansion: Ratio to expand boxes (e.g., 0.1 = 10%).
            class_label: Class index to assign to road regions.

        Returns:
            Semantic mask (H, W) with road regions labeled.
        """
        if not detections:
            logger.debug("No detections provided, returning empty mask")
            return np.zeros(image.shape[:2], dtype=np.uint8)

        self.set_image(image)

        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for det in detections:
            # Expand bounding box
            expanded = det.expand(bbox_expansion)

            # Clip to image bounds
            box = (
                max(0, int(expanded.x1)),
                max(0, int(expanded.y1)),
                min(w, int(expanded.x2)),
                min(h, int(expanded.y2)),
            )

            # Skip invalid boxes
            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            try:
                masks, scores, _ = self.predictor.predict(
                    box=np.array(box),
                    multimask_output=True,
                )

                # Use best scoring mask
                best_idx = np.argmax(scores)
                mask = masks[best_idx]

                # Union with combined mask
                combined_mask = np.maximum(combined_mask, mask.astype(np.uint8))

            except Exception as e:
                logger.warning(f"SAM prediction failed for box {box}: {e}")
                continue

        # Convert to semantic mask with class label
        semantic_mask = np.where(combined_mask > 0, class_label, 0).astype(np.uint8)

        return semantic_mask

    def segment_everything(
        self,
        image: np.ndarray,
        points_per_side: int | None = None,
    ) -> list[np.ndarray]:
        """Generate all possible masks for an image.

        Args:
            image: RGB image (H, W, 3).
            points_per_side: Grid density for automatic prompts.

        Returns:
            List of binary masks.
        """
        try:
            from segment_anything import SamAutomaticMaskGenerator

            if self._model is None:
                self._load_model()

            generator = SamAutomaticMaskGenerator(
                self._model,
                points_per_side=points_per_side or self.config.points_per_side,
                pred_iou_thresh=self.config.pred_iou_thresh,
                stability_score_thresh=self.config.stability_score_thresh,
                box_nms_thresh=self.config.box_nms_thresh,
            )

            masks_data = generator.generate(image)

            return [m["segmentation"].astype(np.uint8) for m in masks_data]

        except Exception as e:
            raise SegmentationError(f"Segment everything failed: {e}")

    def cleanup(self) -> None:
        """Release GPU memory."""
        if self._model is not None:
            del self._model
            del self._predictor
            self._model = None
            self._predictor = None
            self._image_set = False

            empty_cache(self.device)

            logger.info("Released SAM GPU memory")

    def __enter__(self) -> "SAMSegmenter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.cleanup()
