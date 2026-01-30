"""Lane-aware predictor with instance segmentation."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from road_topology.core.device import get_device
from road_topology.core.exceptions import InferenceError, ModelLoadError
from road_topology.core.logging import get_logger
from road_topology.segmentation.postprocess import LanePostProcessor
from road_topology.segmentation.transforms import get_val_transforms

logger = get_logger(__name__)


class LanePredictor:
    """End-to-end predictor for lane instance segmentation.

    Loads a SegFormerLaneModel checkpoint and provides inference
    with post-processing to extract lane instances.

    Args:
        model_path: Path to model checkpoint.
        device: Inference device.
        image_size: Input image size for preprocessing.
        clustering_method: Clustering method for post-processing ("meanshift" or "dbscan").
        bandwidth: Bandwidth for mean-shift clustering.
        min_pixels: Minimum pixels per lane instance.
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "auto",
        image_size: tuple[int, int] = (512, 512),
        clustering_method: str = "meanshift",
        bandwidth: float = 0.5,
        min_pixels: int = 100,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = str(get_device(device))
        self.image_size = image_size

        self._model = None
        self._transforms = get_val_transforms(image_size[0], normalize=False)
        self._postprocessor = LanePostProcessor(
            clustering_method=clustering_method,
            bandwidth=bandwidth,
            min_pixels=min_pixels,
        )

    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self) -> None:
        """Load SegFormerLaneModel from checkpoint."""
        try:
            logger.info(f"Loading lane model from {self.model_path}")

            # Import here to avoid circular dependency
            from road_topology.segmentation.models.segformer_lane import SegFormerLaneModel

            self._model = SegFormerLaneModel.load(self.model_path, self.device)
            self._model.eval()

            logger.info("Lane model loaded successfully")

        except Exception as e:
            raise ModelLoadError(f"Failed to load lane model: {e}")

    def warmup(self, input_shape: tuple[int, int, int] = (3, 512, 512)) -> None:
        """Warmup model with dummy input."""
        dummy = torch.randn(1, *input_shape).to(self.device)
        with torch.no_grad():
            self.model(dummy)
        logger.info("Lane model warmed up")

    def predict(self, image: np.ndarray, lane_class_id: int = 1) -> dict:
        """Predict lane instance segmentation for single image.

        Args:
            image: RGB image (H, W, 3).
            lane_class_id: Class ID for lane markings (default 1).

        Returns:
            Dictionary with:
                - semantic_mask: (H, W) semantic segmentation
                - lane_instances: (H, W) instance IDs (0=background)
                - crosswalk_mask: (H, W) crosswalk segmentation
                - lane_count: Number of detected lanes
                - lane_centers: (N, 2) array of lane centers
                - inference_time_ms: Total inference time
        """
        original_shape = image.shape[:2]

        # Preprocess
        transformed = self._transforms(image=image)
        input_tensor = (
            torch.from_numpy(transformed["image"].transpose(2, 0, 1))
            .float()
            .unsqueeze(0)
            / 255.0
        )
        input_tensor = input_tensor.to(self.device)

        # Inference
        start = perf_counter()
        with torch.no_grad():
            outputs = self.model(input_tensor)

            # Upsample to original size
            logits = F.interpolate(
                outputs["semantic_logits"],
                size=original_shape,
                mode="bilinear",
                align_corners=False,
            )
            embeddings = F.interpolate(
                outputs["embeddings"],
                size=original_shape,
                mode="bilinear",
                align_corners=False,
            )

            semantic_pred = logits.argmax(dim=1)[0].cpu().numpy()
            embeddings_np = embeddings[0].cpu().numpy()  # (D, H, W)

        inference_time = (perf_counter() - start) * 1000

        # Post-process to get lane instances
        postprocess_result = self._postprocessor.process(
            semantic_pred, embeddings_np, lane_class_id=lane_class_id
        )

        # Extract crosswalk mask (assuming class 3)
        crosswalk_mask = (semantic_pred == 3).astype(np.uint8)

        return {
            "semantic_mask": postprocess_result["semantic_mask"].astype(np.uint8),
            "lane_instances": postprocess_result["lane_instances"].astype(np.uint8),
            "crosswalk_mask": crosswalk_mask,
            "lane_count": postprocess_result["lane_count"],
            "lane_centers": postprocess_result["lane_centers"],
            "inference_time_ms": inference_time,
        }

    def predict_batch(
        self, images: list[np.ndarray], lane_class_id: int = 1
    ) -> list[dict]:
        """Predict lane instance segmentation for batch of images.

        Args:
            images: List of RGB images.
            lane_class_id: Class ID for lane markings.

        Returns:
            List of prediction dictionaries.
        """
        return [self.predict(img, lane_class_id) for img in images]
