"""Single image and batch prediction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from road_topology.core.device import get_device
from road_topology.core.exceptions import InferenceError, ModelLoadError
from road_topology.core.logging import get_logger
from road_topology.core.types import PredictionResult
from road_topology.segmentation.models import SegFormerModel
from road_topology.segmentation.transforms import get_val_transforms

logger = get_logger(__name__)


class RoadTopologyPredictor:
    """Production inference predictor.

    Loads a trained model and provides efficient single-image
    and batch inference with preprocessing.

    Args:
        model_path: Path to model checkpoint.
        model_type: Model architecture (segformer, mask2former).
        device: Inference device.
        image_size: Input image size for preprocessing.
    """

    def __init__(
        self,
        model_path: Path,
        model_type: str = "segformer",
        device: str = "auto",
        image_size: tuple[int, int] = (512, 512),
    ) -> None:
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.device = str(get_device(device))
        self.image_size = image_size

        self._model = None
        self._transforms = get_val_transforms(image_size[0], normalize=False)

    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self) -> None:
        """Load model from checkpoint."""
        try:
            logger.info(f"Loading model from {self.model_path}")

            if self.model_type == "segformer":
                self._model = SegFormerModel.load(self.model_path, self.device)
            else:
                raise ModelLoadError(f"Unknown model type: {self.model_type}")

            self._model.eval()
            logger.info("Model loaded successfully")

        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}")

    def warmup(self, input_shape: tuple[int, int, int] = (3, 512, 512)) -> None:
        """Warmup model with dummy input."""
        dummy = torch.randn(1, *input_shape).to(self.device)
        with torch.no_grad():
            self.model.predict(dummy)
        logger.info("Model warmed up")

    def predict(self, image: np.ndarray) -> PredictionResult:
        """Predict segmentation for single image.

        Args:
            image: RGB image (H, W, 3) or BGR image.

        Returns:
            PredictionResult with mask, confidence, and timing.
        """
        original_shape = image.shape[:2]

        # Preprocess
        transformed = self._transforms(image=image)
        input_tensor = torch.from_numpy(
            transformed["image"].transpose(2, 0, 1)
        ).float().unsqueeze(0) / 255.0
        input_tensor = input_tensor.to(self.device)

        # Inference
        start = perf_counter()
        with torch.no_grad():
            probs = self.model.predict(input_tensor, return_probs=True)
        inference_time = (perf_counter() - start) * 1000

        # Postprocess
        probs = F.interpolate(
            probs.unsqueeze(0),
            size=original_shape,
            mode="bilinear",
            align_corners=False,
        )[0]

        mask = probs.argmax(dim=0).cpu().numpy()
        confidence = probs.max(dim=0)[0].cpu().numpy()
        class_probs = probs.cpu().numpy().transpose(1, 2, 0)

        return PredictionResult(
            mask=mask.astype(np.uint8),
            confidence=confidence,
            class_probabilities=class_probs,
            inference_time_ms=inference_time,
        )

    def predict_batch(
        self,
        images: list[np.ndarray],
    ) -> list[PredictionResult]:
        """Predict segmentation for batch of images.

        Args:
            images: List of RGB images.

        Returns:
            List of PredictionResults.
        """
        return [self.predict(img) for img in images]
