"""YOLOv8 vehicle detector wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO

from road_topology.core.config import DetectionConfig
from road_topology.core.device import empty_cache
from road_topology.core.exceptions import DetectionError, ModelLoadError
from road_topology.core.logging import get_logger
from road_topology.core.types import BoundingBox

logger = get_logger(__name__)


class VehicleDetector:
    """YOLOv8-based vehicle detector.

    Wraps ultralytics YOLO model for vehicle detection,
    filtering results to only vehicle classes.

    Args:
        model_path: Path to YOLOv8 weights or model name.
        config: Detection configuration.
    """

    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        model_path: str | Path = "yolov8m.pt",
        config: DetectionConfig | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.config = config or DetectionConfig()
        self._model: YOLO | None = None

        # Filter to configured vehicle classes
        self.target_classes = set(
            self.config.vehicle_classes
            if hasattr(self.config, 'vehicle_classes')
            else self.VEHICLE_CLASSES.keys()
        )

    @property
    def model(self) -> YOLO:
        """Lazy load model on first use."""
        if self._model is None:
            try:
                logger.info("Loading YOLOv8 model", model_path=str(self.model_path))
                self._model = YOLO(str(self.model_path))
            except Exception as e:
                raise ModelLoadError(
                    f"Failed to load YOLO model: {e}",
                    model_path=str(self.model_path),
                )
        return self._model

    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        """Detect vehicles in a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of detected vehicle bounding boxes.
        """
        try:
            results = self.model(
                frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                verbose=False,
            )

            boxes = []
            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    class_id = int(box.cls.item())

                    # Filter to vehicle classes only
                    if class_id not in self.target_classes:
                        continue

                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf.item())

                    boxes.append(BoundingBox(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                        confidence=conf,
                        class_id=class_id,
                        class_name=self.VEHICLE_CLASSES.get(class_id),
                    ))

            return boxes

        except Exception as e:
            raise DetectionError(f"Detection failed: {e}")

    def detect_batch(
        self,
        frames: list[np.ndarray],
    ) -> list[list[BoundingBox]]:
        """Detect vehicles in multiple frames.

        Args:
            frames: List of BGR images.

        Returns:
            List of detection results per frame.
        """
        try:
            results = self.model(
                frames,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                verbose=False,
            )

            all_boxes = []
            for result in results:
                frame_boxes = []
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        if class_id not in self.target_classes:
                            continue

                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf.item())

                        frame_boxes.append(BoundingBox(
                            x1=float(xyxy[0]),
                            y1=float(xyxy[1]),
                            x2=float(xyxy[2]),
                            y2=float(xyxy[3]),
                            confidence=conf,
                            class_id=class_id,
                            class_name=self.VEHICLE_CLASSES.get(class_id),
                        ))
                all_boxes.append(frame_boxes)

            return all_boxes

        except Exception as e:
            raise DetectionError(f"Batch detection failed: {e}")

    def cleanup(self) -> None:
        """Release GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            empty_cache()
            logger.info("Released detector GPU memory")

    def __enter__(self) -> "VehicleDetector":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit with cleanup."""
        self.cleanup()
