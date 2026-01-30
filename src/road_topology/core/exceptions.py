"""Custom exception hierarchy for Road Topology Segmentation."""
from __future__ import annotations

from typing import Any


class RoadTopologyError(Exception):
    """Base exception for Road Topology Segmentation.

    All custom exceptions inherit from this class.

    Args:
        message: Human-readable error message.
        context: Additional context dictionary for structured logging.
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            return f"{self.message} | Context: {self.context}"
        return self.message


class ConfigurationError(RoadTopologyError):
    """Raised when configuration is invalid or missing."""
    pass


class ModelLoadError(RoadTopologyError):
    """Raised when a model fails to load.

    Args:
        message: Error message.
        model_path: Path to the model that failed to load.
        context: Additional context.
    """

    def __init__(
        self,
        message: str,
        model_path: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if model_path:
            ctx["model_path"] = model_path
        super().__init__(message, ctx)
        self.model_path = model_path


class DetectionError(RoadTopologyError):
    """Raised when object detection fails."""
    pass


class TrackingError(RoadTopologyError):
    """Raised when object tracking fails."""
    pass


class SegmentationError(RoadTopologyError):
    """Raised when segmentation fails."""
    pass


class PseudoLabelError(RoadTopologyError):
    """Raised when pseudo-label generation fails."""
    pass


class InferenceError(RoadTopologyError):
    """Raised when inference fails."""
    pass


class VideoProcessingError(RoadTopologyError):
    """Raised when video processing fails.

    Args:
        message: Error message.
        video_path: Path to the video that failed.
        frame_id: Frame number where the error occurred.
        context: Additional context.
    """

    def __init__(
        self,
        message: str,
        video_path: str | None = None,
        frame_id: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if video_path:
            ctx["video_path"] = video_path
        if frame_id is not None:
            ctx["frame_id"] = frame_id
        super().__init__(message, ctx)
        self.video_path = video_path
        self.frame_id = frame_id


class CVATError(RoadTopologyError):
    """Raised when CVAT integration fails."""
    pass


class ExportError(RoadTopologyError):
    """Raised when model export fails."""
    pass


class ValidationError(RoadTopologyError):
    """Raised when data validation fails."""
    pass
