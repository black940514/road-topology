"""Core utilities and base classes for Road Topology Segmentation."""
from road_topology.core.exceptions import (
    RoadTopologyError,
    ConfigurationError,
    ModelLoadError,
    DetectionError,
    TrackingError,
    SegmentationError,
    PseudoLabelError,
    InferenceError,
    VideoProcessingError,
    CVATError,
    ExportError,
    ValidationError,
)
from road_topology.core.device import (
    DeviceInfo,
    DeviceNotAvailableError,
    get_device,
    get_device_info,
    get_autocast_context,
    get_grad_scaler,
    empty_cache,
    to_device,
)

__all__ = [
    # Exceptions
    "RoadTopologyError",
    "ConfigurationError",
    "ModelLoadError",
    "DetectionError",
    "TrackingError",
    "SegmentationError",
    "PseudoLabelError",
    "InferenceError",
    "VideoProcessingError",
    "CVATError",
    "ExportError",
    "ValidationError",
    # Device utilities
    "DeviceInfo",
    "DeviceNotAvailableError",
    "get_device",
    "get_device_info",
    "get_autocast_context",
    "get_grad_scaler",
    "empty_cache",
    "to_device",
]
