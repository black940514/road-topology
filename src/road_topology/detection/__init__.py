"""Vehicle detection module for Road Topology Segmentation.

This module provides YOLOv8-based vehicle detection with batch processing
and video frame iteration utilities.
"""
from __future__ import annotations

from road_topology.detection.detector import VehicleDetector
from road_topology.detection.models import DetectionBatch, DetectionResult

__all__ = [
    "VehicleDetector",
    "DetectionResult",
    "DetectionBatch",
]
