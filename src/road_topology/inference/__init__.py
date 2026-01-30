"""Inference module for road topology segmentation."""
from __future__ import annotations

from road_topology.inference.predictor import RoadTopologyPredictor
from road_topology.inference.video_processor import VideoProcessor

__all__ = ["RoadTopologyPredictor", "VideoProcessor"]
