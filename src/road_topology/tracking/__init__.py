"""Multi-object tracking module for Road Topology Segmentation.

This module provides:
- VehicleTracker: ByteTrack-based multi-object tracking
- TrajectoryProcessor: Trajectory smoothing and filtering
- Tracking models and utilities
"""

from road_topology.tracking.tracker import VehicleTracker
from road_topology.tracking.trajectory import TrajectoryProcessor

__all__ = ["VehicleTracker", "TrajectoryProcessor"]
