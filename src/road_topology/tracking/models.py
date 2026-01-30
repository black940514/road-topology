"""Additional tracking models and data structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class TrackState(Enum):
    """Track lifecycle states."""
    NEW = "new"           # Just created
    TRACKED = "tracked"   # Successfully tracked
    LOST = "lost"         # Temporarily lost
    REMOVED = "removed"   # Permanently removed


@dataclass
class TrackMetrics:
    """Metrics for a track.

    Attributes:
        track_id: Track identifier.
        duration: Track duration in frames.
        avg_confidence: Average detection confidence.
        total_detections: Total number of detections.
        occlusion_count: Number of times track was occluded.
        avg_velocity: Average velocity (pixels/frame).
    """
    track_id: int
    duration: int = 0
    avg_confidence: float = 0.0
    total_detections: int = 0
    occlusion_count: int = 0
    avg_velocity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "track_id": self.track_id,
            "duration": self.duration,
            "avg_confidence": self.avg_confidence,
            "total_detections": self.total_detections,
            "occlusion_count": self.occlusion_count,
            "avg_velocity": self.avg_velocity,
        }


@dataclass
class TrackingStats:
    """Overall tracking statistics.

    Attributes:
        total_tracks: Total number of tracks created.
        active_tracks: Number of currently active tracks.
        completed_tracks: Number of completed tracks.
        avg_track_duration: Average track duration in frames.
        avg_track_confidence: Average confidence across all tracks.
        total_frames_processed: Total frames processed.
    """
    total_tracks: int = 0
    active_tracks: int = 0
    completed_tracks: int = 0
    avg_track_duration: float = 0.0
    avg_track_confidence: float = 0.0
    total_frames_processed: int = 0
    track_metrics: dict[int, TrackMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tracks": self.total_tracks,
            "active_tracks": self.active_tracks,
            "completed_tracks": self.completed_tracks,
            "avg_track_duration": self.avg_track_duration,
            "avg_track_confidence": self.avg_track_confidence,
            "total_frames_processed": self.total_frames_processed,
            "track_metrics": {
                tid: metrics.to_dict()
                for tid, metrics in self.track_metrics.items()
            },
        }


@dataclass
class KalmanState:
    """Kalman filter state for a track.

    Used for motion prediction and smoothing.

    Attributes:
        mean: State mean vector [x, y, vx, vy].
        covariance: State covariance matrix (4, 4).
        track_id: Associated track ID.
    """
    mean: np.ndarray
    covariance: np.ndarray
    track_id: int

    def __post_init__(self) -> None:
        """Validate Kalman state."""
        if self.mean.shape != (4,):
            raise ValueError(f"mean must have shape (4,), got {self.mean.shape}")
        if self.covariance.shape != (4, 4):
            raise ValueError(f"covariance must have shape (4, 4), got {self.covariance.shape}")

    def predict(self, dt: float = 1.0) -> "KalmanState":
        """Predict next state.

        Args:
            dt: Time step (default 1 frame).

        Returns:
            Predicted KalmanState.
        """
        # Simple constant velocity model
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        mean = F @ self.mean
        covariance = F @ self.covariance @ F.T

        # Add process noise
        Q = np.eye(4) * 0.01
        covariance += Q

        return KalmanState(mean=mean, covariance=covariance, track_id=self.track_id)

    @classmethod
    def from_detection(cls, x: float, y: float, track_id: int) -> "KalmanState":
        """Initialize Kalman state from detection.

        Args:
            x: X coordinate.
            y: Y coordinate.
            track_id: Track identifier.

        Returns:
            Initialized KalmanState.
        """
        mean = np.array([x, y, 0.0, 0.0])  # Initial velocity = 0
        covariance = np.eye(4) * 10.0  # High initial uncertainty
        return cls(mean=mean, covariance=covariance, track_id=track_id)


@dataclass
class AssociationMatrix:
    """Cost matrix for data association.

    Attributes:
        costs: Cost matrix of shape (num_tracks, num_detections).
        track_ids: Track IDs corresponding to rows.
        detection_ids: Detection IDs corresponding to columns.
    """
    costs: np.ndarray
    track_ids: list[int]
    detection_ids: list[int]

    def __post_init__(self) -> None:
        """Validate association matrix."""
        if self.costs.ndim != 2:
            raise ValueError(f"costs must be 2D, got shape {self.costs.shape}")
        if self.costs.shape[0] != len(self.track_ids):
            raise ValueError(
                f"costs rows ({self.costs.shape[0]}) must match "
                f"track_ids length ({len(self.track_ids)})"
            )
        if self.costs.shape[1] != len(self.detection_ids):
            raise ValueError(
                f"costs columns ({self.costs.shape[1]}) must match "
                f"detection_ids length ({len(self.detection_ids)})"
            )

    def get_matches(self, threshold: float = 0.5) -> list[tuple[int, int]]:
        """Get matched pairs using greedy assignment.

        Args:
            threshold: Maximum cost threshold for valid matches.

        Returns:
            List of (track_id, detection_id) tuples.
        """
        matches = []
        costs = self.costs.copy()

        while True:
            # Find minimum cost
            if costs.size == 0:
                break

            min_idx = np.unravel_index(np.argmin(costs), costs.shape)
            min_cost = costs[min_idx]

            if min_cost > threshold:
                break

            track_id = self.track_ids[min_idx[0]]
            det_id = self.detection_ids[min_idx[1]]
            matches.append((track_id, det_id))

            # Remove matched row and column
            costs = np.delete(costs, min_idx[0], axis=0)
            costs = np.delete(costs, min_idx[1], axis=1)
            del self.track_ids[min_idx[0]]
            del self.detection_ids[min_idx[1]]

        return matches
