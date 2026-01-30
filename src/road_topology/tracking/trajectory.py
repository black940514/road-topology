"""Trajectory accumulation and processing."""
from __future__ import annotations

import numpy as np

from road_topology.core.logging import get_logger
from road_topology.core.types import Track, Trajectory

logger = get_logger(__name__)


class TrajectoryAccumulator:
    """Accumulates and processes vehicle trajectories.

    Collects tracks over video duration and provides
    trajectory extraction with smoothing and filtering.

    Args:
        min_length: Minimum trajectory points to keep.
        smooth_window: Window size for smoothing.
        max_jump: Maximum allowed jump between consecutive points.
    """

    def __init__(
        self,
        min_length: int = 10,
        smooth_window: int = 5,
        max_jump: float = 100.0,
    ) -> None:
        self.min_length = min_length
        self.smooth_window = smooth_window
        self.max_jump = max_jump
        self._trajectories: list[Trajectory] = []

    def add_tracks(self, tracks: list[Track]) -> None:
        """Add tracks to accumulator.

        Args:
            tracks: List of completed tracks.
        """
        for track in tracks:
            if len(track) < self.min_length:
                continue

            trajectory = track.trajectory()

            # Remove outliers (sudden jumps)
            trajectory = self._remove_outliers(trajectory)

            if len(trajectory) >= self.min_length:
                self._trajectories.append(trajectory)

        logger.debug(f"Accumulated {len(self._trajectories)} trajectories")

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """Add a single trajectory.

        Args:
            trajectory: Trajectory to add.
        """
        if len(trajectory) >= self.min_length:
            trajectory = self._remove_outliers(trajectory)
            if len(trajectory) >= self.min_length:
                self._trajectories.append(trajectory)

    def _remove_outliers(self, trajectory: Trajectory) -> Trajectory:
        """Remove points with sudden jumps.

        Args:
            trajectory: Input trajectory.

        Returns:
            Filtered trajectory.
        """
        if len(trajectory) < 2:
            return trajectory

        points = trajectory.points
        confidences = trajectory.confidence_scores

        # Calculate distances between consecutive points
        diffs = np.diff(points, axis=0)
        distances = np.linalg.norm(diffs, axis=1)

        # Keep points where jump is within threshold
        valid_mask = np.ones(len(points), dtype=bool)
        valid_mask[1:] = distances <= self.max_jump

        # Also check backward (keep previous point if next is invalid)
        for i in range(len(valid_mask) - 1, 0, -1):
            if not valid_mask[i] and i < len(distances) and distances[i-1] > self.max_jump:
                valid_mask[i-1] = False

        filtered_points = points[valid_mask]
        filtered_confidences = confidences[valid_mask]

        return Trajectory(
            points=filtered_points,
            confidence_scores=filtered_confidences,
            track_id=trajectory.track_id,
        )

    def get_trajectories(
        self,
        smooth: bool = True,
    ) -> list[Trajectory]:
        """Get accumulated trajectories.

        Args:
            smooth: Whether to apply smoothing.

        Returns:
            List of trajectories.
        """
        if smooth:
            return [t.smooth(self.smooth_window) for t in self._trajectories]
        return self._trajectories.copy()

    def filter_short(self, min_points: int) -> "TrajectoryAccumulator":
        """Filter out short trajectories.

        Args:
            min_points: Minimum number of points.

        Returns:
            New accumulator with filtered trajectories.
        """
        new_acc = TrajectoryAccumulator(
            min_length=min_points,
            smooth_window=self.smooth_window,
            max_jump=self.max_jump,
        )
        new_acc._trajectories = [
            t for t in self._trajectories if len(t) >= min_points
        ]
        return new_acc

    def to_numpy(self) -> np.ndarray:
        """Convert all trajectories to padded numpy array.

        Returns:
            Array of shape (N_trajectories, max_points, 2).
            Shorter trajectories are padded with zeros.
        """
        if not self._trajectories:
            return np.empty((0, 0, 2))

        max_len = max(len(t) for t in self._trajectories)
        result = np.zeros((len(self._trajectories), max_len, 2))

        for i, traj in enumerate(self._trajectories):
            result[i, :len(traj), :] = traj.points

        return result

    def to_points_array(self) -> np.ndarray:
        """Concatenate all trajectory points.

        Returns:
            Array of shape (total_points, 2).
        """
        if not self._trajectories:
            return np.empty((0, 2))

        return np.concatenate([t.points for t in self._trajectories], axis=0)

    def clear(self) -> None:
        """Clear all accumulated trajectories."""
        self._trajectories.clear()

    def __len__(self) -> int:
        """Return number of trajectories."""
        return len(self._trajectories)

    @property
    def total_points(self) -> int:
        """Total number of points across all trajectories."""
        return sum(len(t) for t in self._trajectories)
