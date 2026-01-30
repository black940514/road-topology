"""Trajectory-to-mask conversion for pseudo-label generation."""
from __future__ import annotations

import cv2
import numpy as np

from road_topology.core.logging import get_logger
from road_topology.core.types import CLASS_NAMES, Trajectory

logger = get_logger(__name__)

# Class indices
ROAD = CLASS_NAMES.index("road")


class TrajectoryMaskBuilder:
    """Build segmentation masks from vehicle trajectories.

    Converts trajectory points to filled road regions
    by drawing thick polylines and accumulating density.

    Args:
        frame_shape: Output mask shape (height, width).
        trajectory_width: Width of trajectory stroke (vehicle width estimate).
    """

    def __init__(
        self,
        frame_shape: tuple[int, int],
        trajectory_width: int = 50,
    ) -> None:
        self.height, self.width = frame_shape
        self.trajectory_width = trajectory_width

        # Accumulation buffers
        self._density_map: np.ndarray = np.zeros((self.height, self.width), dtype=np.float32)
        self._trajectory_count = 0

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """Add a trajectory to the mask builder.

        Args:
            trajectory: Vehicle trajectory to add.
        """
        if len(trajectory) < 2:
            return

        # Convert points to integer coordinates
        points = trajectory.points.astype(np.int32)

        # Clip to frame bounds
        points[:, 0] = np.clip(points[:, 0], 0, self.width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, self.height - 1)

        # Draw thick polyline on temporary mask
        temp_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.polylines(
            temp_mask,
            [points],
            isClosed=False,
            color=1,
            thickness=self.trajectory_width,
        )

        # Optionally weight by confidence
        weight = trajectory.mean_confidence() if len(trajectory.confidence_scores) > 0 else 1.0

        # Accumulate
        self._density_map += temp_mask.astype(np.float32) * weight
        self._trajectory_count += 1

    def add_trajectories(self, trajectories: list[Trajectory]) -> None:
        """Add multiple trajectories.

        Args:
            trajectories: List of trajectories to add.
        """
        for traj in trajectories:
            self.add_trajectory(traj)

        logger.debug(f"Added {len(trajectories)} trajectories, total: {self._trajectory_count}")

    def build_mask(self, threshold: float = 0.1) -> np.ndarray:
        """Build binary road mask.

        Args:
            threshold: Density threshold for road classification (0-1).

        Returns:
            Binary mask (H, W) with 1 for road, 0 for background.
        """
        if self._trajectory_count == 0:
            return np.zeros((self.height, self.width), dtype=np.uint8)

        # Normalize density map
        normalized = self._density_map / self._trajectory_count

        # Apply threshold
        mask = (normalized > threshold).astype(np.uint8)

        return mask

    def build_semantic_mask(self, threshold: float = 0.1) -> np.ndarray:
        """Build semantic segmentation mask with class indices.

        Args:
            threshold: Density threshold for road classification.

        Returns:
            Semantic mask (H, W) with class indices.
        """
        binary_mask = self.build_mask(threshold)

        # Convert to semantic mask (0=background, 1=road)
        semantic_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        semantic_mask[binary_mask > 0] = ROAD

        return semantic_mask

    def build_confidence_map(self) -> np.ndarray:
        """Build per-pixel confidence map.

        Returns:
            Confidence map (H, W) with values [0, 1].
            Higher values = more trajectories passed through.
        """
        if self._trajectory_count == 0:
            return np.zeros((self.height, self.width), dtype=np.float32)

        # Normalize to [0, 1] based on maximum density
        confidence = self._density_map / (self._density_map.max() + 1e-6)

        return confidence

    def reset(self) -> None:
        """Reset accumulation buffers."""
        self._density_map.fill(0)
        self._trajectory_count = 0

    @property
    def trajectory_count(self) -> int:
        """Number of accumulated trajectories."""
        return self._trajectory_count

    @property
    def coverage_ratio(self) -> float:
        """Ratio of pixels covered by at least one trajectory."""
        if self._trajectory_count == 0:
            return 0.0
        return (self._density_map > 0).sum() / (self.height * self.width)
