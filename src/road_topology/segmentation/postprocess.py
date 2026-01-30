"""Post-processing for lane instance segmentation."""
from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.cluster import DBSCAN, MeanShift

from road_topology.core.logging import get_logger

logger = get_logger(__name__)


class LanePostProcessor:
    """Post-processing for lane instance segmentation.

    Converts pixel embeddings and semantic predictions into lane instances
    by clustering embeddings and ordering lanes spatially.

    Steps:
    1. Extract embeddings for lane_marking pixels
    2. Cluster using mean-shift or DBSCAN
    3. Order lanes left-to-right
    4. Assign instance IDs

    Args:
        clustering_method: Clustering algorithm ("meanshift" or "dbscan").
        bandwidth: Bandwidth for mean-shift clustering.
        eps: DBSCAN epsilon parameter (max distance between samples).
        min_samples: DBSCAN min_samples parameter.
        min_pixels: Minimum pixels per lane instance to keep.
    """

    def __init__(
        self,
        clustering_method: Literal["meanshift", "dbscan"] = "meanshift",
        bandwidth: float = 0.5,
        eps: float = 0.3,
        min_samples: int = 10,
        min_pixels: int = 100,
    ) -> None:
        self.clustering_method = clustering_method
        self.bandwidth = bandwidth
        self.eps = eps
        self.min_samples = min_samples
        self.min_pixels = min_pixels

    def process(
        self,
        semantic_pred: np.ndarray,
        embeddings: np.ndarray,
        lane_class_id: int = 1,
    ) -> dict[str, np.ndarray | int]:
        """Process semantic predictions and embeddings into lane instances.

        Args:
            semantic_pred: (H, W) class predictions.
            embeddings: (D, H, W) pixel embeddings (D=32 typically).
            lane_class_id: Class ID for lane markings (default 1).

        Returns:
            Dictionary with:
                - semantic_mask: (H, W) semantic segmentation
                - lane_instances: (H, W) instance IDs (0=background)
                - lane_count: Number of detected lanes
                - lane_centers: (N, 2) array of lane x-centroids and y-centroids
        """
        h, w = semantic_pred.shape
        d = embeddings.shape[0]

        # Initialize output
        lane_instances = np.zeros((h, w), dtype=np.int32)

        # Extract lane marking pixels
        lane_mask = semantic_pred == lane_class_id
        num_lane_pixels = lane_mask.sum()

        if num_lane_pixels < self.min_pixels:
            logger.warning(f"Only {num_lane_pixels} lane pixels detected, skipping clustering")
            return {
                "semantic_mask": semantic_pred,
                "lane_instances": lane_instances,
                "lane_count": 0,
                "lane_centers": np.array([]),
            }

        # Extract embeddings for lane pixels (N, D)
        lane_embeddings = embeddings[:, lane_mask].T  # (N_pixels, D)

        # Cluster embeddings
        cluster_labels = self._cluster_embeddings(lane_embeddings)

        # Filter small clusters
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        valid_labels = unique_labels[counts >= self.min_pixels]

        if len(valid_labels) == 0:
            logger.warning("No valid lane clusters found after filtering")
            return {
                "semantic_mask": semantic_pred,
                "lane_instances": lane_instances,
                "lane_count": 0,
                "lane_centers": np.array([]),
            }

        # Map cluster labels to instance IDs
        label_to_instance = {label: idx + 1 for idx, label in enumerate(valid_labels)}

        # Assign instance IDs to pixels
        lane_instances_flat = np.zeros(num_lane_pixels, dtype=np.int32)
        for label, instance_id in label_to_instance.items():
            lane_instances_flat[cluster_labels == label] = instance_id

        lane_instances[lane_mask] = lane_instances_flat

        # Compute lane centers for ordering
        lane_centers = []
        for instance_id in range(1, len(valid_labels) + 1):
            instance_mask = lane_instances == instance_id
            y_coords, x_coords = np.where(instance_mask)
            x_center = x_coords.mean()
            y_center = y_coords.mean()
            lane_centers.append([x_center, y_center])

        lane_centers = np.array(lane_centers)

        # Order lanes left-to-right
        lane_instances = self._order_lanes_left_to_right(
            lane_instances, lane_centers, w
        )

        # Recompute centers after reordering
        final_centers = []
        for instance_id in range(1, len(valid_labels) + 1):
            instance_mask = lane_instances == instance_id
            y_coords, x_coords = np.where(instance_mask)
            if len(x_coords) > 0:
                x_center = x_coords.mean()
                y_center = y_coords.mean()
                final_centers.append([x_center, y_center])

        return {
            "semantic_mask": semantic_pred,
            "lane_instances": lane_instances,
            "lane_count": len(valid_labels),
            "lane_centers": np.array(final_centers),
        }

    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster pixel embeddings.

        Args:
            embeddings: (N, D) pixel embeddings.

        Returns:
            (N,) cluster labels.
        """
        if self.clustering_method == "meanshift":
            clusterer = MeanShift(bandwidth=self.bandwidth, bin_seeding=True)
            labels = clusterer.fit_predict(embeddings)
        elif self.clustering_method == "dbscan":
            clusterer = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = clusterer.fit_predict(embeddings)
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")

        return labels

    def _order_lanes_left_to_right(
        self,
        lane_instances: np.ndarray,
        lane_centers: np.ndarray,
        image_width: int,
    ) -> np.ndarray:
        """Order lane instances from left to right by x-centroid.

        Args:
            lane_instances: (H, W) instance segmentation.
            lane_centers: (N, 2) array of [x_center, y_center].
            image_width: Image width.

        Returns:
            (H, W) reordered instance segmentation.
        """
        if len(lane_centers) == 0:
            return lane_instances

        # Sort by x-coordinate (left to right)
        x_coords = lane_centers[:, 0]
        sorted_indices = np.argsort(x_coords)

        # Create mapping from old instance ID to new instance ID
        reorder_map = np.zeros(len(sorted_indices) + 1, dtype=np.int32)
        for new_id, old_idx in enumerate(sorted_indices, start=1):
            old_id = old_idx + 1  # Instance IDs start at 1
            reorder_map[old_id] = new_id

        # Apply reordering
        reordered = np.zeros_like(lane_instances)
        for old_id, new_id in enumerate(reorder_map):
            if old_id > 0:  # Skip background (0)
                reordered[lane_instances == old_id] = new_id

        return reordered
