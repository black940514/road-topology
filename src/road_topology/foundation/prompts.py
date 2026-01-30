"""Prompting strategies for foundation models."""
from __future__ import annotations

import numpy as np

from road_topology.core.types import BoundingBox


def generate_grid_points(
    image_shape: tuple[int, int],
    grid_size: int = 32,
) -> np.ndarray:
    """Generate grid of points for automatic segmentation.

    Args:
        image_shape: Image (height, width).
        grid_size: Points per side.

    Returns:
        Points array (N, 2) with (x, y) coordinates.
    """
    h, w = image_shape
    y_coords = np.linspace(0, h - 1, grid_size)
    x_coords = np.linspace(0, w - 1, grid_size)

    xx, yy = np.meshgrid(x_coords, y_coords)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    return points


def bbox_to_prompt(bbox: BoundingBox, expansion: float = 0.0) -> tuple[int, int, int, int]:
    """Convert BoundingBox to SAM prompt format.

    Args:
        bbox: Bounding box object.
        expansion: Ratio to expand box (0.1 = 10%).

    Returns:
        Box coordinates (x1, y1, x2, y2).
    """
    if expansion > 0:
        bbox = bbox.expand(expansion)

    return (
        int(bbox.x1),
        int(bbox.y1),
        int(bbox.x2),
        int(bbox.y2),
    )


def center_point_prompt(bbox: BoundingBox) -> tuple[np.ndarray, np.ndarray]:
    """Generate center point prompt from bounding box.

    Args:
        bbox: Bounding box.

    Returns:
        Tuple of (points, labels) where points is (1, 2) and labels is (1,).
    """
    cx, cy = bbox.center
    points = np.array([[cx, cy]])
    labels = np.array([1])  # 1 = foreground

    return points, labels


def multi_point_prompt(
    bboxes: list[BoundingBox],
    points_per_box: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate multiple point prompts from bounding boxes.

    Args:
        bboxes: List of bounding boxes.
        points_per_box: Number of points to sample per box.

    Returns:
        Tuple of (points, labels) arrays.
    """
    all_points = []
    all_labels = []

    for bbox in bboxes:
        if points_per_box == 1:
            # Use center point
            cx, cy = bbox.center
            all_points.append([cx, cy])
            all_labels.append(1)
        else:
            # Sample multiple points within box
            x_coords = np.linspace(bbox.x1, bbox.x2, points_per_box + 2)[1:-1]
            y_coords = np.linspace(bbox.y1, bbox.y2, points_per_box + 2)[1:-1]

            for x, y in zip(x_coords, y_coords):
                all_points.append([x, y])
                all_labels.append(1)

    return np.array(all_points), np.array(all_labels)
