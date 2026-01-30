"""Confidence estimation for pseudo-labels."""
from __future__ import annotations

import numpy as np

from road_topology.core.logging import get_logger

logger = get_logger(__name__)


def estimate_pixel_confidence(
    density_map: np.ndarray,
    trajectory_count: int,
    min_trajectories: int = 5,
) -> np.ndarray:
    """Estimate per-pixel confidence based on trajectory density.

    Higher confidence when more trajectories pass through a pixel.

    Args:
        density_map: Accumulated trajectory density.
        trajectory_count: Total number of trajectories.
        min_trajectories: Minimum trajectories for full confidence.

    Returns:
        Confidence map (H, W) with values [0, 1].
    """
    if trajectory_count == 0:
        return np.zeros_like(density_map, dtype=np.float32)

    # Normalize by trajectory count
    normalized = density_map / trajectory_count

    # Scale confidence based on minimum threshold
    confidence = np.clip(normalized * trajectory_count / min_trajectories, 0, 1)

    return confidence


def estimate_region_confidence(
    mask: np.ndarray,
    confidence_map: np.ndarray,
) -> float:
    """Estimate overall confidence for masked region.

    Args:
        mask: Binary mask defining region.
        confidence_map: Per-pixel confidence map.

    Returns:
        Mean confidence in masked region [0, 1].
    """
    if mask.sum() == 0:
        return 0.0

    # Average confidence over masked pixels
    return float(confidence_map[mask > 0].mean())


def filter_by_confidence(
    mask: np.ndarray,
    confidence_map: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Filter mask by confidence threshold.

    Args:
        mask: Binary mask to filter.
        confidence_map: Per-pixel confidence map.
        threshold: Minimum confidence to keep.

    Returns:
        Filtered binary mask.
    """
    filtered = mask.copy()
    filtered[confidence_map < threshold] = 0

    removed_pixels = mask.sum() - filtered.sum()
    logger.debug(f"Removed {removed_pixels} low-confidence pixels")

    return filtered


def adaptive_threshold(
    confidence_map: np.ndarray,
    percentile: float = 50.0,
) -> float:
    """Compute adaptive confidence threshold.

    Args:
        confidence_map: Per-pixel confidence map.
        percentile: Percentile for threshold (0-100).

    Returns:
        Confidence threshold value.
    """
    # Ignore zero-confidence pixels
    nonzero = confidence_map[confidence_map > 0]

    if len(nonzero) == 0:
        return 0.0

    threshold = float(np.percentile(nonzero, percentile))
    logger.debug(f"Adaptive threshold at {percentile}th percentile: {threshold:.3f}")

    return threshold
