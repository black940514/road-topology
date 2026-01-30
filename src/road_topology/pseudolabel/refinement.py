"""Mask refinement operations for pseudo-label post-processing."""
from __future__ import annotations

import cv2
import numpy as np

from road_topology.core.logging import get_logger

logger = get_logger(__name__)


def morphological_refinement(
    mask: np.ndarray,
    kernel_size: int = 5,
    iterations: int = 1,
) -> np.ndarray:
    """Apply morphological operations to clean up mask.

    Args:
        mask: Binary mask to refine.
        kernel_size: Size of morphological kernel.
        iterations: Number of morphological iterations.

    Returns:
        Refined binary mask.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Close small gaps
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    # Remove small noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=iterations)

    return opened


def remove_small_components(
    mask: np.ndarray,
    min_area: int = 100,
) -> np.ndarray:
    """Remove small connected components from mask.

    Args:
        mask: Binary mask to filter.
        min_area: Minimum component area to keep.

    Returns:
        Filtered binary mask.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Create output mask
    filtered = np.zeros_like(mask)

    # Keep components above minimum area (skip background label 0)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered[labels == label] = 1

    logger.debug(f"Removed {num_labels - 1 - (filtered > 0).sum()} small components")

    return filtered


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes in binary mask.

    Args:
        mask: Binary mask with holes.

    Returns:
        Mask with holes filled.
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Fill all contours
    filled = mask.copy()
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)

    return filled


def smooth_boundaries(
    mask: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """Smooth mask boundaries using Gaussian blur.

    Args:
        mask: Binary mask to smooth.
        kernel_size: Size of Gaussian kernel (must be odd).

    Returns:
        Smoothed binary mask.
    """
    # Convert to float for blurring
    mask_float = mask.astype(np.float32)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(mask_float, (kernel_size, kernel_size), 0)

    # Re-threshold
    smoothed = (blurred > 0.5).astype(np.uint8)

    return smoothed
