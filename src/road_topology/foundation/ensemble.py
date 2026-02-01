"""Ensemble strategies for combining multiple foundation model outputs."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label

from road_topology.core.logging import get_logger

logger = get_logger(__name__)


def union_masks(masks: list[np.ndarray]) -> np.ndarray:
    """Combine masks using union (logical OR).

    Args:
        masks: List of binary masks (H, W).

    Returns:
        Combined mask (H, W).
    """
    if not masks:
        raise ValueError("No masks to combine")

    result = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        result |= mask.astype(bool)

    return result.astype(np.uint8)


def intersection_masks(masks: list[np.ndarray]) -> np.ndarray:
    """Combine masks using intersection (logical AND).

    Args:
        masks: List of binary masks (H, W).

    Returns:
        Combined mask (H, W).
    """
    if not masks:
        raise ValueError("No masks to combine")

    result = np.ones_like(masks[0], dtype=bool)
    for mask in masks:
        result &= mask.astype(bool)

    return result.astype(np.uint8)


def majority_vote_masks(masks: list[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """Combine masks using majority voting.

    Args:
        masks: List of binary masks (H, W).
        threshold: Voting threshold (0.5 = majority).

    Returns:
        Combined mask (H, W).
    """
    if not masks:
        raise ValueError("No masks to combine")

    stacked = np.stack([m.astype(np.float32) for m in masks], axis=0)
    votes = np.mean(stacked, axis=0)

    return (votes >= threshold).astype(np.uint8)


def weighted_ensemble_masks(
    masks: list[np.ndarray],
    weights: list[float],
    threshold: float = 0.5,
) -> np.ndarray:
    """Combine masks using weighted averaging.

    Args:
        masks: List of binary masks (H, W).
        weights: Weight for each mask (must sum to 1.0).
        threshold: Decision threshold.

    Returns:
        Combined mask (H, W).
    """
    if not masks:
        raise ValueError("No masks to combine")

    if len(masks) != len(weights):
        raise ValueError("Number of masks must match number of weights")

    if not np.isclose(sum(weights), 1.0):
        raise ValueError("Weights must sum to 1.0")

    weighted = np.zeros_like(masks[0], dtype=np.float32)
    for mask, weight in zip(masks, weights):
        weighted += mask.astype(np.float32) * weight

    return (weighted >= threshold).astype(np.uint8)


def largest_component_filter(mask: np.ndarray, min_size: int = 100) -> np.ndarray:
    """Keep only the largest connected component.

    Args:
        mask: Binary mask (H, W).
        min_size: Minimum component size to keep.

    Returns:
        Filtered mask (H, W).
    """
    labeled, num_features = label(mask)

    if num_features == 0:
        return mask

    # Find largest component
    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0  # Ignore background

    if component_sizes.max() < min_size:
        return np.zeros_like(mask)

    largest_label = component_sizes.argmax()

    return (labeled == largest_label).astype(np.uint8)


def ensemble_semantic_masks(
    masks: list[np.ndarray],
    strategy: str = "union",
    **kwargs,
) -> np.ndarray:
    """High-level ensemble function for semantic masks.

    Args:
        masks: List of semantic masks (H, W) with class labels.
        strategy: Combination strategy (union, intersection, majority, weighted).
        **kwargs: Additional arguments for specific strategies.

    Returns:
        Combined semantic mask (H, W).
    """
    if not masks:
        raise ValueError("No masks to combine")

    # Extract unique classes
    unique_classes = np.unique(np.concatenate([np.unique(m) for m in masks]))
    unique_classes = unique_classes[unique_classes > 0]  # Exclude background

    result = np.zeros_like(masks[0], dtype=np.uint8)

    # Process each class separately
    for cls in unique_classes:
        class_masks = [(m == cls).astype(np.uint8) for m in masks]

        if strategy == "union":
            combined = union_masks(class_masks)
        elif strategy == "intersection":
            combined = intersection_masks(class_masks)
        elif strategy == "majority":
            threshold = kwargs.get("threshold", 0.5)
            combined = majority_vote_masks(class_masks, threshold=threshold)
        elif strategy == "weighted":
            weights = kwargs.get("weights")
            threshold = kwargs.get("threshold", 0.5)
            if weights is None:
                raise ValueError("Weighted strategy requires 'weights' parameter")
            combined = weighted_ensemble_masks(class_masks, weights, threshold=threshold)
        else:
            raise ValueError(f"Unknown ensemble strategy: {strategy}")

        # Assign class to result
        result[combined > 0] = cls

    return result
