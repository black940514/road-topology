"""Evaluation metrics for segmentation."""
from __future__ import annotations

import numpy as np

from road_topology.core.types import CLASS_NAMES, NUM_CLASSES


def compute_miou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = 255,
) -> float:
    """Compute mean Intersection over Union.

    Args:
        pred: Predicted masks (N, H, W) or (H, W).
        target: Ground truth masks (N, H, W) or (H, W).
        num_classes: Number of classes.
        ignore_index: Index to ignore in computation.

    Returns:
        Mean IoU across all classes.
    """
    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]
        target = target[np.newaxis, ...]

    ious = []

    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        # Ignore specified index
        valid = target != ignore_index
        pred_cls = pred_cls & valid
        target_cls = target_cls & valid

        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()

        if union > 0:
            ious.append(intersection / union)

    return float(np.mean(ious)) if ious else 0.0


def compute_per_class_iou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = NUM_CLASSES,
    class_names: list[str] = CLASS_NAMES,
) -> dict[str, float]:
    """Compute IoU per class.

    Args:
        pred: Predicted masks.
        target: Ground truth masks.
        num_classes: Number of classes.
        class_names: Names for each class.

    Returns:
        Dictionary mapping class names to IoU values.
    """
    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]
        target = target[np.newaxis, ...]

    result = {}

    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()

        iou = intersection / union if union > 0 else 0.0
        name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        result[name] = float(iou)

    return result


def compute_pixel_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
    ignore_index: int = 255,
) -> float:
    """Compute pixel accuracy.

    Args:
        pred: Predicted masks.
        target: Ground truth masks.
        ignore_index: Index to ignore.

    Returns:
        Pixel accuracy.
    """
    valid = target != ignore_index
    correct = (pred == target) & valid
    return float(correct.sum() / valid.sum()) if valid.sum() > 0 else 0.0


def compute_confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        pred: Predicted masks.
        target: Ground truth masks.
        num_classes: Number of classes.

    Returns:
        Confusion matrix (num_classes, num_classes).
    """
    pred = pred.flatten()
    target = target.flatten()

    mask = (target >= 0) & (target < num_classes)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for i in range(num_classes):
        for j in range(num_classes):
            cm[i, j] = ((target == i) & (pred == j) & mask).sum()

    return cm


def compute_boundary_f1(
    pred: np.ndarray,
    target: np.ndarray,
    tolerance: int = 2,
) -> float:
    """Compute boundary F1 score.

    Args:
        pred: Predicted masks.
        target: Ground truth masks.
        tolerance: Pixel tolerance for boundary matching.

    Returns:
        Boundary F1 score.
    """
    import cv2

    # Extract boundaries
    pred_boundary = cv2.Canny(pred.astype(np.uint8), 0, 1)
    target_boundary = cv2.Canny(target.astype(np.uint8), 0, 1)

    # Dilate for tolerance
    kernel = np.ones((tolerance * 2 + 1, tolerance * 2 + 1), np.uint8)
    pred_dilated = cv2.dilate(pred_boundary, kernel)
    target_dilated = cv2.dilate(target_boundary, kernel)

    # Compute precision and recall
    true_pos_pred = (pred_boundary & target_dilated).sum()
    true_pos_target = (target_boundary & pred_dilated).sum()

    precision = true_pos_pred / pred_boundary.sum() if pred_boundary.sum() > 0 else 0
    recall = true_pos_target / target_boundary.sum() if target_boundary.sum() > 0 else 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return float(f1)


def compute_class_frequencies(
    masks: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> dict[int, float]:
    """Compute class frequency distribution.

    Args:
        masks: Array of masks (N, H, W).
        num_classes: Number of classes.

    Returns:
        Dictionary mapping class ID to frequency.
    """
    total_pixels = masks.size
    frequencies = {}

    for cls in range(num_classes):
        count = (masks == cls).sum()
        frequencies[cls] = float(count / total_pixels)

    return frequencies


def compute_dice_score(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> float:
    """Compute Dice coefficient (F1 score for segmentation).

    Args:
        pred: Predicted masks.
        target: Ground truth masks.
        num_classes: Number of classes.

    Returns:
        Mean Dice score across all classes.
    """
    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]
        target = target[np.newaxis, ...]

    dice_scores = []

    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum()
        pred_sum = pred_cls.sum()
        target_sum = target_cls.sum()

        if pred_sum + target_sum > 0:
            dice = (2 * intersection) / (pred_sum + target_sum)
            dice_scores.append(dice)

    return float(np.mean(dice_scores)) if dice_scores else 0.0


def compute_precision_recall(
    pred: np.ndarray,
    target: np.ndarray,
    class_id: int,
) -> tuple[float, float]:
    """Compute precision and recall for a specific class.

    Args:
        pred: Predicted masks.
        target: Ground truth masks.
        class_id: Class to compute metrics for.

    Returns:
        Tuple of (precision, recall).
    """
    pred_cls = (pred == class_id)
    target_cls = (target == class_id)

    true_pos = (pred_cls & target_cls).sum()
    false_pos = (pred_cls & ~target_cls).sum()
    false_neg = (~pred_cls & target_cls).sum()

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0

    return float(precision), float(recall)
