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


def compute_lane_instance_metrics(
    pred_instances: np.ndarray,
    gt_instances: np.ndarray,
    num_lanes_pred: int,
    num_lanes_gt: int,
) -> dict[str, float]:
    """Compute lane instance segmentation metrics.

    Args:
        pred_instances: Predicted instance mask (H, W) with instance IDs.
        gt_instances: Ground truth instance mask (H, W) with instance IDs.
        num_lanes_pred: Number of predicted lane instances.
        num_lanes_gt: Number of ground truth lane instances.

    Returns:
        Dictionary containing:
        - lane_detection_accuracy: Percentage of lanes correctly detected.
        - lane_ordering_accuracy: Percentage of lanes in correct order.
        - mean_instance_iou: Average IoU across lane instances.
        - panoptic_quality: PQ metric for instance segmentation.
    """
    # Match predicted instances to ground truth instances
    iou_matrix = np.zeros((num_lanes_pred, num_lanes_gt))

    for i in range(1, num_lanes_pred + 1):
        pred_mask = (pred_instances == i)
        for j in range(1, num_lanes_gt + 1):
            gt_mask = (gt_instances == j)
            intersection = (pred_mask & gt_mask).sum()
            union = (pred_mask | gt_mask).sum()
            iou_matrix[i - 1, j - 1] = intersection / union if union > 0 else 0.0

    # Hungarian matching (greedy approximation)
    matched_pairs = []
    iou_threshold = 0.5

    used_pred = set()
    used_gt = set()

    # Greedy matching based on IoU
    for _ in range(min(num_lanes_pred, num_lanes_gt)):
        max_iou = 0
        max_i, max_j = -1, -1

        for i in range(num_lanes_pred):
            if i in used_pred:
                continue
            for j in range(num_lanes_gt):
                if j in used_gt:
                    continue
                if iou_matrix[i, j] > max_iou:
                    max_iou = iou_matrix[i, j]
                    max_i, max_j = i, j

        if max_iou >= iou_threshold:
            matched_pairs.append((max_i, max_j, max_iou))
            used_pred.add(max_i)
            used_gt.add(max_j)
        else:
            break

    # Compute metrics
    num_matched = len(matched_pairs)
    lane_detection_accuracy = num_matched / num_lanes_gt if num_lanes_gt > 0 else 0.0

    mean_instance_iou = 0.0
    if matched_pairs:
        mean_instance_iou = sum(iou for _, _, iou in matched_pairs) / len(matched_pairs)

    # Panoptic Quality: PQ = SQ * RQ
    # SQ (Segmentation Quality) = mean IoU of matched instances
    # RQ (Recognition Quality) = F1 of detection
    sq = mean_instance_iou

    tp = num_matched
    fp = num_lanes_pred - num_matched
    fn = num_lanes_gt - num_matched

    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    pq = sq * rq

    # Compute lane ordering accuracy (left-to-right)
    pred_order = _get_lane_order(pred_instances, num_lanes_pred)
    gt_order = _get_lane_order(gt_instances, num_lanes_gt)
    lane_ordering_accuracy = compute_lane_ordering_accuracy(pred_order, gt_order)

    return {
        "lane_detection_accuracy": float(lane_detection_accuracy),
        "lane_ordering_accuracy": float(lane_ordering_accuracy),
        "mean_instance_iou": float(mean_instance_iou),
        "panoptic_quality": float(pq),
    }


def compute_lane_ordering_accuracy(pred_order: list[int], gt_order: list[int]) -> float:
    """Check if lanes are ordered correctly left-to-right.

    Args:
        pred_order: Predicted lane instance IDs in left-to-right order.
        gt_order: Ground truth lane instance IDs in left-to-right order.

    Returns:
        Ordering accuracy (0-1).
    """
    if not pred_order or not gt_order:
        return 0.0

    # Use longest common subsequence to measure ordering similarity
    n, m = len(pred_order), len(gt_order)

    # Simple LCS-based metric
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pred_order[i - 1] == gt_order[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[n][m]
    ordering_accuracy = lcs_length / max(n, m)

    return float(ordering_accuracy)


def _get_lane_order(instance_mask: np.ndarray, num_lanes: int) -> list[int]:
    """Get lane instance IDs ordered from left to right.

    Args:
        instance_mask: Instance mask (H, W) with instance IDs.
        num_lanes: Number of lane instances.

    Returns:
        List of instance IDs ordered left-to-right.
    """
    centroids = []

    for lane_id in range(1, num_lanes + 1):
        lane_mask = (instance_mask == lane_id)
        if lane_mask.sum() == 0:
            continue

        # Compute centroid
        y_coords, x_coords = np.where(lane_mask)
        centroid_x = x_coords.mean()
        centroids.append((centroid_x, lane_id))

    # Sort by x-coordinate (left to right)
    centroids.sort(key=lambda c: c[0])

    return [lane_id for _, lane_id in centroids]
