"""Loss functions for segmentation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for segmentation.

    Computes 1 - Dice coefficient, which is differentiable and suitable for
    optimization.

    Args:
        smooth: Smoothing factor to avoid division by zero.
        ignore_index: Class index to ignore in loss computation.
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int | None = None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            pred: Predictions (B, C, H, W) logits.
            target: Ground truth (B, H, W) class indices.
            weight: Optional pixel weights (B, H, W).

        Returns:
            Scalar loss.
        """
        # Convert predictions to probabilities
        pred = F.softmax(pred, dim=1)

        # One-hot encode targets
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        # Apply ignore mask if specified
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).unsqueeze(1).float()
            pred = pred * mask
            target_one_hot = target_one_hot * mask

        # Apply pixel weights if provided
        if weight is not None:
            weight = weight.unsqueeze(1)
            pred = pred * weight
            target_one_hot = target_one_hot * weight

        # Compute Dice coefficient per class
        intersection = (pred * target_one_hot).sum(dim=(0, 2, 3))
        union = pred.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return mean Dice loss
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance.

    Focal loss down-weights easy examples and focuses on hard negatives.

    Args:
        alpha: Weighting factor for each class.
        gamma: Focusing parameter (higher = more focus on hard examples).
        ignore_index: Class index to ignore.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        ignore_index: int | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            pred: Predictions (B, C, H, W) logits.
            target: Ground truth (B, H, W) class indices.
            weight: Optional pixel weights (B, H, W).

        Returns:
            Scalar loss.
        """
        # Compute cross-entropy
        ce_loss = F.cross_entropy(
            pred,
            target,
            reduction="none",
            ignore_index=self.ignore_index if self.ignore_index is not None else -100,
        )

        # Compute focal weight
        p = torch.exp(-ce_loss)
        focal_weight = (1 - p) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_weight = self.alpha[target]
            focal_loss = alpha_weight * focal_loss

        # Apply pixel weights if provided
        if weight is not None:
            focal_loss = focal_loss * weight

        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss: weighted sum of multiple loss functions.

    Args:
        losses: Dictionary of loss name -> (loss_fn, weight).
        ignore_index: Class index to ignore.
    """

    def __init__(
        self,
        losses: dict[str, tuple[nn.Module, float]],
        ignore_index: int | None = None,
    ):
        super().__init__()
        self.losses = nn.ModuleDict({name: loss for name, (loss, _) in losses.items()})
        self.weights = {name: weight for name, (_, weight) in losses.items()}
        self.ignore_index = ignore_index

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss.

        Args:
            pred: Predictions (B, C, H, W) logits.
            target: Ground truth (B, H, W) class indices.
            weight: Optional pixel weights (B, H, W).

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        total_loss = 0.0
        loss_dict = {}

        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(pred, target, weight)
            weighted_loss = loss_value * self.weights[name]
            total_loss = total_loss + weighted_loss
            loss_dict[name] = loss_value.item()

        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict


def weighted_cross_entropy(
    pred: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    pixel_weights: torch.Tensor | None = None,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """Weighted cross-entropy loss.

    Supports both class-level and pixel-level weighting.

    Args:
        pred: Predictions (B, C, H, W) logits.
        target: Ground truth (B, H, W) class indices.
        class_weights: Per-class weights (C,).
        pixel_weights: Per-pixel weights (B, H, W).
        ignore_index: Class index to ignore.

    Returns:
        Scalar loss.
    """
    ce_loss = F.cross_entropy(
        pred,
        target,
        weight=class_weights,
        reduction="none",
        ignore_index=ignore_index if ignore_index is not None else -100,
    )

    if pixel_weights is not None:
        ce_loss = ce_loss * pixel_weights

    return ce_loss.mean()


def compute_class_weights(
    dataset: torch.utils.data.Dataset,
    num_classes: int,
    method: str = "inverse_frequency",
) -> torch.Tensor:
    """Compute class weights from dataset.

    Args:
        dataset: Dataset to compute weights from.
        num_classes: Number of classes.
        method: Weighting method ("inverse_frequency" or "effective_number").

    Returns:
        Class weights tensor (C,).
    """
    # Count pixels per class
    class_counts = torch.zeros(num_classes)

    for i in range(len(dataset)):
        sample = dataset[i]
        mask = sample["mask"]
        for c in range(num_classes):
            class_counts[c] += (mask == c).sum()

    if method == "inverse_frequency":
        # Inverse frequency weighting
        total_pixels = class_counts.sum()
        weights = total_pixels / (num_classes * class_counts)
        weights = weights / weights.sum()  # Normalize

    elif method == "effective_number":
        # Effective number of samples (ENS) weighting
        # From: "Class-Balanced Loss Based on Effective Number of Samples"
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum()  # Normalize

    else:
        raise ValueError(f"Unknown weighting method: {method}")

    return weights
