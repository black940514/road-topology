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


class DiscriminativeLoss(nn.Module):
    """Discriminative loss for instance segmentation.

    Based on https://arxiv.org/abs/1708.02551
    "Semantic Instance Segmentation with a Discriminative Loss Function"

    Components:
    1. Variance term: Pull pixels toward instance mean (delta_var=0.5)
    2. Distance term: Push instance means apart (delta_dist=1.5)
    3. Regularization: Keep means close to origin (gamma=0.001)

    Args:
        delta_var: Margin for variance term (pull force).
        delta_dist: Margin for distance term (push force).
        norm: Norm type for distance computation (1 or 2).
        alpha: Weight for variance term.
        beta: Weight for distance term.
        gamma: Weight for regularization term.
    """

    def __init__(
        self,
        delta_var: float = 0.5,
        delta_dist: float = 1.5,
        norm: int = 2,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.001,
    ):
        super().__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(
        self,
        embeddings: torch.Tensor,
        instance_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute discriminative loss.

        Args:
            embeddings: Predicted embeddings (B, E, H, W)
            instance_labels: Ground truth instance IDs (B, H, W)
                            - 0 is background (ignored)
                            - Each unique value > 0 is a separate instance

        Returns:
            Tuple of (total_loss, loss_dict with components)
        """
        batch_size, embed_dim, height, width = embeddings.shape

        # Reshape embeddings: (B, E, H, W) -> (B*H*W, E)
        embeddings_flat = embeddings.permute(0, 2, 3, 1).reshape(-1, embed_dim)
        labels_flat = instance_labels.reshape(-1)

        total_var_loss = 0.0
        total_dist_loss = 0.0
        total_reg_loss = 0.0

        num_instances_total = 0

        for b in range(batch_size):
            # Get embeddings and labels for this batch item
            batch_mask = torch.arange(batch_size, device=embeddings.device).repeat_interleave(height * width) == b
            batch_embeddings = embeddings_flat[batch_mask]
            batch_labels = labels_flat[batch_mask]

            # Find unique instances (excluding background = 0)
            unique_instances = torch.unique(batch_labels)
            unique_instances = unique_instances[unique_instances > 0]

            if len(unique_instances) == 0:
                continue

            num_instances = len(unique_instances)
            num_instances_total += num_instances

            # Compute instance means
            instance_means = []
            for instance_id in unique_instances:
                instance_mask = batch_labels == instance_id
                instance_embeddings = batch_embeddings[instance_mask]
                instance_mean = instance_embeddings.mean(dim=0)
                instance_means.append(instance_mean)

            instance_means = torch.stack(instance_means)  # (N, E)

            # 1. Variance term: pull pixels toward instance mean
            var_loss = 0.0
            for i, instance_id in enumerate(unique_instances):
                instance_mask = batch_labels == instance_id
                instance_embeddings = batch_embeddings[instance_mask]
                mean = instance_means[i]

                # Distance from mean
                if self.norm == 1:
                    distances = torch.abs(instance_embeddings - mean).sum(dim=1)
                else:  # norm == 2
                    distances = torch.norm(instance_embeddings - mean, p=2, dim=1)

                # Hinge loss with margin delta_var
                var_loss += torch.clamp(distances - self.delta_var, min=0).sum()

            var_loss = var_loss / num_instances
            total_var_loss += var_loss

            # 2. Distance term: push instance means apart
            dist_loss = 0.0
            if num_instances > 1:
                for i in range(num_instances):
                    for j in range(i + 1, num_instances):
                        mean_i = instance_means[i]
                        mean_j = instance_means[j]

                        if self.norm == 1:
                            distance = torch.abs(mean_i - mean_j).sum()
                        else:  # norm == 2
                            distance = torch.norm(mean_i - mean_j, p=2)

                        # Hinge loss with margin delta_dist
                        dist_loss += torch.clamp(2 * self.delta_dist - distance, min=0)

                # Normalize by number of pairs
                num_pairs = num_instances * (num_instances - 1) / 2
                dist_loss = dist_loss / num_pairs

            total_dist_loss += dist_loss

            # 3. Regularization term: keep means close to origin
            reg_loss = torch.norm(instance_means, p=self.norm, dim=1).sum() / num_instances
            total_reg_loss += reg_loss

        # Average over batch
        if num_instances_total > 0:
            total_var_loss = total_var_loss / batch_size
            total_dist_loss = total_dist_loss / batch_size
            total_reg_loss = total_reg_loss / batch_size

        # Combine losses
        total_loss = (
            self.alpha * total_var_loss +
            self.beta * total_dist_loss +
            self.gamma * total_reg_loss
        )

        loss_dict = {
            "var_loss": total_var_loss.item() if isinstance(total_var_loss, torch.Tensor) else total_var_loss,
            "dist_loss": total_dist_loss.item() if isinstance(total_dist_loss, torch.Tensor) else total_dist_loss,
            "reg_loss": total_reg_loss.item() if isinstance(total_reg_loss, torch.Tensor) else total_reg_loss,
            "total": total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
        }

        return total_loss, loss_dict
