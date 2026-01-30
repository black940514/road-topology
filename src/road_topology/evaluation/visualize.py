"""Visualization utilities for evaluation."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from road_topology.core.types import CLASS_COLORS, CLASS_NAMES, NUM_CLASSES


def visualize_prediction(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    class_names: list[str] = CLASS_NAMES,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Visualize prediction with optional ground truth and confidence.

    Args:
        image: Original RGB image (H, W, 3).
        pred_mask: Predicted segmentation mask (H, W).
        gt_mask: Optional ground truth mask (H, W).
        confidence: Optional confidence map (H, W).
        class_names: List of class names for legend.
        save_path: Optional path to save figure.
        show: Whether to display the figure.
    """
    num_cols = 2 + (gt_mask is not None) + (confidence is not None)
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))

    if num_cols == 1:
        axes = [axes]

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Predicted mask
    pred_colored = _mask_to_color(pred_mask)
    axes[1].imshow(pred_colored)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    col_idx = 2

    # Ground truth (if provided)
    if gt_mask is not None:
        gt_colored = _mask_to_color(gt_mask)
        axes[col_idx].imshow(gt_colored)
        axes[col_idx].set_title("Ground Truth")
        axes[col_idx].axis("off")
        col_idx += 1

    # Confidence map (if provided)
    if confidence is not None:
        im = axes[col_idx].imshow(confidence, cmap="viridis", vmin=0, vmax=1)
        axes[col_idx].set_title("Confidence")
        axes[col_idx].axis("off")
        plt.colorbar(im, ax=axes[col_idx], fraction=0.046, pad=0.04)

    # Add legend
    _add_legend(fig, class_names)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_predictions_grid(
    images: list[np.ndarray],
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray] | None = None,
    max_samples: int = 8,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Visualize multiple predictions in a grid.

    Args:
        images: List of RGB images.
        pred_masks: List of predicted masks.
        gt_masks: Optional list of ground truth masks.
        max_samples: Maximum number of samples to show.
        save_path: Optional path to save figure.
        show: Whether to display the figure.
    """
    n_samples = min(len(images), max_samples)
    n_cols = 3 if gt_masks is not None else 2

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(5 * n_cols, 5 * n_samples))

    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_samples):
        # Original image
        axes[i, 0].imshow(images[i])
        if i == 0:
            axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        # Prediction
        pred_colored = _mask_to_color(pred_masks[i])
        axes[i, 1].imshow(pred_colored)
        if i == 0:
            axes[i, 1].set_title("Prediction")
        axes[i, 1].axis("off")

        # Ground truth (if provided)
        if gt_masks is not None:
            gt_colored = _mask_to_color(gt_masks[i])
            axes[i, 2].imshow(gt_colored)
            if i == 0:
                axes[i, 2].set_title("Ground Truth")
            axes[i, 2].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
    normalize: bool = True,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plot confusion matrix.

    Args:
        cm: Confusion matrix (num_classes, num_classes).
        class_names: List of class names.
        normalize: Whether to normalize by row (true class).
        save_path: Optional path to save figure.
        show: Whether to display the figure.
    """
    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm = cm / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Normalized Count" if normalize else "Count"},
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_class_distribution(
    frequencies: dict[int, float],
    class_names: list[str] = CLASS_NAMES,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plot class frequency distribution.

    Args:
        frequencies: Dictionary mapping class ID to frequency.
        class_names: List of class names.
        save_path: Optional path to save figure.
        show: Whether to display the figure.
    """
    classes = [class_names[i] for i in sorted(frequencies.keys())]
    freqs = [frequencies[i] for i in sorted(frequencies.keys())]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(classes, freqs, color="steelblue")

    # Add percentage labels
    for bar, freq in zip(bars, freqs):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{freq * 100:.1f}%",
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("Class")
    ax.set_ylabel("Frequency")
    ax.set_title("Class Distribution")
    ax.set_ylim(0, max(freqs) * 1.1)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Create colored overlay of mask on image.

    Args:
        image: RGB image (H, W, 3).
        mask: Segmentation mask (H, W).
        alpha: Transparency of overlay [0, 1].

    Returns:
        RGB image with overlay (H, W, 3).
    """
    colored_mask = _mask_to_color(mask)
    overlay = (image * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
    return overlay


def _mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Convert class mask to RGB colored mask.

    Args:
        mask: Class indices (H, W).

    Returns:
        RGB colored mask (H, W, 3).
    """
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS.items():
        colored[mask == class_id] = color

    return colored


def _add_legend(fig, class_names: list[str]) -> None:
    """Add class legend to figure.

    Args:
        fig: Matplotlib figure.
        class_names: List of class names.
    """
    from matplotlib.patches import Patch

    legend_elements = []
    for class_id, name in enumerate(class_names):
        if class_id in CLASS_COLORS:
            color = np.array(CLASS_COLORS[class_id]) / 255.0
            legend_elements.append(Patch(facecolor=color, label=name))

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(class_names),
        frameon=False,
    )


def visualize_boundary_errors(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    tolerance: int = 2,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Visualize boundary prediction errors.

    Args:
        pred_mask: Predicted mask (H, W).
        gt_mask: Ground truth mask (H, W).
        tolerance: Pixel tolerance for boundary matching.
        save_path: Optional path to save figure.
        show: Whether to display the figure.
    """
    import cv2

    # Extract boundaries
    pred_boundary = cv2.Canny(pred_mask.astype(np.uint8), 0, 1)
    gt_boundary = cv2.Canny(gt_mask.astype(np.uint8), 0, 1)

    # Dilate for tolerance
    kernel = np.ones((tolerance * 2 + 1, tolerance * 2 + 1), np.uint8)
    pred_dilated = cv2.dilate(pred_boundary, kernel)
    gt_dilated = cv2.dilate(gt_boundary, kernel)

    # Compute true positives, false positives, false negatives
    true_pos = pred_boundary & gt_dilated
    false_pos = pred_boundary & ~gt_dilated
    false_neg = gt_boundary & ~pred_dilated

    # Create visualization
    viz = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    viz[true_pos > 0] = [0, 255, 0]  # Green: correct
    viz[false_pos > 0] = [255, 0, 0]  # Red: false positive
    viz[false_neg > 0] = [0, 0, 255]  # Blue: false negative

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(gt_boundary, cmap="gray")
    axes[0].set_title("Ground Truth Boundaries")
    axes[0].axis("off")

    axes[1].imshow(viz)
    axes[1].set_title("Boundary Errors (Green=TP, Red=FP, Blue=FN)")
    axes[1].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
