"""Data augmentation transforms for road topology segmentation."""
from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    image_size: int = 512,
    normalize: bool = True,
) -> A.Compose:
    """Get training augmentation pipeline.

    Includes:
    - Random crops and resizing
    - Geometric transforms (flip, rotate, shift)
    - Color jitter
    - Blur and noise

    Args:
        image_size: Target image size.
        normalize: Whether to apply ImageNet normalization.

    Returns:
        Albumentations composition.
    """
    transforms = [
        # Geometric transforms
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Affine(
            translate_percent=0.1,
            scale=(0.85, 1.15),
            rotate=(-15, 15),
            border_mode=0,
            p=0.5,
        ),

        # Resize
        A.Resize(image_size, image_size),

        # Color transforms
        A.OneOf([
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=1.0,
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0,
            ),
        ], p=0.5),

        # Blur and noise
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=7, p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.3),

        A.GaussNoise(std_range=(0.02, 0.1), p=0.2),

        # Weather effects
        A.OneOf([
            A.RandomBrightnessContrast(p=1.0),
            A.RandomGamma(p=1.0),
        ], p=0.3),

        # Coarse dropout (random erasing)
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(image_size // 32, image_size // 16),
            hole_width_range=(image_size // 32, image_size // 16),
            fill=0,
            p=0.2,
        ),
    ]

    if normalize:
        transforms.append(
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )

    transforms.append(ToTensorV2())

    return A.Compose(
        transforms,
        additional_targets={"confidence": "mask"},
    )


def get_val_transforms(
    image_size: int = 512,
    normalize: bool = True,
) -> A.Compose:
    """Get validation transforms.

    Only resizes and normalizes, no augmentation.

    Args:
        image_size: Target image size.
        normalize: Whether to apply ImageNet normalization.

    Returns:
        Albumentations composition.
    """
    transforms = [
        A.Resize(image_size, image_size),
    ]

    if normalize:
        transforms.append(
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )

    transforms.append(ToTensorV2())

    return A.Compose(
        transforms,
        additional_targets={"confidence": "mask"},
    )


def get_test_transforms(
    image_size: int = 512,
    normalize: bool = True,
) -> A.Compose:
    """Get test-time transforms.

    Same as validation transforms.

    Args:
        image_size: Target image size.
        normalize: Whether to apply ImageNet normalization.

    Returns:
        Albumentations composition.
    """
    return get_val_transforms(image_size, normalize)


def get_tta_transforms(
    image_size: int = 512,
    normalize: bool = True,
) -> list[A.Compose]:
    """Get test-time augmentation (TTA) transforms.

    Returns a list of different augmentations for TTA ensemble.

    Args:
        image_size: Target image size.
        normalize: Whether to apply ImageNet normalization.

    Returns:
        List of albumentations compositions for TTA.
    """
    base_transforms = [A.Resize(image_size, image_size)]

    if normalize:
        base_transforms.append(
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )

    tta_list = [
        # Original
        A.Compose(base_transforms + [ToTensorV2()]),

        # Horizontal flip
        A.Compose(
            base_transforms + [A.HorizontalFlip(p=1.0), ToTensorV2()]
        ),

        # Vertical flip
        A.Compose(
            base_transforms + [A.VerticalFlip(p=1.0), ToTensorV2()]
        ),

        # Rotate 90
        A.Compose(
            base_transforms + [A.Rotate(limit=(90, 90), p=1.0), ToTensorV2()]
        ),

        # Rotate 180
        A.Compose(
            base_transforms + [A.Rotate(limit=(180, 180), p=1.0), ToTensorV2()]
        ),

        # Rotate 270
        A.Compose(
            base_transforms + [A.Rotate(limit=(270, 270), p=1.0), ToTensorV2()]
        ),
    ]

    return tta_list
