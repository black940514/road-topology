"""Segmentation module for road topology.

This module provides:
- Dataset classes for loading image/mask pairs
- Data augmentation transforms
- Segmentation loss functions
"""
from road_topology.segmentation.dataset import (
    PseudoLabelDataset,
    RoadTopologyDataset,
)
from road_topology.segmentation.losses import (
    CombinedLoss,
    DiceLoss,
    FocalLoss,
    weighted_cross_entropy,
)
from road_topology.segmentation.transforms import (
    get_test_transforms,
    get_train_transforms,
    get_val_transforms,
)

__all__ = [
    "RoadTopologyDataset",
    "PseudoLabelDataset",
    "get_train_transforms",
    "get_val_transforms",
    "get_test_transforms",
    "DiceLoss",
    "FocalLoss",
    "CombinedLoss",
    "weighted_cross_entropy",
]
