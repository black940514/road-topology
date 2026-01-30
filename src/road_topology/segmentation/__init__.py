"""Segmentation module for road topology.

This module provides:
- Dataset classes for loading image/mask pairs
- Data augmentation transforms
- Segmentation loss functions
- Lane instance segmentation components
"""
from road_topology.segmentation.dataset import (
    LaneInstanceDataset,
    PseudoLabelDataset,
    RoadTopologyDataset,
)
from road_topology.segmentation.instance_generator import InstanceMaskGenerator
from road_topology.segmentation.losses import (
    CombinedLoss,
    DiceLoss,
    DiscriminativeLoss,
    FocalLoss,
    weighted_cross_entropy,
)
from road_topology.segmentation.postprocess import LanePostProcessor
from road_topology.segmentation.trainer_lane import LaneSegmentationTrainer
from road_topology.segmentation.transforms import (
    get_test_transforms,
    get_train_transforms,
    get_val_transforms,
)

__all__ = [
    # Datasets
    "RoadTopologyDataset",
    "PseudoLabelDataset",
    "LaneInstanceDataset",
    # Transforms
    "get_train_transforms",
    "get_val_transforms",
    "get_test_transforms",
    # Losses
    "DiceLoss",
    "FocalLoss",
    "CombinedLoss",
    "DiscriminativeLoss",
    "weighted_cross_entropy",
    # Lane instance components
    "InstanceMaskGenerator",
    "LanePostProcessor",
    "LaneSegmentationTrainer",
]
