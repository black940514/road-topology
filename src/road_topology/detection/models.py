"""Detection result models for batch processing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from road_topology.core.types import BoundingBox


@dataclass
class DetectionResult:
    """Result from object detection on a single frame.

    Attributes:
        frame_id: Frame identifier/number.
        boxes: List of detected bounding boxes.
        inference_time_ms: Inference time in milliseconds.
        metadata: Optional additional metadata.
    """
    frame_id: int
    boxes: list[BoundingBox]
    inference_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def filter_by_confidence(self, threshold: float) -> "DetectionResult":
        """Filter detections by confidence threshold.

        Args:
            threshold: Minimum confidence score.

        Returns:
            New DetectionResult with filtered boxes.
        """
        filtered = [b for b in self.boxes if b.confidence >= threshold]
        return DetectionResult(
            frame_id=self.frame_id,
            boxes=filtered,
            inference_time_ms=self.inference_time_ms,
            metadata=self.metadata,
        )

    def filter_by_class(self, class_ids: list[int]) -> "DetectionResult":
        """Filter detections by class IDs.

        Args:
            class_ids: List of class IDs to keep.

        Returns:
            New DetectionResult with filtered boxes.
        """
        filtered = [b for b in self.boxes if b.class_id in class_ids]
        return DetectionResult(
            frame_id=self.frame_id,
            boxes=filtered,
            inference_time_ms=self.inference_time_ms,
            metadata=self.metadata,
        )

    def count_by_class(self) -> dict[int, int]:
        """Count detections by class ID.

        Returns:
            Dictionary mapping class_id to count.
        """
        counts: dict[int, int] = {}
        for box in self.boxes:
            counts[box.class_id] = counts.get(box.class_id, 0) + 1
        return counts


@dataclass
class DetectionBatch:
    """Batch of detection results from multiple frames.

    Attributes:
        results: List of DetectionResult for each frame.
        total_inference_time_ms: Total inference time for the batch.
        batch_size: Number of frames in the batch.
    """
    results: list[DetectionResult]
    total_inference_time_ms: float
    batch_size: int

    def __post_init__(self) -> None:
        """Validate batch data."""
        if len(self.results) != self.batch_size:
            raise ValueError(
                f"results length ({len(self.results)}) does not match "
                f"batch_size ({self.batch_size})"
            )

    def __len__(self) -> int:
        """Return number of frames in batch."""
        return self.batch_size

    def __getitem__(self, idx: int) -> DetectionResult:
        """Get detection result by index."""
        return self.results[idx]

    def total_detections(self) -> int:
        """Get total number of detections across all frames."""
        return sum(len(r.boxes) for r in self.results)

    def mean_inference_time(self) -> float:
        """Get mean inference time per frame in milliseconds."""
        return self.total_inference_time_ms / self.batch_size if self.batch_size > 0 else 0.0

    def filter_by_confidence(self, threshold: float) -> "DetectionBatch":
        """Filter all detections by confidence threshold.

        Args:
            threshold: Minimum confidence score.

        Returns:
            New DetectionBatch with filtered results.
        """
        filtered_results = [r.filter_by_confidence(threshold) for r in self.results]
        return DetectionBatch(
            results=filtered_results,
            total_inference_time_ms=self.total_inference_time_ms,
            batch_size=self.batch_size,
        )

    def filter_by_class(self, class_ids: list[int]) -> "DetectionBatch":
        """Filter all detections by class IDs.

        Args:
            class_ids: List of class IDs to keep.

        Returns:
            New DetectionBatch with filtered results.
        """
        filtered_results = [r.filter_by_class(class_ids) for r in self.results]
        return DetectionBatch(
            results=filtered_results,
            total_inference_time_ms=self.total_inference_time_ms,
            batch_size=self.batch_size,
        )

    def aggregate_counts(self) -> dict[int, int]:
        """Aggregate detection counts by class across all frames.

        Returns:
            Dictionary mapping class_id to total count.
        """
        total_counts: dict[int, int] = {}
        for result in self.results:
            for class_id, count in result.count_by_class().items():
                total_counts[class_id] = total_counts.get(class_id, 0) + count
        return total_counts
