"""Type definitions for Road Topology Segmentation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BoundingBox:
    """Bounding box for object detection.

    Attributes:
        x1: Left coordinate.
        y1: Top coordinate.
        x2: Right coordinate.
        y2: Bottom coordinate.
        confidence: Detection confidence score [0, 1].
        class_id: Class identifier.
        class_name: Optional class name.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str | None = None

    def __post_init__(self) -> None:
        """Validate bounding box coordinates."""
        if self.x1 > self.x2:
            raise ValueError(f"x1 ({self.x1}) must be <= x2 ({self.x2})")
        if self.y1 > self.y2:
            raise ValueError(f"y1 ({self.y1}) must be <= y2 ({self.y2})")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def center(self) -> tuple[float, float]:
        """Get center point of bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def area(self) -> float:
        """Get area of bounding box."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def width(self) -> float:
        """Get width of bounding box."""
        return self.x2 - self.x1

    def height(self) -> float:
        """Get height of bounding box."""
        return self.y2 - self.y1

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Return coordinates as (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)

    def to_xywh(self) -> tuple[float, float, float, float]:
        """Return coordinates as (x, y, width, height) tuple."""
        return (self.x1, self.y1, self.width(), self.height())

    def expand(self, ratio: float) -> "BoundingBox":
        """Expand bounding box by a ratio.

        Args:
            ratio: Expansion ratio (e.g., 0.1 for 10% expansion).

        Returns:
            New expanded BoundingBox.
        """
        w, h = self.width(), self.height()
        dx, dy = w * ratio / 2, h * ratio / 2
        return BoundingBox(
            x1=self.x1 - dx,
            y1=self.y1 - dy,
            x2=self.x2 + dx,
            y2=self.y2 + dy,
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name,
        )


@dataclass
class Trajectory:
    """Vehicle trajectory as a sequence of points.

    Attributes:
        points: Array of (x, y) coordinates with shape (N, 2).
        confidence_scores: Confidence scores for each point with shape (N,).
        track_id: Optional track identifier.
    """
    points: np.ndarray
    confidence_scores: np.ndarray
    track_id: int | None = None

    def __post_init__(self) -> None:
        """Validate trajectory data."""
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError(f"points must have shape (N, 2), got {self.points.shape}")
        if len(self.points) != len(self.confidence_scores):
            raise ValueError(
                f"points and confidence_scores must have same length: "
                f"{len(self.points)} vs {len(self.confidence_scores)}"
            )

    def __len__(self) -> int:
        """Return number of points in trajectory."""
        return len(self.points)

    def to_polyline(self) -> np.ndarray:
        """Return trajectory as polyline array (N, 2)."""
        return self.points.copy()

    def smooth(self, window: int = 5) -> "Trajectory":
        """Smooth trajectory using moving average.

        Args:
            window: Smoothing window size (must be odd).

        Returns:
            New smoothed Trajectory.
        """
        if window % 2 == 0:
            window += 1
        if len(self.points) < window:
            return Trajectory(
                points=self.points.copy(),
                confidence_scores=self.confidence_scores.copy(),
                track_id=self.track_id,
            )

        # Apply moving average
        kernel = np.ones(window) / window
        smoothed_x = np.convolve(self.points[:, 0], kernel, mode='same')
        smoothed_y = np.convolve(self.points[:, 1], kernel, mode='same')
        smoothed_points = np.stack([smoothed_x, smoothed_y], axis=1)

        return Trajectory(
            points=smoothed_points,
            confidence_scores=self.confidence_scores.copy(),
            track_id=self.track_id,
        )

    def mean_confidence(self) -> float:
        """Get mean confidence score."""
        return float(np.mean(self.confidence_scores))


@dataclass
class Track:
    """Object track containing a sequence of bounding boxes.

    Attributes:
        track_id: Unique track identifier.
        boxes: List of bounding boxes over time.
        timestamps: Frame timestamps for each box.
    """
    track_id: int
    boxes: list[BoundingBox] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate track data."""
        if len(self.boxes) != len(self.timestamps):
            raise ValueError(
                f"boxes and timestamps must have same length: "
                f"{len(self.boxes)} vs {len(self.timestamps)}"
            )

    def __len__(self) -> int:
        """Return number of detections in track."""
        return len(self.boxes)

    def add_detection(self, box: BoundingBox, timestamp: float) -> None:
        """Add a detection to the track."""
        self.boxes.append(box)
        self.timestamps.append(timestamp)

    def trajectory(self) -> Trajectory:
        """Extract trajectory from track.

        Returns:
            Trajectory of box centers.
        """
        points = np.array([box.center() for box in self.boxes])
        confidences = np.array([box.confidence for box in self.boxes])
        return Trajectory(points=points, confidence_scores=confidences, track_id=self.track_id)

    def duration(self) -> float:
        """Get track duration in seconds."""
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]


@dataclass
class DetectionResult:
    """Result from object detection on a single frame.

    Attributes:
        frame_id: Frame identifier.
        boxes: List of detected bounding boxes.
        inference_time_ms: Inference time in milliseconds.
        metadata: Optional additional metadata.
    """
    frame_id: int
    boxes: list[BoundingBox]
    inference_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def filter_by_confidence(self, threshold: float) -> "DetectionResult":
        """Filter detections by confidence threshold."""
        filtered = [b for b in self.boxes if b.confidence >= threshold]
        return DetectionResult(
            frame_id=self.frame_id,
            boxes=filtered,
            inference_time_ms=self.inference_time_ms,
            metadata=self.metadata,
        )

    def filter_by_class(self, class_ids: list[int]) -> "DetectionResult":
        """Filter detections by class IDs."""
        filtered = [b for b in self.boxes if b.class_id in class_ids]
        return DetectionResult(
            frame_id=self.frame_id,
            boxes=filtered,
            inference_time_ms=self.inference_time_ms,
            metadata=self.metadata,
        )


@dataclass
class PseudoLabelResult:
    """Result from pseudo-label generation.

    Attributes:
        mask: Semantic segmentation mask with class indices (H, W).
        confidence: Per-pixel confidence map [0, 1] (H, W).
        trajectories: Trajectories used to generate the mask.
        metadata: Additional metadata.
    """
    mask: np.ndarray
    confidence: np.ndarray
    trajectories: list[Trajectory]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate pseudo-label data."""
        if self.mask.shape != self.confidence.shape:
            raise ValueError(
                f"mask and confidence must have same shape: "
                f"{self.mask.shape} vs {self.confidence.shape}"
            )


@dataclass
class PredictionResult:
    """Result from model inference.

    Attributes:
        mask: Predicted segmentation mask with class indices (H, W).
        confidence: Per-pixel confidence map [0, 1] (H, W).
        class_probabilities: Class probability map (H, W, C).
        inference_time_ms: Inference time in milliseconds.
    """
    mask: np.ndarray
    confidence: np.ndarray
    class_probabilities: np.ndarray
    inference_time_ms: float


# Class definitions (original, kept for backward compatibility)
CLASS_NAMES = ["background", "road", "lane", "crosswalk", "sidewalk"]
NUM_CLASSES = len(CLASS_NAMES)

CLASS_COLORS = {
    0: (0, 0, 0),       # background - black
    1: (128, 128, 128), # road - gray
    2: (255, 255, 255), # lane - white
    3: (255, 255, 0),   # crosswalk - yellow
    4: (0, 255, 0),     # sidewalk - green
}

# Extended class definitions for lane instance segmentation
SEMANTIC_CLASS_NAMES = ["background", "road", "lane_marking", "crosswalk", "sidewalk", "lane_boundary"]
NUM_SEMANTIC_CLASSES = 6

LANE_INSTANCE_NAMES = ["background", "lane_1", "lane_2", "lane_3", "lane_4", "lane_5", "lane_6"]
MAX_LANES = 6

SEMANTIC_CLASS_COLORS = {
    0: (0, 0, 0),       # background
    1: (128, 128, 128), # road
    2: (255, 255, 255), # lane_marking
    3: (255, 255, 0),   # crosswalk
    4: (0, 255, 0),     # sidewalk
    5: (0, 0, 255),     # lane_boundary
}

LANE_INSTANCE_COLORS = {
    0: (0, 0, 0),       # background
    1: (255, 0, 0),     # lane_1
    2: (0, 255, 0),     # lane_2
    3: (0, 0, 255),     # lane_3
    4: (255, 255, 0),   # lane_4
    5: (255, 0, 255),   # lane_5
    6: (0, 255, 255),   # lane_6
}
