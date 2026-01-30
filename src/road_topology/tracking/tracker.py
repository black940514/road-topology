"""ByteTrack vehicle tracker via boxmot."""
from __future__ import annotations

from typing import Any

import numpy as np

from road_topology.core.config import TrackingConfig
from road_topology.core.exceptions import TrackingError
from road_topology.core.logging import get_logger
from road_topology.core.types import BoundingBox, Track

logger = get_logger(__name__)


class VehicleTracker:
    """ByteTrack-based vehicle tracker.

    Wraps boxmot BYTETracker for multi-object tracking.

    Args:
        config: Tracking configuration.
    """

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self._tracker = None
        self._tracks: dict[int, Track] = {}
        self._completed_tracks: list[Track] = []
        self._frame_id = 0

    @property
    def tracker(self):
        """Lazy initialize tracker."""
        if self._tracker is None:
            try:
                from boxmot import BYTETracker
                self._tracker = BYTETracker(
                    track_thresh=self.config.track_thresh,
                    track_buffer=self.config.track_buffer,
                    match_thresh=self.config.match_thresh,
                )
                logger.info("Initialized BYTETracker")
            except ImportError:
                raise TrackingError(
                    "boxmot not installed. Install with: pip install boxmot"
                )
        return self._tracker

    def update(
        self,
        detections: list[BoundingBox],
        frame_id: int | None = None,
    ) -> list[Track]:
        """Update tracker with new detections.

        Args:
            detections: List of detected bounding boxes.
            frame_id: Optional frame identifier.

        Returns:
            List of currently active tracks.
        """
        if frame_id is not None:
            self._frame_id = frame_id
        else:
            self._frame_id += 1

        try:
            # Convert detections to numpy array for boxmot
            # Format: [x1, y1, x2, y2, confidence, class_id]
            if len(detections) == 0:
                dets = np.empty((0, 6))
            else:
                dets = np.array([
                    [d.x1, d.y1, d.x2, d.y2, d.confidence, d.class_id]
                    for d in detections
                ])

            # Update tracker
            # boxmot returns: [x1, y1, x2, y2, track_id, confidence, class_id, ...]
            tracks_out = self.tracker.update(dets, None)  # img=None for speed

            # Process tracker output
            current_track_ids = set()

            for track_data in tracks_out:
                track_id = int(track_data[4])
                current_track_ids.add(track_id)

                box = BoundingBox(
                    x1=float(track_data[0]),
                    y1=float(track_data[1]),
                    x2=float(track_data[2]),
                    y2=float(track_data[3]),
                    confidence=float(track_data[5]) if len(track_data) > 5 else 1.0,
                    class_id=int(track_data[6]) if len(track_data) > 6 else 2,
                )

                # Update or create track
                if track_id not in self._tracks:
                    self._tracks[track_id] = Track(track_id=track_id)

                self._tracks[track_id].add_detection(box, float(self._frame_id))

            # Check for completed tracks (no longer active)
            completed_ids = set(self._tracks.keys()) - current_track_ids
            for track_id in completed_ids:
                track = self._tracks.pop(track_id)
                if len(track) >= self.config.min_trajectory_length:
                    self._completed_tracks.append(track)

            return list(self._tracks.values())

        except Exception as e:
            raise TrackingError(f"Tracking update failed: {e}")

    def reset(self) -> None:
        """Reset tracker state."""
        self._tracker = None
        self._tracks.clear()
        self._completed_tracks.clear()
        self._frame_id = 0
        logger.info("Tracker reset")

    @property
    def active_tracks(self) -> list[Track]:
        """Get currently active tracks."""
        return list(self._tracks.values())

    @property
    def completed_tracks(self) -> list[Track]:
        """Get completed (lost) tracks."""
        return self._completed_tracks.copy()

    @property
    def all_tracks(self) -> list[Track]:
        """Get all tracks (active + completed)."""
        return self.active_tracks + self.completed_tracks
