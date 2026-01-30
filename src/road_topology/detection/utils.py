"""Video processing utilities for detection."""
from __future__ import annotations

from pathlib import Path
from typing import Generator, Iterator

import cv2
import numpy as np

from road_topology.core.exceptions import VideoProcessingError
from road_topology.core.logging import get_logger

logger = get_logger(__name__)


class VideoFrameIterator:
    """Iterator for reading video frames with context management.

    Provides an iterable interface for OpenCV video capture with
    automatic resource cleanup.

    Args:
        video_path: Path to video file.
        start_frame: Frame index to start reading from (default: 0).
        end_frame: Frame index to stop reading at (default: None = read all).
        stride: Frame skip interval (default: 1 = every frame).

    Example:
        >>> with VideoFrameIterator("video.mp4") as frames:
        ...     for frame_id, frame in frames:
        ...         process(frame)
    """

    def __init__(
        self,
        video_path: str | Path,
        start_frame: int = 0,
        end_frame: int | None = None,
        stride: int = 1,
    ) -> None:
        self.video_path = Path(video_path)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.stride = stride

        if not self.video_path.exists():
            raise VideoProcessingError(
                f"Video file not found: {self.video_path}",
                video_path=str(self.video_path),
            )

        self.cap: cv2.VideoCapture | None = None
        self._current_frame = 0
        self._total_frames = 0
        self._fps = 0.0
        self._width = 0
        self._height = 0

    def __enter__(self) -> "VideoFrameIterator":
        """Open video capture."""
        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise VideoProcessingError(
                f"Failed to open video: {self.video_path}",
                video_path=str(self.video_path),
            )

        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Seek to start frame
        if self.start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self._current_frame = self.start_frame

        logger.info(
            "Opened video",
            video_path=str(self.video_path),
            total_frames=self._total_frames,
            fps=self._fps,
            resolution=f"{self._width}x{self._height}",
        )

        return self

    def __exit__(self, *args) -> None:
        """Release video capture."""
        if self.cap is not None:
            self.cap.release()
            logger.info("Released video capture", video_path=str(self.video_path))

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over video frames.

        Yields:
            Tuple of (frame_id, frame) where frame is a BGR image.
        """
        if self.cap is None:
            raise VideoProcessingError(
                "VideoFrameIterator must be used as context manager",
                video_path=str(self.video_path),
            )

        while True:
            # Check if we've reached end frame
            if self.end_frame is not None and self._current_frame >= self.end_frame:
                break

            ret, frame = self.cap.read()
            if not ret:
                break

            # Yield current frame
            frame_id = self._current_frame
            yield frame_id, frame

            # Skip frames according to stride
            self._current_frame += self.stride
            if self.stride > 1:
                # Seek to next frame to process
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame)

    @property
    def total_frames(self) -> int:
        """Get total number of frames in video."""
        return self._total_frames

    @property
    def fps(self) -> float:
        """Get video frames per second."""
        return self._fps

    @property
    def resolution(self) -> tuple[int, int]:
        """Get video resolution as (width, height)."""
        return (self._width, self._height)

    @property
    def current_frame(self) -> int:
        """Get current frame index."""
        return self._current_frame


def read_frames_batch(
    video_path: str | Path,
    batch_size: int = 4,
    start_frame: int = 0,
    end_frame: int | None = None,
    stride: int = 1,
) -> Generator[list[tuple[int, np.ndarray]], None, None]:
    """Read video frames in batches.

    Args:
        video_path: Path to video file.
        batch_size: Number of frames per batch.
        start_frame: Frame index to start reading from.
        end_frame: Frame index to stop reading at (None = read all).
        stride: Frame skip interval.

    Yields:
        List of (frame_id, frame) tuples for each batch.

    Example:
        >>> for batch in read_frames_batch("video.mp4", batch_size=4):
        ...     frame_ids = [fid for fid, _ in batch]
        ...     frames = [f for _, f in batch]
        ...     process_batch(frames)
    """
    with VideoFrameIterator(video_path, start_frame, end_frame, stride) as frames:
        batch = []
        for frame_id, frame in frames:
            batch.append((frame_id, frame))

            if len(batch) >= batch_size:
                yield batch
                batch = []

        # Yield remaining frames as final batch
        if batch:
            yield batch


def get_video_info(video_path: str | Path) -> dict[str, int | float]:
    """Get video metadata without reading frames.

    Args:
        video_path: Path to video file.

    Returns:
        Dictionary with keys: total_frames, fps, width, height, duration_sec.

    Raises:
        VideoProcessingError: If video cannot be opened.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise VideoProcessingError(
            f"Video file not found: {video_path}",
            video_path=str(video_path),
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoProcessingError(
            f"Failed to open video: {video_path}",
            video_path=str(video_path),
        )

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps if fps > 0 else 0.0

        return {
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "duration_sec": duration_sec,
        }
    finally:
        cap.release()


def write_video(
    output_path: str | Path,
    frames: Iterator[np.ndarray],
    fps: float = 30.0,
    fourcc: str = "mp4v",
) -> int:
    """Write frames to video file.

    Args:
        output_path: Output video file path.
        frames: Iterator of BGR frames to write.
        fps: Output video FPS.
        fourcc: Video codec fourcc code.

    Returns:
        Number of frames written.

    Raises:
        VideoProcessingError: If video writer cannot be initialized.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    frame_count = 0

    try:
        for frame in frames:
            if writer is None:
                # Initialize writer with first frame's dimensions
                height, width = frame.shape[:2]
                fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc_code,
                    fps,
                    (width, height),
                )

                if not writer.isOpened():
                    raise VideoProcessingError(
                        f"Failed to initialize video writer: {output_path}",
                        video_path=str(output_path),
                    )

            writer.write(frame)
            frame_count += 1

        logger.info(
            "Wrote video",
            output_path=str(output_path),
            frame_count=frame_count,
            fps=fps,
        )

        return frame_count

    finally:
        if writer is not None:
            writer.release()
