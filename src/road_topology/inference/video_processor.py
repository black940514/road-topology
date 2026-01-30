"""Video processing for road topology segmentation."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from tqdm import tqdm

from road_topology.core.exceptions import VideoProcessingError
from road_topology.core.logging import get_logger
from road_topology.core.types import PredictionResult
from road_topology.inference.predictor import RoadTopologyPredictor

logger = get_logger(__name__)


class VideoProcessor:
    """Process videos frame-by-frame with segmentation.

    Args:
        predictor: Trained predictor instance.
        output_dir: Directory for saving output videos.
        fps: Output video FPS (None = use source FPS).
        save_masks: Whether to save raw masks alongside visualization.
    """

    def __init__(
        self,
        predictor: RoadTopologyPredictor,
        output_dir: Path,
        fps: int | None = None,
        save_masks: bool = False,
    ) -> None:
        self.predictor = predictor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.save_masks = save_masks

    def process_video(
        self,
        video_path: Path,
        output_name: str | None = None,
        overlay_alpha: float = 0.5,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Process a video file and save segmented output.

        Args:
            video_path: Input video path.
            output_name: Output filename (default: input_name_segmented.mp4).
            overlay_alpha: Transparency of segmentation overlay [0, 1].
            progress_callback: Optional callback(current_frame, total_frames).

        Returns:
            Path to output video.

        Raises:
            VideoProcessingError: If video processing fails.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise VideoProcessingError(f"Video not found: {video_path}")

        try:
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise VideoProcessingError(f"Failed to open video: {video_path}")

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            output_fps = self.fps if self.fps is not None else source_fps

            # Prepare output path
            if output_name is None:
                output_name = f"{video_path.stem}_segmented.mp4"
            output_path = self.output_dir / output_name

            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                output_fps,
                (width, height),
            )

            logger.info(
                f"Processing video: {video_path.name} "
                f"({total_frames} frames @ {source_fps:.1f} FPS)"
            )

            # Process frames
            frame_idx = 0
            with tqdm(total=total_frames, desc="Processing frames") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Predict
                    result = self.predictor.predict(frame_rgb)

                    # Create overlay
                    overlay = self._create_overlay(
                        frame,
                        result.mask,
                        alpha=overlay_alpha,
                    )

                    # Add FPS info
                    fps_text = f"FPS: {1000 / result.inference_time_ms:.1f}"
                    cv2.putText(
                        overlay,
                        fps_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                    # Write frame
                    out.write(overlay)

                    # Save mask if requested
                    if self.save_masks:
                        mask_dir = self.output_dir / "masks" / video_path.stem
                        mask_dir.mkdir(parents=True, exist_ok=True)
                        mask_path = mask_dir / f"frame_{frame_idx:06d}.png"
                        cv2.imwrite(str(mask_path), result.mask)

                    # Progress callback
                    if progress_callback is not None:
                        progress_callback(frame_idx + 1, total_frames)

                    frame_idx += 1
                    pbar.update(1)

            cap.release()
            out.release()

            logger.info(f"Saved segmented video to {output_path}")
            return output_path

        except Exception as e:
            raise VideoProcessingError(
                f"Failed to process video: {e}",
                video_path=str(video_path),
            )

    def _create_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Create segmentation overlay on image.

        Args:
            image: Original BGR image.
            mask: Segmentation mask with class indices.
            alpha: Overlay transparency.

        Returns:
            Overlayed BGR image.
        """
        from road_topology.core.types import CLASS_COLORS

        # Create colored mask
        colored_mask = np.zeros_like(image)
        for class_id, color in CLASS_COLORS.items():
            # OpenCV uses BGR, CLASS_COLORS are RGB
            colored_mask[mask == class_id] = color[::-1]

        # Blend with original
        overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

        return overlay

    def process_frames(
        self,
        frames: list[np.ndarray],
        show_progress: bool = True,
    ) -> list[PredictionResult]:
        """Process a list of frames.

        Args:
            frames: List of RGB frames.
            show_progress: Whether to show progress bar.

        Returns:
            List of prediction results.
        """
        results = []

        iterator = tqdm(frames, desc="Processing frames") if show_progress else frames

        for frame in iterator:
            result = self.predictor.predict(frame)
            results.append(result)

        return results

    def get_frame_at_time(
        self,
        video_path: Path,
        timestamp: float,
    ) -> np.ndarray:
        """Extract a single frame at specific timestamp.

        Args:
            video_path: Input video path.
            timestamp: Time in seconds.

        Returns:
            RGB frame array.

        Raises:
            VideoProcessingError: If frame extraction fails.
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise VideoProcessingError(f"Failed to open video: {video_path}")

            # Seek to timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

            ret, frame = cap.read()
            cap.release()

            if not ret:
                raise VideoProcessingError(
                    f"Failed to read frame at {timestamp}s",
                    video_path=str(video_path),
                )

            # Convert BGR to RGB
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        except Exception as e:
            raise VideoProcessingError(
                f"Failed to extract frame: {e}",
                video_path=str(video_path),
            )

    def extract_frames(
        self,
        video_path: Path,
        interval: float = 1.0,
        max_frames: int | None = None,
    ) -> list[np.ndarray]:
        """Extract frames at regular intervals.

        Args:
            video_path: Input video path.
            interval: Time interval in seconds between frames.
            max_frames: Maximum number of frames to extract.

        Returns:
            List of RGB frames.

        Raises:
            VideoProcessingError: If extraction fails.
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise VideoProcessingError(f"Failed to open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps * interval)

            frames = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)

                    if max_frames is not None and len(frames) >= max_frames:
                        break

                frame_idx += 1

            cap.release()

            logger.info(f"Extracted {len(frames)} frames from {video_path.name}")
            return frames

        except Exception as e:
            raise VideoProcessingError(
                f"Failed to extract frames: {e}",
                video_path=str(video_path),
            )
