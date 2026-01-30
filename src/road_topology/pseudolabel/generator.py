"""End-to-end pseudo-label generation pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from road_topology.core.config import Config
from road_topology.core.exceptions import PseudoLabelError
from road_topology.core.logging import get_logger, log_duration
from road_topology.core.types import PseudoLabelResult, Trajectory
from road_topology.detection.detector import VehicleDetector
from road_topology.pseudolabel.confidence import compute_density_confidence, apply_edge_decay
from road_topology.pseudolabel.mask_builder import TrajectoryMaskBuilder
from road_topology.pseudolabel.refinement import refine_road_mask
from road_topology.tracking.tracker import VehicleTracker
from road_topology.tracking.trajectory import TrajectoryAccumulator

logger = get_logger(__name__)


@dataclass
class PseudoLabelConfig:
    """Configuration for pseudo-label generation."""
    trajectory_width: int = 50
    mask_threshold: float = 0.1
    min_trajectory_length: int = 10
    smooth_window: int = 5
    max_jump: float = 100.0
    frame_skip: int = 1
    refine_mask: bool = True
    save_intermediates: bool = False


class PseudoLabelGenerator:
    """End-to-end pseudo-label generation pipeline.

    Combines detection, tracking, and mask building to generate
    road segmentation pseudo-labels from video.

    Args:
        detector: Vehicle detector instance.
        tracker: Vehicle tracker instance.
        config: Pseudo-label generation config.
    """

    def __init__(
        self,
        detector: VehicleDetector,
        tracker: VehicleTracker,
        config: PseudoLabelConfig | None = None,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.config = config or PseudoLabelConfig()
        self._accumulator = TrajectoryAccumulator(
            min_length=self.config.min_trajectory_length,
            smooth_window=self.config.smooth_window,
            max_jump=self.config.max_jump,
        )

    def process_video(
        self,
        video_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> PseudoLabelResult:
        """Generate pseudo-labels from video file.

        Args:
            video_path: Path to input video.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            PseudoLabelResult with mask, confidence, and trajectories.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise PseudoLabelError(f"Video not found: {video_path}")

        with log_duration(logger, "process_video", video_path=str(video_path)):
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise PseudoLabelError(f"Failed to open video: {video_path}")

            try:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                logger.info(
                    "Processing video",
                    total_frames=total_frames,
                    resolution=f"{width}x{height}",
                    fps=fps,
                )

                # Reset state
                self.tracker.reset()
                self._accumulator.clear()

                frame_id = 0
                processed = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Skip frames if configured
                    if frame_id % (self.config.frame_skip + 1) != 0:
                        frame_id += 1
                        continue

                    # Detect vehicles
                    detections = self.detector.detect(frame)

                    # Update tracker
                    active_tracks = self.tracker.update(detections, frame_id)

                    frame_id += 1
                    processed += 1

                    if progress_callback:
                        progress_callback(processed, total_frames // (self.config.frame_skip + 1))

                # Collect all tracks
                all_tracks = self.tracker.all_tracks
                self._accumulator.add_tracks(all_tracks)

                logger.info(
                    "Tracking complete",
                    total_tracks=len(all_tracks),
                    trajectories=len(self._accumulator),
                )

                # Build mask
                return self._build_result((height, width))

            finally:
                cap.release()

    def process_frames(
        self,
        frames: list[np.ndarray],
    ) -> PseudoLabelResult:
        """Generate pseudo-labels from frame list.

        Args:
            frames: List of BGR images.

        Returns:
            PseudoLabelResult with mask, confidence, and trajectories.
        """
        if not frames:
            raise PseudoLabelError("Empty frame list")

        height, width = frames[0].shape[:2]

        # Reset state
        self.tracker.reset()
        self._accumulator.clear()

        for frame_id, frame in enumerate(frames):
            if frame_id % (self.config.frame_skip + 1) != 0:
                continue

            detections = self.detector.detect(frame)
            self.tracker.update(detections, frame_id)

        # Collect tracks
        self._accumulator.add_tracks(self.tracker.all_tracks)

        return self._build_result((height, width))

    def _build_result(
        self,
        frame_shape: tuple[int, int],
    ) -> PseudoLabelResult:
        """Build final pseudo-label result.

        Args:
            frame_shape: (height, width) of output mask.

        Returns:
            PseudoLabelResult.
        """
        trajectories = self._accumulator.get_trajectories(smooth=True)

        if not trajectories:
            logger.warning("No trajectories collected, returning empty mask")
            return PseudoLabelResult(
                mask=np.zeros(frame_shape, dtype=np.uint8),
                confidence=np.zeros(frame_shape, dtype=np.float32),
                trajectories=[],
                metadata={"trajectory_count": 0},
            )

        # Build mask
        builder = TrajectoryMaskBuilder(
            frame_shape=frame_shape,
            trajectory_width=self.config.trajectory_width,
        )
        builder.add_trajectories(trajectories)

        mask = builder.build_semantic_mask(threshold=self.config.mask_threshold)

        # Refine if configured
        if self.config.refine_mask:
            binary = (mask > 0).astype(np.uint8)
            binary = refine_road_mask(binary)
            mask = np.where(binary > 0, mask, 0)

        # Compute confidence
        density_conf = compute_density_confidence(trajectories, frame_shape)
        confidence = apply_edge_decay(density_conf, mask > 0)

        metadata = {
            "trajectory_count": len(trajectories),
            "total_points": sum(len(t) for t in trajectories),
            "coverage_ratio": builder.coverage_ratio,
            "config": {
                "trajectory_width": self.config.trajectory_width,
                "mask_threshold": self.config.mask_threshold,
            },
        }

        logger.info(
            "Pseudo-labels generated",
            trajectory_count=len(trajectories),
            coverage_ratio=f"{builder.coverage_ratio:.2%}",
        )

        return PseudoLabelResult(
            mask=mask,
            confidence=confidence,
            trajectories=trajectories,
            metadata=metadata,
        )

    def save_result(
        self,
        result: PseudoLabelResult,
        output_dir: Path,
        prefix: str = "pseudo",
    ) -> dict[str, Path]:
        """Save pseudo-label result to files.

        Args:
            result: PseudoLabelResult to save.
            output_dir: Output directory.
            prefix: Filename prefix.

        Returns:
            Dictionary of saved file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Save mask
        mask_path = output_dir / f"{prefix}_mask.png"
        cv2.imwrite(str(mask_path), result.mask)
        paths["mask"] = mask_path

        # Save confidence
        conf_path = output_dir / f"{prefix}_confidence.npy"
        np.save(conf_path, result.confidence)
        paths["confidence"] = conf_path

        # Save visualization
        vis_path = output_dir / f"{prefix}_visualization.png"
        vis = self._create_visualization(result.mask, result.confidence)
        cv2.imwrite(str(vis_path), vis)
        paths["visualization"] = vis_path

        logger.info("Saved pseudo-labels", output_dir=str(output_dir))

        return paths

    def _create_visualization(
        self,
        mask: np.ndarray,
        confidence: np.ndarray,
    ) -> np.ndarray:
        """Create colored visualization of pseudo-labels.

        Args:
            mask: Semantic segmentation mask.
            confidence: Confidence map.

        Returns:
            BGR visualization image.
        """
        from road_topology.core.types import CLASS_COLORS

        h, w = mask.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in CLASS_COLORS.items():
            vis[mask == class_id] = color

        # Overlay confidence as alpha
        conf_uint8 = (confidence * 255).astype(np.uint8)
        conf_colored = cv2.applyColorMap(conf_uint8, cv2.COLORMAP_JET)

        # Blend
        alpha = 0.7
        result = cv2.addWeighted(vis, alpha, conf_colored, 1 - alpha, 0)

        return result


def create_generator(config: Config | None = None) -> PseudoLabelGenerator:
    """Factory function to create PseudoLabelGenerator.

    Args:
        config: Optional configuration.

    Returns:
        Configured PseudoLabelGenerator.
    """
    config = config or Config()

    detector = VehicleDetector(
        model_path=config.detection.model_name,
        config=config.detection,
    )

    tracker = VehicleTracker(config=config.tracking)

    pl_config = PseudoLabelConfig(
        trajectory_width=50,
        min_trajectory_length=config.tracking.min_trajectory_length,
        smooth_window=config.tracking.smooth_window,
    )

    return PseudoLabelGenerator(detector, tracker, pl_config)
