"""Configuration management for Road Topology Segmentation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "console"  # console or json
    file: Path | None = None


class PathsConfig(BaseModel):
    """Path configuration."""
    models: Path = Path("./models")
    data: Path = Path("./data")
    outputs: Path = Path("./outputs")


class DetectionConfig(BaseModel):
    """YOLOv8 detection configuration."""
    model_name: str = "yolov8m.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    vehicle_classes: list[int] = Field(default_factory=lambda: [2, 3, 5, 7])
    batch_size: int = 4
    half_precision: bool = True
    device: str = "auto"

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device value."""
        allowed = {"auto", "cuda", "mps", "cpu"}
        if v not in allowed:
            raise ValueError(f"device must be one of {allowed}, got {v}")
        return v


class TrackingConfig(BaseModel):
    """ByteTrack tracking configuration."""
    tracker_type: str = "bytetrack"
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    min_trajectory_length: int = 10
    smooth_window: int = 5
    max_gap: int = 5


class SegmentationConfig(BaseModel):
    """Segmentation model configuration."""
    model_type: str = "segformer"
    backbone: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
    num_classes: int = 5
    pretrained: bool = True
    image_size: tuple[int, int] = (512, 512)


class SAMConfig(BaseModel):
    """SAM model configuration."""
    model_type: str = "vit_h"
    checkpoint: Path | None = None
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    box_nms_thresh: float = 0.7
    bbox_expansion: float = 0.1


class TrainingConfig(BaseModel):
    """Training configuration."""
    epochs: int = 100
    batch_size: int = 8
    num_workers: int = 4
    lr: float = 6e-5
    weight_decay: float = 0.01
    mixed_precision: bool = True
    gradient_accumulation: int = 2
    early_stopping_patience: int = 10


class CVATConfig(BaseModel):
    """CVAT integration configuration."""
    url: str = "http://localhost:8080"
    username: str = ""
    password: str = ""
    project_name: str = "road-topology-labeling"
    export_format: str = "COCO 1.0"


class Config(BaseSettings):
    """Main configuration class."""
    model_config = SettingsConfigDict(
        env_prefix="ROAD_TOPO_",
        env_nested_delimiter="__",
    )

    project_name: str = "road-topology"
    seed: int = 42
    device: str = "auto"

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device value."""
        allowed = {"auto", "cuda", "mps", "cpu"}
        if v not in allowed:
            raise ValueError(f"device must be one of {allowed}, got {v}")
        return v

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    sam: SAMConfig = Field(default_factory=SAMConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    cvat: CVATConfig = Field(default_factory=CVATConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**_flatten_config(data))

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


def _flatten_config(data: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested config dict for Pydantic."""
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                result[f"{key}_{sub_key}" if key != "project" else sub_key] = sub_value
        else:
            result[key] = value
    return result


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration from file or use defaults."""
    if config_path and config_path.exists():
        return Config.from_yaml(config_path)
    return Config()
