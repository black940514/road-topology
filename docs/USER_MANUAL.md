# Road Topology Segmentation - User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start Guide](#quick-start-guide)
4. [Configuration](#configuration)
5. [CLI Reference](#cli-reference)
6. [Datasets](#datasets)
7. [Training Guide](#training-guide)
8. [Inference Guide](#inference-guide)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Topics](#advanced-topics)
12. [Examples](#examples)

---

## Introduction

### Project Overview

Road Topology Segmentation is a comprehensive system for segmenting road infrastructure elements from CCTV video feeds. The system uses a multi-stage pipeline combining vehicle trajectory analysis, zero-shot segmentation with Segment Anything Model (SAM), and deep neural networks for final road topology classification.

### Key Capabilities

- **Trajectory-based Pseudo-Labeling**: Automatically generates training labels from vehicle trajectories
- **Zero-shot Segmentation**: Leverages SAM for segmentation without task-specific training
- **High-Precision Segmentation**: Uses SegFormer/Mask2Former models for production inference
- **Human-in-the-Loop**: CVAT integration for manual annotation and refinement
- **Multi-class Classification**: Segments 5 distinct road infrastructure classes

### Target Classes

The system recognizes and segments five classes in road scenes:

| Class ID | Class Name | Color | Description |
|----------|-----------|-------|-------------|
| 0 | Background | Black (0, 0, 0) | Non-road regions (buildings, trees, sky) |
| 1 | Road | Gray (128, 128, 128) | Main road surface |
| 2 | Lane | White (255, 255, 255) | Lane markings and divisions |
| 3 | Crosswalk | Yellow (255, 255, 0) | Pedestrian crossing areas |
| 4 | Sidewalk | Green (0, 255, 0) | Pedestrian walkways |

### Architecture Overview

```
Input Video
    |
    v
Vehicle Detection (YOLOv8)
    |
    +----> Vehicle Tracking (ByteTrack)
    |
    v
Trajectory Extraction & Smoothing
    |
    v
Pseudo-Label Generation (Trajectory -> Mask)
    |
    v
SAM Zero-shot Segmentation (Optional)
    |
    v
Confidence Weighting & Refinement
    |
    v
Training Dataset with Confidence Maps
    |
    v
SegFormer/Mask2Former Training
    |
    v
Final Segmentation Model
    |
    v
Production Inference (Image/Video/Batch)
```

### Key Components

- **Detection Module**: YOLOv8 for vehicle detection with configurable confidence thresholds
- **Tracking Module**: ByteTrack for multi-object tracking with configurable parameters
- **Pseudo-Label Generator**: Converts vehicle trajectories into semantic segmentation masks
- **Foundation Models**: SAM for zero-shot segmentation prompting
- **Segmentation Models**: SegFormer B2 backbone with configurable heads
- **Training Pipeline**: Mixed-precision training with gradient accumulation
- **Inference Engine**: Optimized predictor with ONNX support

---

## Installation

### System Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| Python | 3.10 or 3.11 or 3.12 | Tested on Python 3.10+ |
| CUDA | 11.8+ | For GPU acceleration (recommended) |
| GPU VRAM | 16 GB+ | For training; 8 GB for inference |
| System RAM | 16 GB+ | For data loading and preprocessing |
| Disk Space | 50 GB+ | For models, datasets, and outputs |

### Quick Installation

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd road_topology
```

#### 2. Create Virtual Environment with uv

The project uses `uv` for Python package management. If you don't have `uv` installed:

```bash
# Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install uv (Windows)
powershell -ExecutionPolicy BypassPolicy -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 3. Install Dependencies

```bash
# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in development mode
uv pip install -e ".[dev]"

# Install PyTorch with CUDA support (if using GPU)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 4. Download Model Weights

```bash
# Download SAM weights
python scripts/download_models.py --sam vit_h

# Download YOLOv8 weights (auto-downloaded on first use)
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

### Optional: Grounded-SAM Installation

For advanced prompting with Grounded-SAM:

```bash
# Clone Grounded-SAM repository
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything

# Install dependencies
pip install -e .

# Download GroundingDINO weights
python -m pip install huggingface-hub
```

### Verify Installation

```bash
# Check version and system info
road-topo version

# Show detailed system information
road-topo info

# Output should show:
# - PyTorch version
# - CUDA availability
# - CUDA device name (if available)
```

---

## Quick Start Guide

### Complete Workflow in 5 Steps

This section shows the fastest path from video to trained model.

#### Step 1: Generate Pseudo-Labels from Video

```bash
road-topo pseudolabel generate \
  --video /path/to/video.mp4 \
  --output ./data/pseudo_labels \
  --config configs/default.yaml \
  --trajectory-width 50 \
  --threshold 0.1
```

**Output Structure:**
```
./data/pseudo_labels/
  ├── masks/
  │   ├── frame_0000.png
  │   ├── frame_0001.png
  │   └── ...
  ├── confidence/
  │   ├── frame_0000.npy
  │   ├── frame_0001.npy
  │   └── ...
  ├── images/
  │   ├── frame_0000.jpg
  │   └── ...
  └── metadata.json
```

#### Step 2: Visualize Pseudo-Labels (Optional)

```bash
road-topo pseudolabel visualize \
  --mask ./data/pseudo_labels/masks/frame_0000.png \
  --video /path/to/video.mp4 \
  --output ./visualization.png
```

#### Step 3: Prepare Dataset Structure

```bash
# Create dataset directory
mkdir -p ./data/dataset/{train,val,test}/{images,masks,confidence}

# Copy pseudo-labeled data (80% train, 10% val, 10% test)
# The dataset module provides automatic splitting
```

Or use the automatic split helper:

```python
from road_topology.segmentation.dataset import RoadTopologyDataset
from road_topology.segmentation.transforms import get_train_transforms, get_val_transforms

train_ds, val_ds, test_ds = RoadTopologyDataset.create_splits(
    root="./data/dataset",
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42,
    transforms_train=get_train_transforms((512, 512)),
    transforms_val=get_val_transforms((512, 512)),
)
```

#### Step 4: Train Segmentation Model

```bash
road-topo train run \
  --config configs/training.yaml \
  --data ./data/dataset \
  --output ./outputs/model_v1 \
  --epochs 100 \
  --batch-size 8
```

**Training Progress:**
- Loss curves plotted in real-time
- Checkpoints saved every epoch
- Best model saved based on mIoU metric
- Early stopping after 10 epochs without improvement

#### Step 5: Run Inference on New Data

```bash
# Single image inference
road-topo infer image \
  --model ./outputs/model_v1/best_model.pth \
  --input test_image.jpg \
  --output result.png

# Video inference
road-topo infer video \
  --model ./outputs/model_v1/best_model.pth \
  --input test_video.mp4 \
  --output result_video.mp4

# Batch inference on directory
road-topo infer batch \
  --model ./outputs/model_v1/best_model.pth \
  --input ./test_images \
  --output ./results \
  --pattern "*.jpg"
```

---

## Configuration

### Configuration Files

Configuration is managed through YAML files in the `configs/` directory. Each configuration file handles a specific component.

### default.yaml

Main project configuration with logging and path settings.

```yaml
project:
  name: "road-topology"
  seed: 42
  device: "cuda"

logging:
  level: "INFO"        # DEBUG, INFO, WARNING, ERROR
  format: "console"    # console or json
  file: null           # Optional log file path

paths:
  models: "./models"
  data: "./data"
  outputs: "./outputs"
```

**Environment Variables:**
```bash
export ROAD_TOPO_PROJECT__SEED=42
export ROAD_TOPO_DEVICE=cuda
export ROAD_TOPO_LOGGING__LEVEL=DEBUG
```

### detection.yaml

YOLOv8 vehicle detection configuration.

```yaml
model:
  name: "yolov8m.pt"           # Model size: n, s, m, l, x
  confidence_threshold: 0.5    # Detection confidence [0, 1]
  iou_threshold: 0.45          # NMS IOU threshold

vehicle_classes:
  - 2    # car (COCO class 2)
  - 3    # motorcycle (COCO class 3)
  - 5    # bus (COCO class 5)
  - 7    # truck (COCO class 7)

inference:
  batch_size: 4
  half_precision: true          # FP16 for speed
  device: "cuda"
```

**Model Size Comparison:**

| Model | Params | Speed (fps) | Accuracy | Memory |
|-------|--------|------------|----------|--------|
| yolov8n | 3.2M | 80 | 80.4 | 1 GB |
| yolov8s | 11.2M | 40 | 86.6 | 3 GB |
| yolov8m | 25.9M | 20 | 88.7 | 5 GB |
| yolov8l | 43.7M | 10 | 89.7 | 8 GB |
| yolov8x | 68.2M | 8 | 90.8 | 11 GB |

### tracking.yaml

ByteTrack vehicle trajectory tracking configuration.

```yaml
tracker:
  type: "bytetrack"              # Tracking algorithm
  track_thresh: 0.5              # Detection threshold for tracking
  track_buffer: 30               # Frames to keep lost tracks
  match_thresh: 0.8              # IoU threshold for matching

trajectory:
  min_trajectory_length: 10      # Minimum frames to form track
  smooth_window: 5               # Moving average window for smoothing
  max_gap: 5                     # Max frames to bridge gaps in tracking
```

**Parameter Tuning:**

- Increase `track_buffer` for scenes with occlusions
- Decrease `match_thresh` for crowded scenes
- Increase `smooth_window` for smoother trajectories
- Increase `min_trajectory_length` to reduce noise

### segmentation.yaml

Semantic segmentation model configuration.

```yaml
model:
  type: "segformer"              # Type: segformer, mask2former
  backbone: "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
  num_classes: 5                 # Road, Lane, Crosswalk, Sidewalk, Background
  pretrained: true               # Use ImageNet pretrained weights

input:
  image_size: [512, 512]         # Input resolution (H, W)
  normalize:
    mean: [0.485, 0.456, 0.406]  # ImageNet normalization
    std: [0.229, 0.224, 0.225]
```

**Available Backbones:**

```yaml
# Lightweight models (fast inference)
- nvidia/segformer-b0-finetuned-cityscapes-1024-1024
- nvidia/segformer-b1-finetuned-cityscapes-1024-1024

# Balanced models (recommended)
- nvidia/segformer-b2-finetuned-cityscapes-1024-1024
- nvidia/segformer-b3-finetuned-cityscapes-1024-1024

# Large models (best accuracy)
- nvidia/segformer-b4-finetuned-cityscapes-1024-1024
- nvidia/segformer-b5-finetuned-cityscapes-1024-1024
```

### sam.yaml

Segment Anything Model (SAM) configuration for zero-shot prompting.

```yaml
model:
  type: "vit_h"                  # vit_h, vit_l, vit_b (smaller = faster)
  checkpoint: null               # Auto-download if null

inference:
  points_per_side: 32            # Grid density for prompts (higher = more thorough)
  pred_iou_thresh: 0.88          # Prediction IoU threshold
  stability_score_thresh: 0.95    # Stability threshold for masks
  box_nms_thresh: 0.7            # NMS for overlapping masks

yolo_prompting:
  enabled: true                  # Use YOLO detections as prompts
  bbox_expansion: 0.1            # Expand boxes by 10% for prompting
  min_confidence: 0.5            # Minimum YOLO confidence to prompt
```

**SAM Model Sizes:**

| Model | Params | Latency | Use Case |
|-------|--------|---------|----------|
| vit_b | 91M | 50 ms | Resource-constrained |
| vit_l | 306M | 100 ms | Balanced |
| vit_h | 632M | 200 ms | Highest quality (recommended) |

### training.yaml

Training hyperparameter configuration.

```yaml
training:
  epochs: 100                    # Number of epochs
  batch_size: 8                  # Batch size per GPU
  num_workers: 4                 # DataLoader workers

optimizer:
  type: "adamw"                  # adamw, sgd
  lr: 6e-5                       # Learning rate (small for fine-tuning)
  weight_decay: 0.01             # L2 regularization

scheduler:
  type: "polynomial"             # polynomial, cosine, linear
  power: 0.9                     # For polynomial decay

loss:
  type: "combined"               # combined = CE + Dice
  ce_weight: 0.5                 # Cross-entropy weight
  dice_weight: 0.5               # Dice loss weight

augmentation:
  level: "medium"                # light, medium, heavy

checkpointing:
  save_best: true                # Save best validation mIoU
  save_last: true                # Save last epoch
  early_stopping_patience: 10    # Stop after N epochs without improvement

mixed_precision: true            # Automatic mixed precision (FP16/FP32)
gradient_accumulation: 2         # Accumulate gradients over N steps
```

**Learning Rate Guidance:**

- Large pre-trained model fine-tuning: 1e-5 to 5e-5
- Medium pre-trained model: 5e-5 to 1e-4
- Small model from scratch: 1e-4 to 1e-3

### cvat.yaml

CVAT annotation platform integration for human-in-the-loop workflows.

```yaml
url: "http://localhost:8080"
username: "admin"
password: "admin"
project_name: "road-topology-labeling"
export_format: "COCO 1.0"        # COCO 1.0, Instance Segmentation, Pascal VOC
```

### Loading Configuration Programmatically

```python
from road_topology.core.config import load_config

# Load from YAML file
config = load_config("configs/training.yaml")

# Access nested configurations
print(config.training.lr)           # 6e-5
print(config.segmentation.backbone) # nvidia/segformer-b2-...

# Override values
config.training.epochs = 50
config.device = "cpu"

# Save modified config
config.to_yaml("configs/custom.yaml")
```

---

## CLI Reference

### Entry Point

All commands are accessed through the `road-topo` CLI tool:

```bash
road-topo [COMMAND] [SUBCOMMAND] [OPTIONS]
```

### Global Commands

#### `road-topo version`

Display package version information.

```bash
$ road-topo version
road-topology version 0.1.0
```

#### `road-topo info`

Show system and configuration information.

```bash
$ road-topo info
Road Topology Segmentation
PyTorch version: 2.1.0
CUDA available: True
CUDA device: NVIDIA A100-PCIE-40GB
```

### Pseudo-Label Generation Commands

#### `road-topo pseudolabel generate`

Generate pseudo-labels from video using vehicle trajectories.

**Usage:**
```bash
road-topo pseudolabel generate \
  --video VIDEO_PATH \
  --output OUTPUT_DIR \
  [--config CONFIG_FILE] \
  [--trajectory-width PIXELS] \
  [--threshold FLOAT] \
  [--frame-skip N]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--video, -v` | Path | Required | Input video file path |
| `--output, -o` | Path | Required | Output directory for masks and metadata |
| `--config, -c` | Path | None | Configuration YAML file |
| `--trajectory-width` | int | 50 | Width in pixels for trajectory mask rendering |
| `--threshold` | float | 0.1 | Confidence threshold for mask generation |
| `--frame-skip` | int | 0 | Skip frames between processing (0 = process all) |

**Examples:**

```bash
# Basic pseudo-label generation
road-topo pseudolabel generate \
  --video video.mp4 \
  --output ./pseudo_labels

# Skip every other frame for faster processing
road-topo pseudolabel generate \
  --video video.mp4 \
  --output ./pseudo_labels \
  --frame-skip 1 \
  --trajectory-width 40

# Use custom configuration
road-topo pseudolabel generate \
  --video video.mp4 \
  --output ./pseudo_labels \
  --config configs/custom.yaml
```

**Output:**
```
pseudo_labels/
├── masks/
│   ├── frame_0000.png       # Semantic segmentation mask
│   ├── frame_0001.png
│   └── ...
├── confidence/
│   ├── frame_0000.npy       # Confidence score map
│   ├── frame_0001.npy
│   └── ...
├── images/
│   ├── frame_0000.jpg       # Extracted frames
│   └── ...
├── trajectories.json        # Vehicle trajectory data
└── metadata.json            # Generation metadata
```

#### `road-topo pseudolabel visualize`

Visualize pseudo-label masks with optional video overlay.

**Usage:**
```bash
road-topo pseudolabel visualize \
  --mask MASK_PATH \
  [--video VIDEO_PATH] \
  [--output OUTPUT_PATH]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--mask, -m` | Path | Required | Path to mask PNG file |
| `--video, -v` | Path | None | Optional video frame for overlay |
| `--output, -o` | Path | None | Save visualization to file (displays if None) |

**Examples:**

```bash
# Display visualization in window
road-topo pseudolabel visualize \
  --mask ./pseudo_labels/masks/frame_0000.png

# Save visualization with original frame overlay
road-topo pseudolabel visualize \
  --mask ./pseudo_labels/masks/frame_0000.png \
  --video video.mp4 \
  --output visualization.png
```

### Training Commands

#### `road-topo train run`

Train a segmentation model on prepared dataset.

**Usage:**
```bash
road-topo train run \
  --config CONFIG_FILE \
  --data DATA_DIR \
  [--output OUTPUT_DIR] \
  [--epochs N] \
  [--batch-size B] \
  [--resume CHECKPOINT]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config, -c` | Path | Required | Training configuration YAML |
| `--data, -d` | Path | Required | Dataset root directory |
| `--output, -o` | Path | `./outputs` | Directory for checkpoints and logs |
| `--epochs` | int | None | Override config epochs |
| `--batch-size, -b` | int | None | Override config batch size |
| `--resume, -r` | Path | None | Resume from checkpoint |

**Examples:**

```bash
# Train with default config
road-topo train run \
  --config configs/training.yaml \
  --data ./dataset

# Override hyperparameters
road-topo train run \
  --config configs/training.yaml \
  --data ./dataset \
  --epochs 50 \
  --batch-size 16 \
  --output ./outputs/experiment_v2

# Resume from checkpoint
road-topo train run \
  --config configs/training.yaml \
  --data ./dataset \
  --resume ./outputs/checkpoint_epoch_30.pth
```

**Output:**
```
outputs/
├── best_model.pth              # Best checkpoint (by mIoU)
├── last_model.pth              # Last epoch checkpoint
├── checkpoint_epoch_10.pth
├── checkpoint_epoch_20.pth
├── ...
├── training_log.json           # Training history
├── metrics.json                # Final metrics
└── config.yaml                 # Used configuration
```

#### `road-topo train validate`

Validate a trained model on validation dataset.

**Usage:**
```bash
road-topo train validate \
  --model MODEL_CHECKPOINT \
  --data DATA_DIR \
  [--batch-size B]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model, -m` | Path | Required | Model checkpoint path |
| `--data, -d` | Path | Required | Validation dataset directory |
| `--batch-size, -b` | int | 8 | Batch size for validation |

**Examples:**

```bash
# Validate best model
road-topo train validate \
  --model ./outputs/best_model.pth \
  --data ./dataset

# Validate with larger batch size
road-topo train validate \
  --model ./outputs/best_model.pth \
  --data ./dataset \
  --batch-size 32
```

**Output:**
```
Validating model: ./outputs/best_model.pth

Results:
  mIoU: 0.6542
  road: 0.7234
  lane: 0.6891
  crosswalk: 0.5543
  sidewalk: 0.6234
```

### Inference Commands

#### `road-topo infer image`

Run inference on a single image.

**Usage:**
```bash
road-topo infer image \
  --model MODEL_CHECKPOINT \
  --input IMAGE_PATH \
  [--output OUTPUT_PATH] \
  [--visualize/--no-visualize] \
  [--overlay/--no-overlay]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model, -m` | Path | Required | Model checkpoint path |
| `--input, -i` | Path | Required | Input image file |
| `--output, -o` | Path | None | Save result to file |
| `--visualize` | bool | True | Display result in window |
| `--overlay` | bool | True | Overlay segmentation on original image |

**Examples:**

```bash
# Basic inference with visualization
road-topo infer image \
  --model ./outputs/best_model.pth \
  --input test_image.jpg

# Save result without displaying
road-topo infer image \
  --model ./outputs/best_model.pth \
  --input test_image.jpg \
  --output result.png \
  --no-visualize

# Generate segmentation without overlay
road-topo infer image \
  --model ./outputs/best_model.pth \
  --input test_image.jpg \
  --output mask.png \
  --no-overlay
```

**Output:**
```
Running inference on: test_image.jpg

Detected classes:
  Class 0 (background): 45.23%
  Class 1 (road): 32.15%
  Class 2 (lane): 12.34%
  Class 3 (crosswalk): 5.28%
  Class 4 (sidewalk): 5.00%
```

#### `road-topo infer video`

Run inference on video file.

**Usage:**
```bash
road-topo infer video \
  --model MODEL_CHECKPOINT \
  --input VIDEO_PATH \
  --output OUTPUT_PATH \
  [--fps FPS] \
  [--overlay/--no-overlay]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model, -m` | Path | Required | Model checkpoint path |
| `--input, -i` | Path | Required | Input video file |
| `--output, -o` | Path | Required | Output video path |
| `--fps` | int | None | Output FPS (default: input FPS) |
| `--overlay` | bool | True | Overlay segmentation on frames |

**Examples:**

```bash
# Process video with same frame rate
road-topo infer video \
  --model ./outputs/best_model.pth \
  --input input.mp4 \
  --output output.mp4

# Process at custom FPS
road-topo infer video \
  --model ./outputs/best_model.pth \
  --input input.mp4 \
  --output output.mp4 \
  --fps 15

# Output masks without overlay
road-topo infer video \
  --model ./outputs/best_model.pth \
  --input input.mp4 \
  --output masks.mp4 \
  --no-overlay
```

#### `road-topo infer batch`

Run inference on directory of images.

**Usage:**
```bash
road-topo infer batch \
  --model MODEL_CHECKPOINT \
  --input INPUT_DIR \
  --output OUTPUT_DIR \
  [--pattern GLOB_PATTERN] \
  [--batch-size B]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model, -m` | Path | Required | Model checkpoint path |
| `--input, -i` | Path | Required | Input directory containing images |
| `--output, -o` | Path | Required | Output directory for results |
| `--pattern, -p` | str | `*.jpg` | File glob pattern to match |
| `--batch-size, -b` | int | 8 | Batch processing size |

**Examples:**

```bash
# Process all JPEGs
road-topo infer batch \
  --model ./outputs/best_model.pth \
  --input ./test_images \
  --output ./results

# Process PNG files with batch size 16
road-topo infer batch \
  --model ./outputs/best_model.pth \
  --input ./test_images \
  --output ./results \
  --pattern "*.png" \
  --batch-size 16

# Process images matching pattern
road-topo infer batch \
  --model ./outputs/best_model.pth \
  --input ./test_images \
  --output ./results \
  --pattern "road_*.jpg"
```

### Evaluation Commands

#### `road-topo evaluate`

Evaluate model performance on test set.

**Usage:**
```bash
road-topo evaluate \
  --model MODEL_CHECKPOINT \
  --data DATA_DIR \
  [--output OUTPUT_DIR] \
  [--batch-size B]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model, -m` | Path | Required | Model checkpoint |
| `--data, -d` | Path | Required | Test dataset directory |
| `--output, -o` | Path | None | Save evaluation report |
| `--batch-size, -b` | int | 8 | Batch size |

---

## Datasets

### Supported Public Datasets

#### BDD100K

Berkeley DeepDrive 100K dataset for autonomous driving.

**Characteristics:**
- 100,000 unlabeled images
- 10,000 images with segmentation labels
- Diverse weather, time-of-day conditions

**Download:**
```bash
# Register at https://bdd-data.berkeley.edu
# Follow instructions and download "Segmentation" data

# Extract and organize
unzip bdd100k_seg_trainval.zip -d ./data/bdd100k
```

**Preparation:**
```python
from road_topology.segmentation.dataset import RoadTopologyDataset

# Symlink BDD100K to expected structure
import os
os.symlink(
    'data/bdd100k/seg/images/train',
    'data/dataset/train/images'
)
os.symlink(
    'data/bdd100k/seg/labels/train',
    'data/dataset/train/masks'
)
```

#### Cityscapes

Large-scale urban scene understanding dataset.

**Characteristics:**
- 5,000 high-quality pixel-level labeled images
- 50 semantic classes (can map to 5 road classes)
- Stereo video sequences

**Download:**
```bash
# Register at https://www.cityscapes-dataset.com
# Download "gtFine_trainvaltest.zip" and "leftImg8bit_trainvaltest.zip"

unzip gtFine_trainvaltest.zip -d ./data/cityscapes
unzip leftImg8bit_trainvaltest.zip -d ./data/cityscapes
```

**Class Mapping:**
```python
# Map 19 Cityscapes classes to 5 road classes
CITYSCAPES_TO_ROAD = {
    0: 0,   # road -> road
    1: 0,   # sidewalk -> background
    2: 1,   # building -> road
    3: 0,   # wall -> background
    # ... more mappings ...
}
```

#### KITTI

Karlsruhe Institute of Technology and Toyota Technological Institute dataset.

**Characteristics:**
- 389 training images with semantic labels
- Autonomous driving scenarios
- High-resolution stereo sequences

**Download:**
```bash
# Register at http://www.cvlibs.net/datasets/kitti/
# Download semantic segmentation dataset

# Extract
tar -xvzf data_semantics.tar.gz -C ./data/kitti
```

### Dataset Structure

Expected directory structure for training:

```
data/
├── train/
│   ├── images/
│   │   ├── image_0001.jpg
│   │   ├── image_0002.jpg
│   │   └── ...
│   ├── masks/
│   │   ├── image_0001.png    # Grayscale with class indices
│   │   ├── image_0002.png
│   │   └── ...
│   └── confidence/
│       ├── image_0001.npy    # Confidence maps (optional)
│       └── ...
├── val/
│   ├── images/
│   ├── masks/
│   └── confidence/
└── test/
    ├── images/
    ├── masks/
    └── confidence/
```

### Creating Custom Dataset

```python
from pathlib import Path
import cv2
import numpy as np
from road_topology.segmentation.dataset import RoadTopologyDataset

# Create directory structure
dataset_root = Path("./data/custom_dataset")
for split in ["train", "val", "test"]:
    (dataset_root / split / "images").mkdir(parents=True, exist_ok=True)
    (dataset_root / split / "masks").mkdir(parents=True, exist_ok=True)
    (dataset_root / split / "confidence").mkdir(parents=True, exist_ok=True)

# Add your images and masks
# Images should be RGB JPG/PNG
# Masks should be grayscale PNG with values 0-4 (class indices)
# Confidence maps should be numpy arrays with values 0.0-1.0

# Create dataset splits
train_ds, val_ds, test_ds = RoadTopologyDataset.create_splits(
    root=dataset_root,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42
)

print(f"Train: {len(train_ds)} samples")
print(f"Val: {len(val_ds)} samples")
print(f"Test: {len(test_ds)} samples")
```

### Data Validation

```bash
# Check dataset integrity
python -c "
from pathlib import Path
from road_topology.segmentation.dataset import RoadTopologyDataset

ds = RoadTopologyDataset('./data/dataset', 'train')
print(f'Samples: {len(ds)}')

# Validate first sample
sample = ds[0]
print(f'Image shape: {sample[\"image\"].shape}')
print(f'Mask shape: {sample[\"mask\"].shape}')
print(f'Classes in mask: {set(sample[\"mask\"].numpy().flatten())}')
"
```

---

## Training Guide

### Data Preparation

#### 1. Generate Pseudo-Labels

```bash
road-topo pseudolabel generate \
  --video training_videos/video1.mp4 \
  --output ./data/pseudo_labels_v1 \
  --config configs/detection.yaml
```

#### 2. Apply Confidence Filtering (Optional)

```python
from pathlib import Path
import numpy as np

confidence_dir = Path("./data/pseudo_labels_v1/confidence")

# Filter samples with mean confidence > 0.7
for conf_file in confidence_dir.glob("*.npy"):
    conf = np.load(conf_file)
    if conf.mean() < 0.7:
        # Remove corresponding files
        mask_path = conf_file.parent.parent / "masks" / f"{conf_file.stem}.png"
        mask_path.unlink()
        conf_file.unlink()
```

#### 3. Create Dataset Splits

```bash
# Copy pseudo-labels to dataset directory
mkdir -p ./data/dataset/{train,val,test}/{images,masks,confidence}

# Use create_splits to organize data
python -c "
from road_topology.segmentation.dataset import RoadTopologyDataset
train_ds, val_ds, test_ds = RoadTopologyDataset.create_splits(
    root='./data/dataset',
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42
)
print('Dataset created successfully!')
"
```

### Hyperparameter Selection

#### Learning Rate Schedule

```yaml
# For fine-tuning pre-trained model
optimizer:
  lr: 6e-5      # Small learning rate for fine-tuning

scheduler:
  type: "polynomial"
  power: 0.9    # Polynomial decay

# Alternative: Cosine annealing
scheduler:
  type: "cosine"
  T_max: 100    # Annealing period (epochs)
```

#### Batch Size Consideration

| Batch Size | GPU Memory | Training Time | Stability |
|-----------|-----------|--------------|-----------|
| 4 | 6 GB | Slower | More stable |
| 8 | 10 GB | Moderate | Recommended |
| 16 | 18 GB | Fast | Less stable |
| 32 | 32+ GB | Very fast | Noisy gradients |

#### Loss Function Weighting

```yaml
loss:
  type: "combined"
  ce_weight: 0.5       # Cross-entropy for class imbalance
  dice_weight: 0.5     # Dice for boundary precision

# For class imbalance, adjust weights:
# - Increase Dice weight for better boundary detection
# - Increase CE weight for overall accuracy
```

### Data Augmentation

```yaml
augmentation:
  level: "medium"  # light, medium, heavy

# Light augmentation (minimal changes)
# - Small rotations (-5 to +5 degrees)
# - Small crops (90-100% of image)
# - Horizontal flip

# Medium augmentation (moderate changes)
# - Rotations (-15 to +15 degrees)
# - Crops (70-100% of image)
# - Flip and blur

# Heavy augmentation (aggressive changes)
# - Rotations (-30 to +30 degrees)
# - Crops (50-100% of image)
# - Color jitter, blur, elastic deformations
```

### Training Monitoring

```bash
# Monitor training in real-time
tail -f outputs/training_log.json

# Expected output structure:
{
  "epoch": 1,
  "train_loss": 0.523,
  "val_loss": 0.456,
  "val_miou": 0.542,
  "learning_rate": 6e-5
}
```

### Best Practices

1. **Early Stopping**: Stop after 10 epochs without mIoU improvement
2. **Mixed Precision**: Use FP16 for 30% faster training with minimal accuracy loss
3. **Gradient Accumulation**: Accumulate gradients over 2 steps for larger effective batch size
4. **Checkpoint Strategy**: Save best model by mIoU and last epoch
5. **Validation Frequency**: Validate every epoch to track progress

### Common Issues

**Issue: Out of Memory (OOM)**
```bash
# Solution: Reduce batch size or image size
road-topo train run \
  --config configs/training.yaml \
  --data ./dataset \
  --batch-size 4  # Reduce from 8
```

**Issue: Training loss not decreasing**
```yaml
# Solution: Reduce learning rate
optimizer:
  lr: 3e-5  # Reduce by 50%
```

**Issue: Validation mIoU plateauing**
```yaml
# Solution: Increase augmentation level
augmentation:
  level: "heavy"  # More aggressive augmentation
```

---

## Inference Guide

### Model Deployment

#### Single Image Inference

```python
from pathlib import Path
import torch
import cv2

from road_topology.segmentation.models import SegFormerModel
from road_topology.segmentation.transforms import get_val_transforms

# Load model
model = SegFormerModel.load("./outputs/best_model.pth")
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Load image
image = cv2.imread("test.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Preprocess
transforms = get_val_transforms((512, 512))
transformed = transforms(image=image_rgb)
input_tensor = transformed["image"].unsqueeze(0)

# Inference
with torch.no_grad():
    input_tensor = input_tensor.to(model.device)
    mask = model.predict(input_tensor)

# Get class predictions
pred_mask = mask[0].cpu().numpy()
print(f"Predicted classes: {set(pred_mask.flatten())}")
```

#### Batch Inference

```python
import torch
from torch.utils.data import DataLoader

from road_topology.segmentation.dataset import RoadTopologyDataset
from road_topology.segmentation.transforms import get_val_transforms
from road_topology.segmentation.models import SegFormerModel

# Load model
model = SegFormerModel.load("./outputs/best_model.pth")
model.eval()

# Create dataloader
dataset = RoadTopologyDataset(
    "./data/dataset",
    split="test",
    transforms=get_val_transforms()
)
loader = DataLoader(dataset, batch_size=16, num_workers=4)

# Run inference
all_masks = []
with torch.no_grad():
    for batch in loader:
        images = batch["image"].to(model.device)
        masks = model.predict(images)
        all_masks.append(masks.cpu().numpy())

# Concatenate results
import numpy as np
predictions = np.concatenate(all_masks, axis=0)
print(f"Predictions shape: {predictions.shape}")
```

### ONNX Export for Production

```python
import torch
import onnx

from road_topology.segmentation.models import SegFormerModel

# Load PyTorch model
model = SegFormerModel.load("./outputs/best_model.pth")
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 512, 512).to(model.device)

# Export to ONNX
torch.onnx.export(
    model.model,  # Internal PyTorch module
    dummy_input,
    "model.onnx",
    input_names=["image"],
    output_names=["logits"],
    opset_version=11,
    do_constant_folding=True,
)

# Verify ONNX model
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")
```

### ONNX Inference

```python
import onnxruntime as ort
import numpy as np
import cv2

# Load ONNX model
session = ort.InferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Load and preprocess image
image = cv2.imread("test.jpg")
image = cv2.resize(image, (512, 512))
image = image.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32) / 255.0

# Run inference
outputs = session.run(None, {"image": image})
logits = outputs[0]  # (1, 5, 512, 512)

# Get predictions
pred_mask = np.argmax(logits[0], axis=0)
print(f"Predicted mask shape: {pred_mask.shape}")
```

### Model Quantization

```python
import torch
import torch.quantization as quantization

from road_topology.segmentation.models import SegFormerModel

# Load model
model = SegFormerModel.load("./outputs/best_model.pth")
model.eval()

# Prepare for quantization
model.qconfig = quantization.get_default_qconfig("fbgemm")
quantization.prepare(model, inplace=True)

# Calibrate on sample data
# ... run inference on calibration dataset ...

# Convert to quantized model
quantization.convert(model, inplace=True)

# Save quantized model
torch.save(model.state_dict(), "model_quantized.pth")
```

### Performance Optimization

#### Model Compilation (PyTorch 2.0+)

```python
import torch
from road_topology.segmentation.models import SegFormerModel

# Load model
model = SegFormerModel.load("./outputs/best_model.pth")

# Compile for faster inference
model = torch.compile(model)

# Run inference
with torch.no_grad():
    output = model.predict(input_tensor)
```

#### Batch Processing Tips

| Setting | Speed Impact | Memory Impact |
|---------|------------|--------------|
| Batch size 16 vs 1 | 6-8x faster | 8x more memory |
| Mixed precision | 1.5-2x faster | Same memory |
| Model quantization | 2-4x faster | 75% less memory |
| ONNX vs PyTorch | 10-30% faster | Similar |

---

## API Reference

### Core Classes

#### SegFormerModel

Wrapper for SegFormer semantic segmentation model.

```python
from road_topology.segmentation.models import SegFormerModel

# Initialize model
model = SegFormerModel(
    backbone="nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    num_classes=5,
    pretrained=True
)

# Move to device
model.to("cuda")

# Set evaluation mode
model.eval()

# Forward pass
output = model(input_tensor)  # (B, 5, H, W)

# Get predictions
predictions = model.predict(input_tensor)  # (B, H, W)

# Load checkpoint
model = SegFormerModel.load("model.pth")

# Save checkpoint
model.save("model.pth")
```

**Methods:**
- `forward(images)`: Raw forward pass returning logits
- `predict(images)`: Returns argmax predictions (class indices)
- `load(path)`: Load from checkpoint
- `save(path)`: Save checkpoint

#### RoadTopologyDataset

PyTorch Dataset for loading paired images and masks.

```python
from road_topology.segmentation.dataset import RoadTopologyDataset
from road_topology.segmentation.transforms import get_train_transforms

# Create dataset
dataset = RoadTopologyDataset(
    root="./data/dataset",
    split="train",
    transforms=get_train_transforms((512, 512)),
    use_confidence_weights=True
)

# Access sample
sample = dataset[0]
# {
#   "image": Tensor(3, 512, 512),
#   "mask": Tensor(512, 512),
#   "confidence": Tensor(512, 512),  # if use_confidence_weights=True
#   "image_path": str
# }

# Create splits
train_ds, val_ds, test_ds = RoadTopologyDataset.create_splits(
    root="./data/dataset",
    train_ratio=0.8,
    val_ratio=0.1
)
```

**Methods:**
- `__len__()`: Number of samples
- `__getitem__(idx)`: Get sample dict
- `create_splits()`: Create train/val/test splits from flat directory

#### PseudoLabelGenerator

Generate pseudo-labels from video.

```python
from road_topology.pseudolabel.generator import create_generator
from road_topology.core.config import load_config

# Create generator
config = load_config("configs/default.yaml")
generator = create_generator(config)

# Process video
result = generator.process_video(
    video_path="input.mp4",
    progress_callback=lambda cur, tot: print(f"{cur}/{tot}")
)

# Result contains:
# - mask: (H, W) semantic segmentation
# - confidence: (H, W) confidence map
# - trajectories: List[Trajectory]
# - metadata: dict

# Save results
paths = generator.save_result(result, output_dir="./pseudo_labels")
```

#### SegmentationTrainer

Training orchestration for segmentation models.

```python
from torch.utils.data import DataLoader
from road_topology.segmentation.trainer import SegmentationTrainer
from road_topology.segmentation.models import SegFormerModel
from road_topology.core.config import TrainingConfig

# Create trainer
trainer = SegmentationTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=TrainingConfig(),
    output_dir="./outputs"
)

# Train
history = trainer.train(epochs=100)
# Returns: {
#   "best_miou": float,
#   "train_losses": List[float],
#   "val_losses": List[float],
#   "val_mious": List[float]
# }

# Resume from checkpoint
trainer.load_checkpoint("checkpoint.pth")
```

### Type Definitions

#### Trajectory

Represents vehicle path with confidence scores.

```python
from road_topology.core.types import Trajectory
import numpy as np

trajectory = Trajectory(
    points=np.array([[100, 200], [101, 205], [102, 210]]),
    confidence_scores=np.array([0.9, 0.95, 0.88]),
    track_id=1
)

# Smooth trajectory
smoothed = trajectory.smooth(window=5)

# Get mean confidence
mean_conf = trajectory.mean_confidence()

# Convert to polyline
polyline = trajectory.to_polyline()
```

#### BoundingBox

Represents object detection results.

```python
from road_topology.core.types import BoundingBox

bbox = BoundingBox(
    x1=100, y1=200, x2=300, y2=400,
    confidence=0.95,
    class_id=2,
    class_name="car"
)

# Get properties
center = bbox.center()
area = bbox.area()
width = bbox.width()
height = bbox.height()

# Expand box
expanded = bbox.expand(ratio=0.1)
```

#### PseudoLabelResult

Result from pseudo-label generation.

```python
from road_topology.core.types import PseudoLabelResult

result = PseudoLabelResult(
    mask=np.zeros((512, 512), dtype=np.uint8),
    confidence=np.ones((512, 512), dtype=np.float32),
    trajectories=[...],
    metadata={"coverage_ratio": 0.75}
)
```

### Configuration API

```python
from road_topology.core.config import Config, load_config

# Load from YAML
config = load_config("configs/training.yaml")

# Access nested configs
lr = config.training.lr
device = config.device

# Modify
config.training.epochs = 50

# Save
config.to_yaml("custom_config.yaml")

# Create from scratch
config = Config(
    project_name="my_project",
    device="cuda",
    training=TrainingConfig(
        epochs=100,
        batch_size=8,
        lr=6e-5
    )
)
```

---

## Troubleshooting

### Installation Issues

#### CUDA Not Found

**Problem:** `torch.cuda.is_available()` returns False

**Solution:**
```bash
# Verify NVIDIA GPU drivers
nvidia-smi

# Install correct PyTorch version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

#### Out of Memory During Installation

**Problem:** pip install fails with memory error

**Solution:**
```bash
# Use uv instead (more efficient)
uv pip install -e .

# Or install packages individually
pip install torch torchvision  # Install large packages first
pip install segment-anything
pip install ultralytics
```

#### Missing Model Weights

**Problem:** SAM model download fails

**Solution:**
```bash
# Manual download
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
mkdir -p ~/.cache/torch/hub/checkpoints/
mv sam_vit_h_4b8939.pth ~/.cache/torch/hub/checkpoints/

# Verify
python -c "
from segment_anything import sam_model_registry
sam = sam_model_registry['vit_h'](checkpoint='~/.cache/torch/hub/checkpoints/sam_vit_h_4b8939.pth')
print('SAM loaded successfully!')
"
```

### Data Issues

#### No Images Found in Dataset

**Problem:** `RuntimeError: No images found in /path/to/images`

**Solution:**
```bash
# Verify directory structure
ls -la ./data/dataset/train/images/

# Ensure images are in correct location
mkdir -p ./data/dataset/train/images
mkdir -p ./data/dataset/train/masks

# Copy images
cp *.jpg ./data/dataset/train/images/

# Verify
python -c "
from pathlib import Path
images = list(Path('./data/dataset/train/images').glob('*.jpg'))
print(f'Found {len(images)} images')
"
```

#### Mask Shape Mismatch

**Problem:** `ValueError: mask and image must have same height/width`

**Solution:**
```python
import cv2
from pathlib import Path

# Resize masks to match images
images_dir = Path("./data/dataset/train/images")
masks_dir = Path("./data/dataset/train/masks")

for img_path in images_dir.glob("*.jpg"):
    image = cv2.imread(str(img_path))
    h, w = image.shape[:2]

    mask_path = masks_dir / f"{img_path.stem}.png"
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask.shape != (h, w):
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(mask_path), mask_resized)
```

### Training Issues

#### NaN Loss During Training

**Problem:** Training loss becomes NaN

**Solution:**
```yaml
# Reduce learning rate
optimizer:
  lr: 3e-5  # Reduce from 6e-5

# Or use gradient clipping
training:
  gradient_clip: 1.0
```

#### Validation mIoU Not Improving

**Problem:** Validation metrics plateau or decrease

**Solution:**
```bash
# Increase training iterations
road-topo train run \
  --config configs/training.yaml \
  --data ./dataset \
  --epochs 200  # Increase from 100

# Or reduce learning rate more aggressively
# Edit configs/training.yaml
# lr: 3e-5  # Reduce
# power: 0.95  # Steeper polynomial decay
```

#### GPU Out of Memory

**Problem:** `RuntimeError: CUDA out of memory`

**Solution:**
```bash
# Reduce batch size
road-topo train run \
  --config configs/training.yaml \
  --data ./dataset \
  --batch-size 4

# Or reduce image size
# Edit configs/segmentation.yaml
# image_size: [384, 384]  # Reduce from 512

# Or use gradient accumulation
# Edit configs/training.yaml
# gradient_accumulation: 4
```

### Inference Issues

#### Model Loading Fails

**Problem:** `RuntimeError: Error(s) in loading state_dict`

**Solution:**
```python
# Check model architecture matches checkpoint
from road_topology.segmentation.models import SegFormerModel

# Verify checkpoint is valid
try:
    model = SegFormerModel.load("checkpoint.pth")
except Exception as e:
    print(f"Error: {e}")

    # Try loading with strict=False
    import torch
    model = SegFormerModel(backbone="nvidia/segformer-b2-finetuned-cityscapes-1024-1024", num_classes=5)
    state = torch.load("checkpoint.pth")
    model.model.load_state_dict(state, strict=False)
```

#### Inference is Slow

**Problem:** Inference takes >500ms per image

**Solution:**
```python
# Use smaller model
from road_topology.segmentation.models import SegFormerModel

model = SegFormerModel(
    backbone="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",  # Lighter
    num_classes=5
)

# Or export to ONNX for speed
# (See API Reference section)

# Or use model.compile() in PyTorch 2.0+
import torch
model = torch.compile(model)
```

#### ONNX Inference Results Don't Match PyTorch

**Problem:** Predictions differ between PyTorch and ONNX

**Solution:**
```python
import torch
import onnxruntime as ort
import numpy as np

# Compare outputs
pytorch_model = SegFormerModel.load("model.pth")
pytorch_model.eval()

onnx_session = ort.InferenceSession("model.onnx")

# Run on same input
dummy_input = torch.randn(1, 3, 512, 512)

with torch.no_grad():
    pytorch_out = pytorch_model(dummy_input).cpu().numpy()

onnx_out = onnx_session.run(None, {"image": dummy_input.numpy()})

# Compare
diff = np.abs(pytorch_out - onnx_out[0]).max()
print(f"Max difference: {diff}")

# If too large, re-export ONNX with opset_version=12
```

---

## Advanced Topics

### Human-in-the-Loop Workflow with CVAT

CVAT (Computer Vision Annotation Tool) integration allows human refinement of pseudo-labels.

#### Setup CVAT Server

```bash
# Install CVAT (requires Docker)
docker pull cvat/cvat:latest
docker pull cvat/cvat_db:latest

# Start CVAT
docker-compose -f docker-compose.yml up -d

# Access at http://localhost:8080
# Default: admin / 12345
```

#### Export Pseudo-Labels to CVAT

```python
from pathlib import Path
import json
from road_topology.core.types import CLASS_NAMES

# Create COCO format dataset
pseudo_labels = Path("./pseudo_labels")
output = {"images": [], "annotations": [], "categories": []}

# Add categories
for class_id, class_name in enumerate(CLASS_NAMES):
    output["categories"].append({
        "id": class_id,
        "name": class_name
    })

# Add images and segmentations
image_id = 0
annotation_id = 0

for mask_file in (pseudo_labels / "masks").glob("*.png"):
    image_id += 1
    output["images"].append({
        "id": image_id,
        "file_name": f"{mask_file.stem}.jpg",
        "height": 720,
        "width": 1280
    })

    # Create segmentation annotation
    # (Convert mask to COCO RLE format)
    annotation_id += 1
    output["annotations"].append({
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,  # Road class
        "area": 1000,
        "bbox": [0, 0, 100, 100],
        "iscrowd": 0,
        "segmentation": []  # RLE format
    })

# Save COCO dataset
with open("dataset_coco.json", "w") as f:
    json.dump(output, f)
```

#### Refined Labels from CVAT

```python
import json
from pathlib import Path

# Download annotations from CVAT as COCO JSON
with open("cvat_export.json") as f:
    cvat_data = json.load(f)

# Convert back to mask format
for annotation in cvat_data["annotations"]:
    image_id = annotation["image_id"]
    segmentation = annotation["segmentation"]

    # Decode RLE to mask
    # ... (use pycocotools.mask.decode)

    # Save refined mask
    mask_path = Path("./refined_masks") / f"image_{image_id:04d}.png"
```

#### CVAT Configuration

Update `configs/cvat.yaml` for your CVAT instance:

```yaml
url: "http://localhost:8080"
username: "admin"
password: "your_password"
project_name: "road-topology-refinement"
export_format: "COCO 1.0"
```

### Transfer Learning from Cityscapes

Fine-tune a model pre-trained on Cityscapes for your specific road topology task.

```python
from road_topology.segmentation.models import SegFormerModel

# Load Cityscapes pre-trained model
model = SegFormerModel(
    backbone="nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    num_classes=5,  # Your task
    pretrained=True
)

# For fine-tuning, use smaller learning rate
# and consider freezing early layers
```

**Fine-tuning Strategy:**
1. Freeze backbone (0 learning rate)
2. Train only decoder head (5x learning rate): 20 epochs
3. Unfreeze backbone layers 4-5 (1x learning rate): 30 epochs
4. Unfreeze all (0.1x learning rate): final epochs

### Model Ensemble

Combine predictions from multiple models for higher accuracy.

```python
import torch
import numpy as np

from road_topology.segmentation.models import SegFormerModel

# Load multiple models
model1 = SegFormerModel.load("model_seed_42.pth")
model2 = SegFormerModel.load("model_seed_123.pth")
model3 = SegFormerModel.load("model_seed_456.pth")

models = [model1, model2, model3]

# Ensemble predictions
def ensemble_predict(image_tensor, models):
    logits_list = []

    with torch.no_grad():
        for model in models:
            model.eval()
            logits = model(image_tensor)  # (B, C, H, W)
            logits_list.append(logits)

    # Average logits
    avg_logits = torch.stack(logits_list).mean(dim=0)

    # Get predictions
    predictions = torch.argmax(avg_logits, dim=1)

    return predictions, avg_logits

# Run ensemble
pred, logits = ensemble_predict(input_tensor, models)
```

### Custom Loss Functions

Implement domain-specific loss functions for class imbalance.

```python
import torch
import torch.nn.functional as F

class WeightedDiceCrossEntropyLoss(torch.nn.Module):
    """Combined Dice + weighted Cross-Entropy loss."""

    def __init__(self, ce_weight=0.5, dice_weight=0.5, class_weights=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.class_weights = class_weights

    def forward(self, logits, targets):
        # Cross-entropy loss
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.class_weights,
            reduction='mean'
        )

        # Dice loss
        probs = F.softmax(logits, dim=1)
        intersection = (probs * F.one_hot(targets, num_classes=logits.shape[1])).sum()
        union = probs.sum() + F.one_hot(targets, num_classes=logits.shape[1]).sum()
        dice_loss = 1 - (2 * intersection + 1e-6) / (union + 1e-6)

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

# Use in training
class_weights = torch.tensor([1.0, 2.0, 1.5, 3.0, 2.0])  # Weights for each class
criterion = WeightedDiceCrossEntropyLoss(
    ce_weight=0.5,
    dice_weight=0.5,
    class_weights=class_weights
)
```

---

## Examples

### Complete End-to-End Pipeline

```bash
#!/bin/bash

# Directories
VIDEO_PATH="data/video.mp4"
OUTPUT_DIR="outputs"
DATASET_DIR="data/dataset"

echo "===== Road Topology Segmentation Pipeline ====="

# Step 1: Generate pseudo-labels
echo "1. Generating pseudo-labels..."
road-topo pseudolabel generate \
  --video "$VIDEO_PATH" \
  --output "$OUTPUT_DIR/pseudo_labels" \
  --config configs/detection.yaml

# Step 2: Create dataset splits
echo "2. Creating dataset splits..."
python -c "
from road_topology.segmentation.dataset import RoadTopologyDataset
from road_topology.segmentation.transforms import get_train_transforms, get_val_transforms

train_ds, val_ds, test_ds = RoadTopologyDataset.create_splits(
    root='$DATASET_DIR',
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42,
    transforms_train=get_train_transforms((512, 512)),
    transforms_val=get_val_transforms((512, 512)),
)
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')
"

# Step 3: Train model
echo "3. Training model..."
road-topo train run \
  --config configs/training.yaml \
  --data "$DATASET_DIR" \
  --output "$OUTPUT_DIR/model" \
  --epochs 100 \
  --batch-size 8

# Step 4: Validate model
echo "4. Validating model..."
road-topo train validate \
  --model "$OUTPUT_DIR/model/best_model.pth" \
  --data "$DATASET_DIR"

# Step 5: Run inference
echo "5. Running inference..."
road-topo infer image \
  --model "$OUTPUT_DIR/model/best_model.pth" \
  --input "test_image.jpg" \
  --output "$OUTPUT_DIR/result.png"

echo "===== Pipeline Complete ====="
```

### Custom Training Loop

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR

from road_topology.segmentation.models import SegFormerModel
from road_topology.segmentation.dataset import RoadTopologyDataset
from road_topology.segmentation.transforms import get_train_transforms, get_val_transforms
from road_topology.evaluation.metrics import compute_miou

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SegFormerModel(
    backbone="nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    num_classes=5
).to(device)

# Data
train_dataset = RoadTopologyDataset(
    "./data/dataset", "train",
    transforms=get_train_transforms((512, 512))
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

val_dataset = RoadTopologyDataset(
    "./data/dataset", "val",
    transforms=get_val_transforms((512, 512))
)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Optimizer & scheduler
optimizer = AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)
scheduler = PolynomialLR(optimizer, total_iters=100, power=0.9)

# Training loop
for epoch in range(100):
    # Train
    model.train()
    total_loss = 0

    for batch in train_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        # Forward
        logits = model(images)
        loss = F.cross_entropy(logits, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    # Validate
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"]

            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.append(preds)
            all_targets.append(masks.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    miou = compute_miou(preds, targets)

    print(f"Epoch {epoch}: loss={total_loss:.4f}, mIoU={miou:.4f}")

    # Save checkpoint
    if epoch % 10 == 0:
        model.save(f"outputs/checkpoint_epoch_{epoch}.pth")

    scheduler.step()
```

### Real-time Video Processing with Threading

```python
import cv2
import threading
import queue
import torch
from pathlib import Path

from road_topology.segmentation.models import SegFormerModel
from road_topology.core.types import CLASS_COLORS

class VideoProcessor:
    def __init__(self, model_path, video_path, output_path):
        self.model = SegFormerModel.load(model_path)
        self.model.eval()
        self.device = next(self.model.parameters()).device

        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)

    def read_frames(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.frame_queue.put(None)
                break
            self.frame_queue.put(frame)

    def process_frames(self):
        while True:
            frame = self.frame_queue.get()
            if frame is None:
                self.result_queue.put(None)
                break

            # Preprocess
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, (512, 512))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

            # Inference
            with torch.no_grad():
                tensor = tensor.to(self.device)
                pred = self.model.predict(tensor)[0].cpu().numpy()

            # Colorize
            pred_resized = cv2.resize(pred.astype(np.uint8), (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            for class_id, color in CLASS_COLORS.items():
                vis[pred_resized == class_id] = color

            # Overlay
            result = cv2.addWeighted(frame, 0.5, vis, 0.5, 0)
            self.result_queue.put(result)

    def write_frames(self):
        while True:
            result = self.result_queue.get()
            if result is None:
                break
            self.writer.write(result)

    def run(self):
        t_read = threading.Thread(target=self.read_frames)
        t_process = threading.Thread(target=self.process_frames)
        t_write = threading.Thread(target=self.write_frames)

        t_read.start()
        t_process.start()
        t_write.start()

        t_read.join()
        t_process.join()
        t_write.join()

        self.cap.release()
        self.writer.release()

# Usage
processor = VideoProcessor(
    "model.pth",
    "input.mp4",
    "output.mp4"
)
processor.run()
```

---

## Getting Help

For issues, questions, or contributions:

- Check the [Troubleshooting](#troubleshooting) section
- Review example configurations in `configs/`
- Inspect example scripts in `scripts/`
- Check test cases in `tests/`
- Review inline code documentation with `help()` in Python

## Citation

If you use this project in research, please cite:

```bibtex
@software{road_topology_2024,
  title={Road Topology Segmentation: Vehicle Trajectory-based Pseudo-Label Generation with Zero-shot SAM},
  year={2024}
}
```

---

**Version:** 0.1.0
**Last Updated:** January 2026
**License:** MIT
