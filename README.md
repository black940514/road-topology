# Road Topology Segmentation

SegFormer-B5 기반 도로 세그멘테이션 시스템으로, 차선(lane) 인스턴스 분리와 횡단보도(crosswalk) 검출을 지원합니다.

## Features

- **Dual-Head Architecture**: Semantic segmentation (6 classes) + Instance embedding (32-dim)
- **SegFormer-B5 Backbone**: 84.7M params, Cityscapes pretrained
- **Lane Instance Separation**: Discriminative loss + MeanShift/DBSCAN clustering
- **Left-to-Right Lane Ordering**: 차선을 좌측부터 우측 순서로 정렬

## Semantic Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | background | 배경 |
| 1 | road | 도로 영역 |
| 2 | lane_marking | 차선 표시 |
| 3 | crosswalk | 횡단보도 |
| 4 | sidewalk | 인도 |
| 5 | lane_boundary | 차선 경계 |

## Installation

```bash
# Clone repository
git clone https://github.com/black940514/road-topology.git
cd road-topology

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

## Dataset Preparation

### TuSimple Lane Dataset

```bash
# Download and convert TuSimple dataset
python scripts/download_tusimple.py --output data/tusimple

# Output structure:
# data/tusimple/
# ├── train/
# │   ├── images/
# │   ├── semantic_masks/
# │   └── instance_masks/
# └── val/
#     ├── images/
#     ├── semantic_masks/
#     └── instance_masks/
```

### Custom Dataset

데이터셋 구조:
```
data/your_dataset/
├── train/
│   ├── images/          # RGB images (*.png, *.jpg)
│   ├── semantic_masks/  # Semantic segmentation masks
│   └── instance_masks/  # Lane instance masks (optional)
└── val/
    ├── images/
    ├── semantic_masks/
    └── instance_masks/
```

## Usage

### Generate Instance Masks

기존 semantic mask에서 lane instance mask 생성:

```bash
road-topology train generate-instances \
    --semantic-dir data/your_dataset/train/semantic_masks \
    --output-dir data/your_dataset/train/instance_masks \
    --min-size 100
```

### Training

```bash
# Lane segmentation training
road-topology train lane \
    --config configs/lane_config.yaml \
    --data data/tusimple \
    --output results/lane_model \
    --epochs 50
```

### Inference

```bash
# Single image inference
road-topology infer lane \
    --model results/lane_model/best_model.pth \
    --input test_image.jpg \
    --output result.png \
    --visualize

# Batch inference
road-topology infer lane \
    --model results/lane_model/best_model.pth \
    --input input_dir/ \
    --output output_dir/
```

## Architecture

```
Input Image (H, W, 3)
        │
        ▼
┌───────────────────────┐
│   SegFormer-B5        │
│   (Shared Encoder)    │
└───────────────────────┘
        │
        ├──────────────────────┐
        ▼                      ▼
┌───────────────┐      ┌───────────────┐
│ Semantic Head │      │ Embedding Head│
│ (6 classes)   │      │ (32-dim)      │
└───────────────┘      └───────────────┘
        │                      │
        ▼                      ▼
  Semantic Logits        Embeddings
   (H/4, W/4, 6)        (H/4, W/4, 32)
        │                      │
        └──────────┬───────────┘
                   ▼
           ┌──────────────┐
           │ PostProcessor│
           │ (Clustering) │
           └──────────────┘
                   │
                   ▼
        Lane Instances (1, 2, 3, ...)
```

## Loss Functions

### Semantic Loss
- CrossEntropy + Dice Loss

### Discriminative Loss (Instance)
Based on [arXiv:1708.02551](https://arxiv.org/abs/1708.02551):
- **Variance term**: Pull embeddings toward instance mean
- **Distance term**: Push instance means apart
- **Regularization term**: Keep means close to origin

## Project Structure

```
road_topology/
├── src/road_topology/
│   ├── core/
│   │   ├── config.py          # Configuration classes
│   │   └── types.py           # Type definitions, constants
│   ├── segmentation/
│   │   ├── models/
│   │   │   ├── segformer.py       # Base SegFormer
│   │   │   └── segformer_lane.py  # Dual-head SegFormer
│   │   ├── dataset.py         # Dataset classes
│   │   ├── losses.py          # Loss functions
│   │   ├── instance_generator.py  # Instance mask generation
│   │   ├── postprocess.py     # Lane clustering
│   │   └── trainer_lane.py    # Training loop
│   ├── inference/
│   │   ├── predictor.py       # Base predictor
│   │   └── predictor_lane.py  # Lane predictor
│   ├── evaluation/
│   │   ├── metrics.py         # Evaluation metrics
│   │   └── visualize.py       # Visualization utilities
│   └── cli/
│       ├── train.py           # Training commands
│       └── infer.py           # Inference commands
├── scripts/
│   └── download_tusimple.py   # Dataset download script
├── configs/                   # Configuration files
├── data/                      # Datasets (gitignored)
└── results/                   # Training outputs (gitignored)
```

## Configuration

`LaneSegmentationConfig` 주요 파라미터:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backbone` | `"b5"` | SegFormer variant (b0-b5) |
| `num_semantic_classes` | `6` | Number of semantic classes |
| `embedding_dim` | `32` | Instance embedding dimension |
| `semantic_weight` | `1.0` | Semantic loss weight |
| `instance_weight` | `0.5` | Instance loss weight |
| `delta_var` | `0.5` | Variance loss margin |
| `delta_dist` | `1.5` | Distance loss margin |

## References

- [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)
- [Semantic Instance Segmentation with a Discriminative Loss Function](https://arxiv.org/abs/1708.02551)
- [LaneNet: Real-Time Lane Detection Networks for Autonomous Driving](https://arxiv.org/abs/1807.01726)

## License

MIT License
