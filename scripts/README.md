# Road Topology Dataset Scripts

This directory contains scripts for downloading and preparing public datasets for road topology segmentation.

## Available Scripts

### download_datasets.py

Download and prepare public datasets with automatic class remapping to project classes.

**Supported Datasets:**
- **BDD100K** (PRIMARY) - Berkeley DeepDrive dataset with 9 lane categories and semantic segmentation
- **Cityscapes** (SECONDARY) - High-quality urban street scenes
- **KITTI Road** (SMALL) - 289 images, good for quick testing

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- typer >= 0.9.0 (CLI framework)
- rich >= 13.0.0 (Terminal UI)
- requests >= 2.31.0 (Downloads)
- matplotlib >= 3.7.0 (Visualization)

## Usage

### Quick Start

```bash
# Show available datasets
python scripts/download_datasets.py info

# Download demo samples and visualize
python scripts/download_datasets.py demo

# Download KITTI (smallest, no auth required)
python scripts/download_datasets.py kitti

# Download all datasets
python scripts/download_datasets.py all-datasets
```

### Individual Datasets

#### KITTI Road (Recommended for Testing)

```bash
# Download KITTI Road dataset (289 images)
python scripts/download_datasets.py kitti --output-dir data/

# Output structure:
# data/kitti/
#   ├── train/images/
#   ├── train/masks/
#   └── test/images/
```

#### BDD100K

BDD100K requires Kaggle authentication or manual download:

```bash
# Show download instructions
python scripts/download_datasets.py bdd100k

# Manual steps:
# 1. Install kaggle CLI: pip install kaggle
# 2. Setup credentials: ~/.kaggle/kaggle.json
# 3. Download:
#    kaggle datasets download -d solesensei/bdd100k_sem_seg
#    kaggle datasets download -d solesensei/bdd100k-lane
# 4. Extract to data/bdd100k/
```

#### Cityscapes

Cityscapes requires account registration:

```bash
# Show download instructions
python scripts/download_datasets.py cityscapes

# Manual steps:
# 1. Register at: https://www.cityscapes-dataset.com/register/
# 2. Download:
#    - leftImg8bit_trainvaltest.zip
#    - gtFine_trainvaltest.zip
# 3. Extract to data/cityscapes/
# 4. Re-run script to process masks
```

### Demo Mode

Download samples and visualize:

```bash
python scripts/download_datasets.py demo --samples 10

# Creates:
# - data/demo/kitti/ with sample images
# - Visualization PNG showing images, masks, and overlays
# - Dataset statistics table
```

## Class Mapping

The script automatically remaps dataset classes to project classes:

### Project Classes (5 classes)
```
0 - Background
1 - Road
2 - Lane
3 - Crosswalk
4 - Sidewalk
```

### BDD100K Mapping

**Semantic Segmentation (19 → 5 classes):**
- Class 0 (road) → Road (1)
- Class 1 (sidewalk) → Sidewalk (4)
- Classes 6-7 (traffic light/sign) → Lane (2)
- Others → Background (0)

**Lane Markings (9 → 5 classes):**
- Class 0 (crosswalk) → Crosswalk (3)
- Classes 1-6 (lane lines) → Lane (2)
- Class 7 (road curb) → Road (1)
- Others → Background (0)

### Cityscapes Mapping

**34 → 5 classes:**
- Class 7 (road) → Road (1)
- Class 8 (sidewalk) → Sidewalk (4)
- Classes 19-20 (traffic signs) → Lane (2)
- Others → Background (0)

### KITTI Road Mapping

**Binary → 5 classes:**
- Road pixels → Road (1)
- Non-road → Background (0)

## Output Structure

All datasets are organized in the same format:

```
data/
├── bdd100k/
│   ├── train/
│   │   ├── images/
│   │   │   ├── img_001.jpg
│   │   │   └── ...
│   │   └── masks/
│   │       ├── img_001.png
│   │       └── ...
│   └── val/
│       ├── images/
│       └── masks/
├── cityscapes/
│   ├── train/
│   ├── val/
│   └── test/
└── kitti/
    ├── train/
    └── test/
```

This structure is compatible with `RoadTopologyDataset`:

```python
from road_topology.segmentation.dataset import RoadTopologyDataset

dataset = RoadTopologyDataset(root="data/kitti", split="train")
```

## Advanced Options

### Skip Existing Downloads

```bash
# Re-download even if exists
python scripts/download_datasets.py kitti --no-skip-existing
```

### Custom Output Directory

```bash
python scripts/download_datasets.py kitti --output-dir /path/to/data
```

### Demo with More Samples

```bash
python scripts/download_datasets.py demo --samples 20
```

## Dataset Comparison

| Dataset | Size | Download | Classes | Best For |
|---------|------|----------|---------|----------|
| **KITTI Road** | 289 images | Direct | 2 (road/non-road) | Quick testing, prototyping |
| **BDD100K** | 10K+100K | Kaggle auth | 19 semantic + 9 lane | Production, crosswalk detection |
| **Cityscapes** | 5K images | Account required | 34 semantic | High-quality urban scenes |

## Troubleshooting

### Kaggle Authentication

```bash
# Install kaggle CLI
pip install kaggle

# Setup credentials (from kaggle.com/account)
mkdir -p ~/.kaggle
# Download kaggle.json from your Kaggle account
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Download Errors

If automatic download fails:
1. Check internet connection
2. Try manual download (URLs shown in error message)
3. Extract manually to correct directory
4. Re-run script to process masks

### Missing Dependencies

```bash
pip install typer rich requests matplotlib
```

## Next Steps

After downloading datasets:

1. **Verify data:**
   ```bash
   python scripts/download_datasets.py demo
   ```

2. **Train a model:**
   ```python
   from road_topology.segmentation.dataset import RoadTopologyDataset
   from torch.utils.data import DataLoader

   dataset = RoadTopologyDataset(root="data/kitti", split="train")
   loader = DataLoader(dataset, batch_size=4, shuffle=True)
   ```

3. **Combine datasets:**
   ```python
   from torch.utils.data import ConcatDataset

   kitti_train = RoadTopologyDataset("data/kitti", "train")
   bdd_train = RoadTopologyDataset("data/bdd100k", "train")
   combined = ConcatDataset([kitti_train, bdd_train])
   ```

## References

- **BDD100K:** https://www.bdd100k.com/
- **Cityscapes:** https://www.cityscapes-dataset.com/
- **KITTI:** http://www.cvlibs.net/datasets/kitti/

## Support

For issues or questions:
1. Check dataset documentation links above
2. Verify all dependencies are installed
3. Try demo mode first to test setup
