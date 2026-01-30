# Quick Start Guide: Dataset Download & Preparation

Get started with road topology datasets in 3 steps.

## Step 1: Install Dependencies

```bash
# Activate virtual environment
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

**Required packages:**
- typer, rich, requests (for download script)
- torch, opencv-python, matplotlib (for data processing)

## Step 2: Download a Dataset

### Option A: Quick Test (Recommended for first-time users)

```bash
# Download KITTI Road (small, 289 images, no auth required)
python scripts/download_datasets.py kitti

# Output: data/kitti/train/ and data/kitti/test/
```

### Option B: View Available Datasets

```bash
# Show information about all datasets
python scripts/download_datasets.py info
```

### Option C: Demo Mode

```bash
# Download samples and visualize
python scripts/download_datasets.py demo --samples 10

# Creates:
# - data/demo/kitti/
# - visualization.png (images + masks + overlays)
# - Dataset statistics
```

## Step 3: Verify & Test

### Test Dataset Loading

```bash
# Test loading KITTI dataset
python scripts/test_dataset_loading.py test-dataset kitti

# Output:
# - Batch statistics
# - Class distribution
# - Visualization (dataset_samples.png)
```

### Test All Available Datasets

```bash
# Test all downloaded datasets
python scripts/test_dataset_loading.py test-all

# Combines all available datasets
# Shows total image count
```

### Benchmark Loading Speed

```bash
# Benchmark KITTI dataset loading
python scripts/test_dataset_loading.py benchmark kitti --batch-size 16 --num-workers 4

# Shows:
# - Images per second
# - Time per batch
# - Optimal worker count
```

---

## Full Dataset Downloads

For production use, download larger datasets:

### BDD100K (PRIMARY - Best for Crosswalks)

**Features:**
- 10K semantic segmentation images
- 100K lane marking images
- **Only dataset with crosswalk annotations**
- 9 lane marking categories

**Steps:**
1. Install Kaggle CLI:
   ```bash
   pip install kaggle
   ```

2. Setup Kaggle credentials:
   ```bash
   mkdir -p ~/.kaggle
   # Download kaggle.json from kaggle.com/account
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. Download datasets:
   ```bash
   # Semantic segmentation
   kaggle datasets download -d solesensei/bdd100k_sem_seg -p data/bdd100k_raw/

   # Lane markings (IMPORTANT for crosswalks!)
   kaggle datasets download -d solesensei/bdd100k-lane -p data/bdd100k_raw/
   ```

4. Extract and process:
   ```bash
   cd data/bdd100k_raw
   unzip bdd100k_sem_seg.zip
   unzip bdd100k-lane.zip
   cd ../..

   # Run processing script (TODO: implement BDD100K processing)
   python scripts/download_datasets.py bdd100k
   ```

### Cityscapes (SECONDARY - Best for Urban Scenes)

**Features:**
- 5K training images
- 500 validation images
- Very high quality manual annotations
- 34 semantic classes

**Steps:**
1. Register at: https://www.cityscapes-dataset.com/register/

2. Download (requires login):
   - leftImg8bit_trainvaltest.zip
   - gtFine_trainvaltest.zip

3. Extract:
   ```bash
   mkdir -p data/cityscapes
   unzip leftImg8bit_trainvaltest.zip -d data/cityscapes/
   unzip gtFine_trainvaltest.zip -d data/cityscapes/
   ```

4. Process:
   ```bash
   python scripts/download_datasets.py cityscapes
   ```

---

## Dataset Structure

After download and processing, datasets follow this structure:

```
data/
├── kitti/
│   ├── train/
│   │   ├── images/
│   │   │   ├── um_000000.png
│   │   │   └── ...
│   │   └── masks/
│   │       ├── um_000000.png
│   │       └── ...
│   └── test/
│       ├── images/
│       └── masks/
├── bdd100k/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── val/
│       ├── images/
│       └── masks/
└── cityscapes/
    ├── train/
    ├── val/
    └── test/
```

**Mask Format:**
- Grayscale PNG images
- Pixel values = class IDs (0-4)
- Classes: Background(0), Road(1), Lane(2), Crosswalk(3), Sidewalk(4)

---

## Using Datasets in Code

### Basic Usage

```python
from pathlib import Path
from road_topology.segmentation.dataset import RoadTopologyDataset
from torch.utils.data import DataLoader

# Load dataset
dataset = RoadTopologyDataset(
    root=Path("data/kitti"),
    split="train"
)

print(f"Loaded {len(dataset)} images")

# Create DataLoader
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4
)

# Iterate
for batch in loader:
    images = batch["image"]  # (B, 3, H, W)
    masks = batch["mask"]    # (B, H, W)
    paths = batch["image_path"]

    # Your training code here
    break
```

### Combine Multiple Datasets

```python
from torch.utils.data import ConcatDataset

# Load all datasets
kitti = RoadTopologyDataset("data/kitti", "train")
bdd = RoadTopologyDataset("data/bdd100k", "train")
cityscapes = RoadTopologyDataset("data/cityscapes", "train")

# Combine
combined = ConcatDataset([kitti, bdd, cityscapes])

print(f"Combined: {len(combined)} images")
# Combined: 10000+ images
```

### Weighted Sampling

```python
from torch.utils.data import WeightedRandomSampler

# Dataset weights (BDD100K has crosswalks, prioritize it)
weights = {
    "bdd100k": 3.0,      # Has crosswalk annotations
    "cityscapes": 2.0,   # High quality
    "kitti": 1.0         # Small, binary only
}

# Create sample weights
sample_weights = []
for dataset_name, dataset in [("bdd100k", bdd), ("cityscapes", cityscapes), ("kitti", kitti)]:
    sample_weights.extend([weights[dataset_name]] * len(dataset))

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

loader = DataLoader(combined, batch_size=8, sampler=sampler)
```

### Custom Transforms

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define transforms
train_transform = A.Compose([
    A.RandomResizedCrop(height=512, width=512, scale=(0.5, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

dataset = RoadTopologyDataset(
    root="data/kitti",
    split="train",
    transforms=train_transform
)
```

---

## Class Distribution

Expected class distribution in urban driving datasets:

| Class | Typical % | Notes |
|-------|-----------|-------|
| Background | 40-60% | Sky, buildings, vehicles |
| Road | 30-40% | Dominant class |
| Lane | 1-3% | **High class imbalance!** |
| Crosswalk | 0.5-2% | **Very rare!** |
| Sidewalk | 5-15% | Urban scenes |

**Important:**
- **Severe class imbalance** for Lane and Crosswalk
- Use **weighted loss** or **focal loss**
- Don't rely on accuracy alone - use **per-class IoU**

---

## Troubleshooting

### "No images found"

```bash
# Check if dataset exists
ls data/kitti/train/images/

# If empty, re-download
python scripts/download_datasets.py kitti --no-skip-existing
```

### "Failed to load image"

Check file permissions:
```bash
chmod -R u+r data/
```

### Slow DataLoader

Increase workers:
```python
loader = DataLoader(dataset, num_workers=8)  # More workers
```

Benchmark optimal worker count:
```bash
python scripts/test_dataset_loading.py benchmark kitti --num-workers 8
```

### Out of Memory

Reduce batch size:
```python
loader = DataLoader(dataset, batch_size=4)  # Smaller batches
```

### Import Errors

```bash
# Install missing packages
pip install torch torchvision opencv-python
pip install albumentations matplotlib
```

---

## Next Steps

After setting up datasets:

1. **Explore data:**
   ```bash
   python scripts/test_dataset_loading.py test-dataset kitti --visualize
   ```

2. **Train baseline model:**
   ```bash
   # TODO: Add training script reference
   python train.py --dataset kitti --epochs 10
   ```

3. **Evaluate:**
   ```bash
   # TODO: Add evaluation script reference
   python evaluate.py --dataset kitti --checkpoint best.pth
   ```

4. **Visualize predictions:**
   ```bash
   # TODO: Add visualization script reference
   python visualize.py --input test_image.jpg --checkpoint best.pth
   ```

---

## Resources

- **Download script:** `scripts/download_datasets.py`
- **Test script:** `scripts/test_dataset_loading.py`
- **Class mappings:** `scripts/CLASS_MAPPINGS.md`
- **Dataset docs:** `scripts/README.md`

## Questions?

- Check `scripts/README.md` for detailed documentation
- Run `python scripts/download_datasets.py --help` for all options
- See `scripts/CLASS_MAPPINGS.md` for class mapping details
