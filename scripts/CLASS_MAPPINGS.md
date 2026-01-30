# Dataset Class Mappings

This document provides detailed class mapping information for all supported datasets.

## Project Classes (Target)

The road topology segmentation project uses 5 classes:

| ID | Name | Color (RGB) | Description |
|----|------|-------------|-------------|
| 0 | Background | (0, 0, 0) Black | Everything else |
| 1 | Road | (128, 128, 128) Gray | Drivable road surface |
| 2 | Lane | (255, 255, 255) White | Lane markings, road signs |
| 3 | Crosswalk | (255, 255, 0) Yellow | Pedestrian crosswalks |
| 4 | Sidewalk | (0, 255, 0) Green | Sidewalks, pedestrian areas |

---

## BDD100K Dataset

BDD100K provides two types of annotations:

### 1. Semantic Segmentation (19 classes)

Original classes → Project mapping:

| Original ID | Original Class | → | Project ID | Project Class |
|-------------|----------------|---|------------|---------------|
| 0 | road | → | 1 | Road |
| 1 | sidewalk | → | 4 | Sidewalk |
| 2 | building | → | 0 | Background |
| 3 | wall | → | 0 | Background |
| 4 | fence | → | 0 | Background |
| 5 | pole | → | 0 | Background |
| 6 | traffic light | → | 2 | Lane |
| 7 | traffic sign | → | 2 | Lane |
| 8 | vegetation | → | 0 | Background |
| 9 | terrain | → | 0 | Background |
| 10 | sky | → | 0 | Background |
| 11 | person | → | 0 | Background |
| 12 | rider | → | 0 | Background |
| 13 | car | → | 0 | Background |
| 14 | truck | → | 0 | Background |
| 15 | bus | → | 0 | Background |
| 16 | train | → | 0 | Background |
| 17 | motorcycle | → | 0 | Background |
| 18 | bicycle | → | 0 | Background |
| 255 | unlabeled | → | 0 | Background |

**Notes:**
- Traffic signs/lights mapped to Lane class as they indicate lane boundaries
- All vehicles and objects mapped to Background
- Semantic segmentation provides general road structure

### 2. Lane Markings (9 categories)

Original classes → Project mapping:

| Original ID | Original Class | → | Project ID | Project Class |
|-------------|----------------|---|------------|---------------|
| 0 | crosswalk | → | 3 | Crosswalk |
| 1 | double white | → | 2 | Lane |
| 2 | double yellow | → | 2 | Lane |
| 3 | double other | → | 2 | Lane |
| 4 | single white | → | 2 | Lane |
| 5 | single yellow | → | 2 | Lane |
| 6 | single other | → | 2 | Lane |
| 7 | road curb | → | 1 | Road |
| 8 | other | → | 0 | Background |
| 255 | unlabeled | → | 0 | Background |

**Notes:**
- **Crosswalk class is ONLY in BDD100K lane markings!**
- All lane line types (single/double, white/yellow) mapped to Lane class
- Road curbs define road boundaries
- Lane marking annotations are more detailed for topology

**Recommended Usage:**
For best results, **combine both annotations**:
1. Use semantic segmentation for Road and Sidewalk
2. Use lane markings for Lane and Crosswalk (more accurate)
3. Merge by priority: Lane markings > Semantic segmentation

---

## Cityscapes Dataset

Cityscapes provides 34 fine-grained classes.

Original classes → Project mapping:

| Original ID | Original Class | → | Project ID | Project Class |
|-------------|----------------|---|------------|---------------|
| 0 | unlabeled | → | 0 | Background |
| 1 | ego vehicle | → | 0 | Background |
| 2 | rectification border | → | 0 | Background |
| 3 | out of roi | → | 0 | Background |
| 4 | static | → | 0 | Background |
| 5 | dynamic | → | 0 | Background |
| 6 | ground | → | 0 | Background |
| 7 | **road** | → | **1** | **Road** |
| 8 | **sidewalk** | → | **4** | **Sidewalk** |
| 9 | parking | → | 0 | Background |
| 10 | rail track | → | 0 | Background |
| 11 | building | → | 0 | Background |
| 12 | wall | → | 0 | Background |
| 13 | fence | → | 0 | Background |
| 14 | guard rail | → | 0 | Background |
| 15 | bridge | → | 0 | Background |
| 16 | tunnel | → | 0 | Background |
| 17 | pole | → | 0 | Background |
| 18 | polegroup | → | 0 | Background |
| 19 | **traffic light** | → | **2** | **Lane** |
| 20 | **traffic sign** | → | **2** | **Lane** |
| 21 | vegetation | → | 0 | Background |
| 22 | terrain | → | 0 | Background |
| 23 | sky | → | 0 | Background |
| 24 | person | → | 0 | Background |
| 25 | rider | → | 0 | Background |
| 26 | car | → | 0 | Background |
| 27 | truck | → | 0 | Background |
| 28 | bus | → | 0 | Background |
| 29 | caravan | → | 0 | Background |
| 30 | trailer | → | 0 | Background |
| 31 | train | → | 0 | Background |
| 32 | motorcycle | → | 0 | Background |
| 33 | bicycle | → | 0 | Background |
| 255 | license plate | → | 0 | Background |

**Notes:**
- Very high quality annotations (manually verified)
- No explicit crosswalk class
- Good for Road and Sidewalk boundaries
- Traffic signs/lights used as lane markers
- 5K training images, 500 validation images

**Limitations:**
- No crosswalk annotations
- Lane markings not as detailed as BDD100K
- Smaller dataset size

---

## KITTI Road Dataset

KITTI Road is a binary segmentation dataset.

Original classes → Project mapping:

| Original ID | Original Class | → | Project ID | Project Class |
|-------------|----------------|---|------------|---------------|
| 0 | non-road | → | 0 | Background |
| 1 | road | → | 1 | Road |
| 2 | road (alternative) | → | 1 | Road |
| 255 | unlabeled | → | 0 | Background |

**Notes:**
- Only 289 images (training + testing)
- Binary classification: road vs. non-road
- Good for quick prototyping and testing
- No lane markings, crosswalks, or sidewalks
- Uses color-coded masks (road pixels in red/pink)

**Limitations:**
- Very small dataset
- Binary only (no detailed topology)
- Limited diversity of scenes

**Recommended Usage:**
- Quick testing of data pipeline
- Prototyping models
- Baseline comparisons

---

## Class Distribution Expectations

Based on typical urban driving scenes:

| Class | Typical % | Notes |
|-------|-----------|-------|
| **Background** | 40-60% | Sky, buildings, vehicles, vegetation |
| **Road** | 30-40% | Dominant class in driving scenes |
| **Lane** | 1-3% | Small but important (high-class imbalance) |
| **Crosswalk** | 0.5-2% | Rare but critical for safety |
| **Sidewalk** | 5-15% | Present in urban scenes |

**Implications:**
- **Severe class imbalance:** Lane and Crosswalk are minority classes
- **Need weighted loss functions** or focal loss
- **Data augmentation** critical for minority classes
- **Metric selection:** Don't rely on accuracy alone, use IoU per class

---

## Dataset Combination Strategy

When combining multiple datasets:

### Priority Order (for conflicting pixels)

1. **BDD100K Lane Markings** (highest priority)
   - Most detailed lane and crosswalk annotations
   - Explicit crosswalk class

2. **Cityscapes Fine**
   - High quality road/sidewalk boundaries
   - Manually verified

3. **BDD100K Semantic**
   - Good general structure
   - Larger quantity

4. **KITTI Road** (lowest priority)
   - Binary only
   - Use for additional road samples

### Recommended Merging

```python
def merge_annotations(sem_mask, lane_mask):
    """Merge semantic and lane marking masks.

    Args:
        sem_mask: Semantic segmentation (road, sidewalk, background)
        lane_mask: Lane markings (lanes, crosswalks)

    Returns:
        Merged mask with priority to lane markings
    """
    merged = sem_mask.copy()

    # Lane markings override semantic (more detailed)
    lane_regions = (lane_mask == 2)  # Lane class
    merged[lane_regions] = 2

    # Crosswalks override everything (highest priority)
    crosswalk_regions = (lane_mask == 3)
    merged[crosswalk_regions] = 3

    return merged
```

### Dataset Weights for Training

Suggested sampling weights:

| Dataset | Weight | Reason |
|---------|--------|--------|
| BDD100K Lane | 3.0 | Has crosswalk class |
| BDD100K Semantic | 2.0 | Large, diverse |
| Cityscapes | 2.0 | High quality |
| KITTI Road | 1.0 | Small, binary only |

---

## Validation Strategy

When evaluating models:

1. **Primary Metrics** (per-class IoU):
   - Road IoU
   - Lane IoU
   - Crosswalk IoU (critical for safety)
   - Sidewalk IoU

2. **Dataset-Specific Evaluation**:
   - BDD100K: Focus on crosswalk detection (only dataset with ground truth)
   - Cityscapes: Focus on road/sidewalk boundary accuracy
   - KITTI: Basic road segmentation baseline

3. **Cross-Dataset Evaluation**:
   - Train on combined dataset
   - Test on each dataset separately
   - Check for dataset bias

---

## References

- **BDD100K:** Yu, F., et al. "BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning." CVPR 2020.
  - https://www.bdd100k.com/
  - https://arxiv.org/abs/1805.04687

- **Cityscapes:** Cordts, M., et al. "The Cityscapes Dataset for Semantic Urban Scene Understanding." CVPR 2016.
  - https://www.cityscapes-dataset.com/
  - https://arxiv.org/abs/1604.01685

- **KITTI Road:** Fritsch, J., et al. "A New Performance Measure and Evaluation Benchmark for Road Detection Algorithms." ITSC 2013.
  - http://www.cvlibs.net/datasets/kitti/eval_road.php

---

## Quick Reference

### BDD100K Lane (BEST for crosswalks)
```python
# Class 0 → Crosswalk (3)
# Classes 1-6 → Lane (2)
# Class 7 → Road (1)
```

### Cityscapes (BEST for road/sidewalk)
```python
# Class 7 → Road (1)
# Class 8 → Sidewalk (4)
# Classes 19-20 → Lane (2)
```

### KITTI (SIMPLEST)
```python
# Classes 1-2 → Road (1)
# Everything else → Background (0)
```
