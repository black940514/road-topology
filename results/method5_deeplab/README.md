# DeepLabV3 Semantic Segmentation Results

## Method Overview
Deep learning-based semantic segmentation using PyTorch DeepLabV3-ResNet101 pretrained on COCO dataset.

## Model Details
- **Architecture**: DeepLabV3 with ResNet101 backbone
- **Pretrained on**: COCO with VOC labels
- **Input size**: 520x520 (resized)
- **Classes detected**: 21 semantic categories (PASCAL VOC)
- **Device**: CPU (due to PyTorch 1.8.0 CUDA incompatibility with RTX 4070 Ti)

## Target Classes for Road Topology
- **Class 2**: Bicycle
- **Class 6**: Bus
- **Class 7**: Car
- **Class 14**: Motorbike

These classes are combined to create binary masks representing road-related objects.

## Outputs Generated

For each input image, three files are created:

1. **`*_deeplab.png`**: Blended visualization
   - Original image overlaid with colored segmentation map
   - Blend ratio: 60% original / 40% segmentation
   - All 21 classes visualized with distinct colors

2. **`*_mask.png`**: Binary mask (vehicles only)
   - White pixels: Vehicle-related classes (bicycle, bus, car, motorbike)
   - Black pixels: Background
   - Used for extracting road topology features

3. **`*_pred.npy`**: Raw prediction map
   - NumPy array of class IDs per pixel
   - Can be used for further analysis or custom visualization

## Processing Statistics

- **Total images available**: 90
- **Sample processed**: 10 images
- **Success rate**: 100% (10/10)
- **Average processing time**: ~1.83 seconds per image (CPU)
- **Total output size**: ~27 MB

## Sample Results

```
01CT000000123_deeplab.png  (472 KB)
01CT000000134_deeplab.png  (727 KB)
01CT000000137_deeplab.png  (594 KB)
01CT000000161_deeplab.png  (655 KB)
01CT000000164_deeplab.png  (770 KB)
01CT000000166_deeplab.png  (545 KB)
01CT000000173_deeplab.png  (547 KB)
01CT000000178_deeplab.png  (702 KB)
01CT000000180_deeplab.png  (604 KB)
01CT000000192_deeplab.png  (547 KB)
```

## Color Palette (PASCAL VOC)

| Class ID | Name | Color RGB |
|----------|------|-----------|
| 0 | Background | (0, 0, 0) |
| 2 | Bicycle | (128, 128, 0) |
| 6 | Bus | (0, 128, 128) |
| 7 | Car | (128, 128, 128) |
| 14 | Motorbike | (64, 128, 128) |
| 15 | Person | (192, 128, 128) |

## Technical Notes

### Performance Considerations
- CPU inference: ~1.8s per image
- GPU would be 10-100x faster but requires PyTorch upgrade
- Current PyTorch 1.8.0 doesn't support RTX 4070 Ti (sm_89)

### Limitations
1. **Pretrained on COCO/VOC**: Not specifically trained on road scenes
2. **No explicit road surface class**: Model detects vehicles but not road surface itself
3. **Urban scenes**: Performance may vary on rural or highway scenes
4. **Resolution**: Input resized to 520x520 may lose fine details

### Advantages
1. **State-of-the-art**: DeepLabV3 is industry standard for semantic segmentation
2. **Multi-class**: Detects 21 object categories simultaneously
3. **Pretrained**: No training required, immediate deployment
4. **Robust**: Handles various lighting and weather conditions

## Usage

Run the segmentation script:

```bash
python3 segment_deeplab.py
```

The script automatically:
1. Downloads pretrained weights (233 MB, one-time)
2. Loads images from `output_dir3/`
3. Processes first 10 images (configurable)
4. Saves results to `results/method5_deeplab/`

## Comparison with Other Methods

| Method | Approach | Speed | Accuracy |
|--------|----------|-------|----------|
| Method 5 (DeepLabV3) | Deep learning | Slow (CPU) | High |
| Method 1 (Perspective) | Geometric | Fast | Medium |
| Method 2 (EdgeDetection) | Classical CV | Fast | Low |
| Method 3 (HSV) | Color-based | Fast | Low |
| Method 4 (Threshold) | Simple threshold | Very fast | Low |

## Future Improvements

1. **GPU acceleration**: Upgrade PyTorch to support RTX 4070 Ti
2. **Fine-tuning**: Train on road-specific dataset (Cityscapes, BDD100K)
3. **Road surface detection**: Add explicit road surface segmentation
4. **Post-processing**: Morphological operations to clean masks
5. **Batch processing**: Process all 90 images instead of sample 10

## References

- [DeepLabV3 Paper](https://arxiv.org/abs/1706.05587)
- [PyTorch Segmentation Models](https://pytorch.org/vision/stable/models.html#semantic-segmentation)
- [PASCAL VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
- [COCO Dataset](https://cocodataset.org/)
