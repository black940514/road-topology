# Method 1: Color-Based Road Segmentation Results

## Overview
Color-based road segmentation using HSV color space thresholding to identify road regions in CCTV images.

## Method Details

### Algorithm
1. **Color Space Conversion**: Convert BGR to HSV color space
2. **Threshold Definition**:
   - Hue (H): 0-180 (all hues for gray colors)
   - Saturation (S): 0-50 (low saturation for gray)
   - Value (V): 40-180 (medium to high brightness)
3. **Morphological Operations**:
   - Closing: Fill small holes in detected regions
   - Opening: Remove small noise artifacts
4. **Visualization**: Create three outputs per image
   - Blended overlay (60% original + 40% gray mask)
   - Binary mask (white=road, black=non-road)
   - Isolated road (original colors on detected regions)

### Parameters
```python
lower_road = np.array([0, 0, 40])
upper_road = np.array([180, 50, 180])
kernel_size = 5x5
```

## Results

### Processing Summary
- **Total images available**: 90 CCTV images
- **Sample processed**: 10 images
- **Success rate**: 100% (10/10)
- **Processing speed**: ~42 images/second

### Output Files
For each input image, three outputs are generated:

1. **`{name}_color.png`**: Blended visualization (original image + gray overlay on roads)
2. **`{name}_mask.png`**: Binary mask (white=road, black=background)
3. **`{name}_isolated.png`**: Isolated road regions with original colors

Total files generated: 31 files (10 images × 3 outputs + 1 script)

### Sample Results

#### Processed Images
1. 01CT000000164 (785K color, 752K isolated, 12K mask)
2. 01CT000000196 (621K color, 379K isolated, 12K mask)
3. 01CT000000220 (621K color, 557K isolated, 8.2K mask)
4. 02CT000000181 (747K color, 531K isolated, 13K mask)
5. 03CT000000034 (539K color, 583K isolated, 4.7K mask)
6. 03CT000000085 (631K color, 579K isolated, 9.0K mask)
7. 04CT000001324 (166K color, sizes vary)
8-10. Additional samples

## Advantages
- **Fast**: ~42 images/second processing speed
- **Simple**: No ML model or training required
- **Interpretable**: Clear color threshold parameters
- **Lightweight**: Low computational requirements

## Limitations
- **Color dependency**: Assumes roads are gray/low-saturation
- **Lighting sensitivity**: May fail in extreme lighting conditions
- **False positives**: May detect other gray objects (buildings, shadows)
- **Weather sensitivity**: Performance may degrade in rain/snow

## File Structure
```
results/method1_color/
├── segment_color.py           # Main segmentation script
├── REPORT.md                  # This report
├── 01CT000000164_color.png    # Blended result
├── 01CT000000164_mask.png     # Binary mask
├── 01CT000000164_isolated.png # Isolated road
└── ... (27 more output files)
```

## Usage
```bash
# Activate virtual environment
source .venv/bin/activate

# Run segmentation
python results/method1_color/segment_color.py

# Process all images (modify script to remove [:10] slice)
```

## Next Steps
- Compare with other methods (edge detection, deep learning)
- Fine-tune color thresholds for different weather conditions
- Add post-processing for connected component analysis
- Evaluate quantitative metrics (IoU, precision, recall)

## Technical Details
- **Libraries**: OpenCV (cv2), NumPy, tqdm, pathlib
- **Input format**: JPG/PNG images
- **Output format**: PNG images
- **Color space**: HSV for robust color-based segmentation
- **Morphology**: 5x5 kernel for noise reduction

---
Generated: 2026-01-29
Script: segment_color.py
Worker: ULTRAPILOT Worker 1/5
