# Method 3: DeepLabV3 Semantic Segmentation Results

## Overview
This folder contains road segmentation results using **DeepLabV3 ResNet101** pretrained model from torchvision.

**Note**: Originally planned to use SegFormer, but switched to DeepLabV3 due to PyTorch version compatibility issues (PyTorch 1.8.0 incompatible with SegFormer's requirements).

## Model Details
- **Model**: DeepLabV3 with ResNet101 backbone
- **Pretrained on**: PASCAL VOC dataset
- **Device**: CPU (CUDA incompatible with PyTorch 1.8.0 on RTX 4070 Ti)
- **Processing Time**: ~2.2 seconds per image

## Results
- **Processed Images**: 10 sample CCTV images
- **Output Files**: 20 files total (10 masks + 10 visualizations)
  - `*_mask.png`: Binary road mask (white=road, black=non-road)
  - `*_segformer.png`: Blended visualization (gray=road, red=vehicle, green=person)

## Limitations
1. **Class Mapping**: PASCAL VOC doesn't have explicit "road" class
   - Approximated road as background + classes 1-6
   - May include sidewalks and parking areas
2. **Accuracy**: Less accurate than SegFormer would be (which is specifically trained on Cityscapes/ADE20K with road classes)
3. **Performance**: CPU-only due to CUDA compatibility issues

## Recommendations
For production use, consider:
- Upgrading PyTorch to >=1.13 to use SegFormer
- Using Cityscapes-trained models for better road detection
- Fine-tuning on Korean CCTV dataset for better accuracy

## Sample Files
- `01CT000000164_segformer.png` - Urban intersection
- `02CT000000181_segformer.png` - Highway view
- `03CT000000034_segformer.png` - Residential road
