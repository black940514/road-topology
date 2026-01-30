#!/usr/bin/env python3
"""
DeepLabV3 Semantic Segmentation for Road Topology
Uses PyTorch DeepLabV3 pretrained on COCO for road-related segmentation
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
from tqdm import tqdm
import sys


def get_color_palette():
    """COCO/Pascal VOC color palette for 21 classes"""
    return np.array([
        [0, 0, 0],       # 0: background
        [128, 0, 0],     # 1: aeroplane
        [0, 128, 0],     # 2: bicycle
        [128, 128, 0],   # 3: bird
        [0, 0, 128],     # 4: boat
        [128, 0, 128],   # 5: bottle
        [0, 128, 128],   # 6: bus
        [128, 128, 128], # 7: car
        [64, 0, 0],      # 8: cat
        [192, 0, 0],     # 9: chair
        [64, 128, 0],    # 10: cow
        [192, 128, 0],   # 11: diningtable
        [64, 0, 128],    # 12: dog
        [192, 0, 128],   # 13: horse
        [64, 128, 128],  # 14: motorbike
        [192, 128, 128], # 15: person
        [0, 64, 0],      # 16: pottedplant
        [128, 64, 0],    # 17: sheep
        [0, 192, 0],     # 18: sofa
        [128, 192, 0],   # 19: train
        [0, 64, 128],    # 20: tvmonitor
    ], dtype=np.uint8)


def segment_with_deeplab(image_path, output_dir, model, transform, device, palette):
    """
    DeepLabV3 semantic segmentation

    Args:
        image_path: Path to input image
        output_dir: Output directory
        model: DeepLabV3 model
        transform: Preprocessing transform
        device: torch device
        palette: Color palette for visualization

    Returns:
        Path to output visualization
    """
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)['out'][0]

        # Get predictions
        pred = output.argmax(0).cpu().numpy()

        # Apply color palette
        colored = palette[pred % len(palette)]

        # Resize to original size
        colored_resized = cv2.resize(colored, original_size)

        # Blend with original image
        img_np = np.array(image)
        blended = cv2.addWeighted(img_np, 0.6, colored_resized, 0.4, 0)

        # Save blended visualization
        output_path = output_dir / f"{image_path.stem}_deeplab.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

        # Create binary mask for road-related classes
        # Classes: 6=bus, 7=car, 14=motorbike, 2=bicycle
        road_related = np.isin(pred, [2, 6, 7, 14]).astype(np.uint8) * 255
        road_resized = cv2.resize(road_related, original_size, interpolation=cv2.INTER_NEAREST)

        # Save mask
        mask_path = output_dir / f"{image_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), road_resized)

        # Save raw prediction map
        pred_path = output_dir / f"{image_path.stem}_pred.npy"
        np.save(str(pred_path), pred)

        return output_path

    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")
        return None


def main():
    # Setup device (force CPU due to CUDA compatibility issues)
    device = "cpu"
    print(f"Using device: {device}")

    # Load pretrained DeepLabV3 model
    print("Loading DeepLabV3-ResNet101 model...")
    model = deeplabv3_resnet101(pretrained=True).to(device)
    model.eval()

    # Standard ImageNet normalization
    transform = T.Compose([
        T.Resize((520, 520)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get color palette
    palette = get_color_palette()

    # Setup paths
    input_dir = Path("/home/bamboos/workspace/black/road_topology/output_dir3")
    output_dir = Path("/home/bamboos/workspace/black/road_topology/results/method5_deeplab")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all images
    images = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
    print(f"Found {len(images)} images")

    # Process sample (first 10 images for quick results)
    sample_size = min(10, len(images))
    print(f"Processing {sample_size} sample images with DeepLabV3...")

    successful = 0
    for img_path in tqdm(images[:sample_size], desc="Segmenting"):
        result = segment_with_deeplab(img_path, output_dir, model, transform, device, palette)
        if result:
            successful += 1

    print(f"\n✓ DeepLabV3 segmentation complete!")
    print(f"  Successfully processed: {successful}/{sample_size} images")
    print(f"  Output directory: {output_dir}")
    print(f"  Generated files per image:")
    print(f"    - *_deeplab.png: Blended visualization")
    print(f"    - *_mask.png: Binary mask (vehicles)")
    print(f"    - *_pred.npy: Raw prediction map")

    # Show sample results
    print("\nSample results:")
    for f in sorted(output_dir.glob("*_deeplab.png"))[:3]:
        print(f"  {f.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
