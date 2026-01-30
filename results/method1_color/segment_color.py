#!/usr/bin/env python3
"""
Color-based Road Segmentation using HSV color space
HSV 색상 공간 기반 도로 영역 추출
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def segment_road_by_color(image_path, output_dir):
    """
    HSV 기반 도로 영역 추출
    Extract road regions using HSV color thresholding

    Args:
        image_path: Path to input image
        output_dir: Directory to save results

    Returns:
        Path to output image
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not read {image_path}")
        return None

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define road color range (gray asphalt)
    # H: 0-180 (all hues for gray)
    # S: 0-50 (low saturation for gray)
    # V: 40-180 (medium to high value)
    lower_road = np.array([0, 0, 40])
    upper_road = np.array([180, 50, 180])

    # Create mask
    mask = cv2.inRange(hsv, lower_road, upper_road)

    # Morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove small noise

    # Create visualization
    result = cv2.bitwise_and(img, img, mask=mask)
    overlay = img.copy()
    overlay[mask > 0] = [128, 128, 128]  # Gray color for road regions
    blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

    # Save blended result
    output_path = output_dir / f"{image_path.stem}_color.png"
    cv2.imwrite(str(output_path), blended)

    # Save binary mask
    mask_path = output_dir / f"{image_path.stem}_mask.png"
    cv2.imwrite(str(mask_path), mask)

    # Save isolated road
    isolated_path = output_dir / f"{image_path.stem}_isolated.png"
    cv2.imwrite(str(isolated_path), result)

    return output_path

def main():
    """Main execution function"""
    # Setup paths
    input_dir = Path("/home/bamboos/workspace/black/road_topology/output_dir3")
    output_dir = Path("/home/bamboos/workspace/black/road_topology/results/method1_color")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get all images
    images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    print(f"Found {len(images)} images")

    if len(images) == 0:
        print("Error: No images found in input directory")
        return

    # Process sample (first 10 images)
    print(f"Processing {min(10, len(images))} sample images with color-based segmentation...")

    success_count = 0
    for img_path in tqdm(images[:10]):
        result = segment_road_by_color(img_path, output_dir)
        if result is not None:
            success_count += 1

    print(f"\nColor segmentation complete!")
    print(f"Successfully processed: {success_count}/{min(10, len(images))} images")
    print(f"Results saved to: {output_dir}")

    # List output files
    output_files = sorted(output_dir.glob("*"))
    print(f"\nGenerated {len(output_files)} output files:")
    for f in output_files[:15]:  # Show first 15 files
        print(f"  - {f.name}")
    if len(output_files) > 15:
        print(f"  ... and {len(output_files) - 15} more files")

if __name__ == "__main__":
    main()
