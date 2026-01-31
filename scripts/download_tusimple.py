#!/usr/bin/env python3
"""Download and convert TuSimple Lane Dataset to segmentation masks.

This script downloads the TuSimple dataset and converts lane polyline
annotations to semantic/instance segmentation masks compatible with
our lane segmentation training pipeline.

Usage:
    python scripts/download_tusimple.py --output data/tusimple

Output structure:
    data/tusimple/
    ├── train/
    │   ├── images/
    │   ├── semantic_masks/
    │   └── instance_masks/
    └── val/
        ├── images/
        ├── semantic_masks/
        └── instance_masks/
"""

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def download_tusimple(output_dir: Path) -> Path:
    """Download TuSimple dataset from Hugging Face or use cached version."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download

        print("Downloading TuSimple dataset from Hugging Face...")
        print("This may take a while (dataset is ~3GB)...")

        # Try to download the dataset
        cache_dir = output_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Download from dhbloo/TuSimple dataset
        dataset_path = snapshot_download(
            repo_id="dhbloo/TuSimple",
            repo_type="dataset",
            local_dir=cache_dir / "tusimple_raw",
            local_dir_use_symlinks=False,
        )
        return Path(dataset_path)

    except Exception as e:
        print(f"Error downloading from Hugging Face: {e}")
        print("\nAlternative: Manual download required")
        print("1. Go to: https://www.kaggle.com/datasets/manideep1108/tusimple")
        print("2. Download the dataset")
        print(f"3. Extract to: {output_dir / 'cache' / 'tusimple_raw'}")
        raise


def parse_tusimple_json(json_path: Path) -> list:
    """Parse TuSimple JSON annotation file.

    TuSimple uses JSON Lines format (one JSON object per line).
    """
    annotations = []
    with open(json_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    annotations.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return annotations


def create_lane_mask(
    lanes: list,
    h_samples: list,
    img_height: int = 720,
    img_width: int = 1280,
    lane_width: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert lane polylines to semantic and instance segmentation masks.

    Args:
        lanes: List of lane x-coordinates (list of lists)
        h_samples: Y-coordinates for each lane point
        img_height: Image height
        img_width: Image width
        lane_width: Width of lane line in pixels

    Returns:
        semantic_mask: Binary mask (0=background, 1=lane)
        instance_mask: Instance mask (0=background, 1-N=lane instances)
    """
    semantic_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    instance_mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # Sort lanes by their average x-coordinate (left to right)
    lane_data = []
    for lane_idx, lane in enumerate(lanes):
        valid_points = [(x, y) for x, y in zip(lane, h_samples) if x >= 0]
        if len(valid_points) >= 2:
            avg_x = np.mean([p[0] for p in valid_points])
            lane_data.append((avg_x, valid_points, lane_idx))

    # Sort by x position (left to right)
    lane_data.sort(key=lambda x: x[0])

    # Draw lanes
    for instance_id, (avg_x, points, orig_idx) in enumerate(lane_data, start=1):
        points_array = np.array(points, dtype=np.int32)

        # Draw on semantic mask (all lanes = class 1)
        cv2.polylines(
            semantic_mask,
            [points_array],
            isClosed=False,
            color=1,
            thickness=lane_width,
        )

        # Draw on instance mask (each lane = different instance ID)
        cv2.polylines(
            instance_mask,
            [points_array],
            isClosed=False,
            color=instance_id,
            thickness=lane_width,
        )

    return semantic_mask, instance_mask


def convert_tusimple(
    raw_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.9,
    lane_width: int = 8,
) -> dict:
    """Convert TuSimple dataset to our segmentation format.

    Args:
        raw_dir: Path to raw TuSimple data
        output_dir: Output directory
        train_ratio: Ratio of training data
        lane_width: Width of lane lines in pixels

    Returns:
        Statistics dictionary
    """
    print(f"Converting TuSimple dataset from {raw_dir}")

    # Find annotation files
    train_json = list(raw_dir.rglob("*label_data_*.json"))
    if not train_json:
        # Try alternative paths
        train_json = list(raw_dir.rglob("*.json"))

    if not train_json:
        raise FileNotFoundError(f"No JSON annotation files found in {raw_dir}")

    print(f"Found {len(train_json)} annotation files")

    # Parse all annotations
    all_annotations = []
    for json_path in train_json:
        annotations = parse_tusimple_json(json_path)
        for ann in annotations:
            ann['_json_path'] = str(json_path)
            ann['_base_dir'] = str(json_path.parent)
        all_annotations.extend(annotations)

    print(f"Total annotations: {len(all_annotations)}")

    # Split into train/val
    np.random.seed(42)
    np.random.shuffle(all_annotations)

    split_idx = int(len(all_annotations) * train_ratio)
    train_annotations = all_annotations[:split_idx]
    val_annotations = all_annotations[split_idx:]

    print(f"Train: {len(train_annotations)}, Val: {len(val_annotations)}")

    # Create output directories
    for split in ['train', 'val']:
        for subdir in ['images', 'semantic_masks', 'instance_masks']:
            (output_dir / split / subdir).mkdir(parents=True, exist_ok=True)

    stats = {
        'total_images': 0,
        'train_images': 0,
        'val_images': 0,
        'skipped': 0,
        'lane_counts': {},
    }

    # Process annotations
    splits = [
        ('train', train_annotations),
        ('val', val_annotations),
    ]

    for split_name, annotations in splits:
        print(f"\nProcessing {split_name} set...")

        for ann in tqdm(annotations, desc=split_name):
            try:
                # Find image path
                raw_file = ann.get('raw_file', '')
                base_dir = Path(ann.get('_base_dir', raw_dir))

                # Try different path combinations
                img_path = None
                for candidate in [
                    base_dir / raw_file,
                    raw_dir / raw_file,
                    base_dir.parent / raw_file,
                    raw_dir.parent / raw_file,
                ]:
                    if candidate.exists():
                        img_path = candidate
                        break

                if img_path is None:
                    stats['skipped'] += 1
                    continue

                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    stats['skipped'] += 1
                    continue

                h, w = img.shape[:2]

                # Get lane annotations
                lanes = ann.get('lanes', [])
                h_samples = ann.get('h_samples', [])

                if not lanes or not h_samples:
                    stats['skipped'] += 1
                    continue

                # Create masks
                semantic_mask, instance_mask = create_lane_mask(
                    lanes, h_samples, h, w, lane_width
                )

                # Count lanes
                num_lanes = int(instance_mask.max())
                stats['lane_counts'][num_lanes] = stats['lane_counts'].get(num_lanes, 0) + 1

                # Generate unique filename
                img_name = f"{split_name}_{stats['total_images']:06d}.png"

                # Save files
                cv2.imwrite(str(output_dir / split_name / 'images' / img_name), img)
                cv2.imwrite(
                    str(output_dir / split_name / 'semantic_masks' / img_name),
                    semantic_mask
                )
                cv2.imwrite(
                    str(output_dir / split_name / 'instance_masks' / img_name),
                    instance_mask
                )

                stats['total_images'] += 1
                stats[f'{split_name}_images'] += 1

            except Exception as e:
                print(f"Error processing annotation: {e}")
                stats['skipped'] += 1
                continue

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Download and convert TuSimple Lane Dataset'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('data/tusimple'),
        help='Output directory'
    )
    parser.add_argument(
        '--lane-width',
        type=int,
        default=8,
        help='Width of lane lines in pixels'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download if already cached'
    )
    args = parser.parse_args()

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = output_dir / 'cache' / 'tusimple_raw'

    # Download or use cached
    if args.skip_download and cache_dir.exists():
        print(f"Using cached dataset at {cache_dir}")
        raw_dir = cache_dir
    else:
        raw_dir = download_tusimple(output_dir)

    # Convert to our format
    stats = convert_tusimple(
        raw_dir,
        output_dir,
        lane_width=args.lane_width,
    )

    print("\n" + "=" * 50)
    print("Conversion complete!")
    print("=" * 50)
    print(f"Total images: {stats['total_images']}")
    print(f"Train images: {stats['train_images']}")
    print(f"Val images: {stats['val_images']}")
    print(f"Skipped: {stats['skipped']}")
    print("\nLane distribution:")
    for num_lanes, count in sorted(stats['lane_counts'].items()):
        print(f"  {num_lanes} lanes: {count} images")

    print(f"\nOutput saved to: {output_dir}")
    print("\nTo use with training:")
    print(f"  road-topology train lane --data {output_dir} --config config.yaml")


if __name__ == '__main__':
    main()
