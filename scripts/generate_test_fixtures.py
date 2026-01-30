#!/usr/bin/env python3
"""Synthetic test data generator for Road Topology Segmentation.

Generates deterministic synthetic images with procedural roads for testing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Class indices
BACKGROUND = 0
ROAD = 1
LANE = 2
CROSSWALK = 3
SIDEWALK = 4


@dataclass
class SyntheticFixture:
    """Container for synthetic test data."""
    image: np.ndarray       # RGB image (H, W, 3)
    mask: np.ndarray        # Class indices (H, W)
    boxes: list[dict]       # Vehicle bounding boxes
    metadata: dict          # Scene parameters


def generate_highway_scene(
    width: int = 640,
    height: int = 480,
    num_lanes: int = 2,
    num_vehicles: int = 3,
    seed: int = 42,
) -> SyntheticFixture:
    """Generate a highway scene with road, lanes, and vehicles.

    Args:
        width: Image width.
        height: Image height.
        num_lanes: Number of lanes.
        num_vehicles: Number of vehicles to place.
        seed: Random seed for reproducibility.

    Returns:
        SyntheticFixture with image, mask, boxes, and metadata.
    """
    np.random.seed(seed)

    # Create blank image and mask
    image = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Fill with sky/background color
    image[:] = (135, 206, 235)  # Sky blue

    # Draw road
    road_top = height // 3
    road_bottom = height
    road_color = (64, 64, 64)  # Dark gray

    image[road_top:road_bottom, :] = road_color
    mask[road_top:road_bottom, :] = ROAD

    # Draw sidewalk on sides
    sidewalk_width = 40
    sidewalk_color = (180, 180, 180)  # Light gray

    # Left sidewalk
    image[road_top:road_bottom, :sidewalk_width] = sidewalk_color
    mask[road_top:road_bottom, :sidewalk_width] = SIDEWALK

    # Right sidewalk
    image[road_top:road_bottom, -sidewalk_width:] = sidewalk_color
    mask[road_top:road_bottom, -sidewalk_width:] = SIDEWALK

    # Draw lane markings
    lane_y = road_top + (road_bottom - road_top) // 2
    lane_width = width - 2 * sidewalk_width
    lane_segment_length = 40
    lane_gap = 20

    for i in range(num_lanes - 1):
        lane_x = sidewalk_width + (i + 1) * lane_width // num_lanes

        # Dashed lane line
        y = road_top + 20
        while y < road_bottom - 20:
            y1, y2 = y, min(y + lane_segment_length, road_bottom - 20)
            image[y1:y2, lane_x-2:lane_x+2] = (255, 255, 255)
            mask[y1:y2, lane_x-2:lane_x+2] = LANE
            y += lane_segment_length + lane_gap

    # Generate vehicle bounding boxes
    boxes = []
    vehicle_colors = [(200, 50, 50), (50, 50, 200), (50, 200, 50), (200, 200, 50)]

    for i in range(num_vehicles):
        # Random position on road
        veh_width = np.random.randint(40, 60)
        veh_height = np.random.randint(60, 90)

        lane_idx = i % num_lanes
        lane_center = sidewalk_width + (lane_idx + 0.5) * lane_width / num_lanes

        veh_x = int(lane_center - veh_width // 2)
        veh_y = np.random.randint(road_top + 50, road_bottom - veh_height - 20)

        # Draw vehicle
        color = vehicle_colors[i % len(vehicle_colors)]
        image[veh_y:veh_y+veh_height, veh_x:veh_x+veh_width] = color

        # Add bounding box
        boxes.append({
            "x1": veh_x,
            "y1": veh_y,
            "x2": veh_x + veh_width,
            "y2": veh_y + veh_height,
            "class_id": 2,  # car
            "class_name": "car",
            "confidence": 0.95,
        })

    metadata = {
        "scene_type": "highway",
        "num_lanes": num_lanes,
        "num_vehicles": num_vehicles,
        "seed": seed,
        "width": width,
        "height": height,
    }

    return SyntheticFixture(image=image, mask=mask, boxes=boxes, metadata=metadata)


def generate_intersection_scene(
    width: int = 640,
    height: int = 480,
    has_crosswalk: bool = True,
    num_vehicles: int = 2,
    seed: int = 42,
) -> SyntheticFixture:
    """Generate an intersection scene with optional crosswalk.

    Args:
        width: Image width.
        height: Image height.
        has_crosswalk: Whether to include a crosswalk.
        num_vehicles: Number of vehicles to place.
        seed: Random seed for reproducibility.

    Returns:
        SyntheticFixture with image, mask, boxes, and metadata.
    """
    np.random.seed(seed)

    # Create blank image and mask
    image = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Fill with background
    image[:] = (100, 150, 100)  # Grass green

    road_color = (64, 64, 64)
    road_width = 150

    # Horizontal road
    h_road_top = height // 2 - road_width // 2
    h_road_bottom = height // 2 + road_width // 2
    image[h_road_top:h_road_bottom, :] = road_color
    mask[h_road_top:h_road_bottom, :] = ROAD

    # Vertical road
    v_road_left = width // 2 - road_width // 2
    v_road_right = width // 2 + road_width // 2
    image[:, v_road_left:v_road_right] = road_color
    mask[:, v_road_left:v_road_right] = ROAD

    # Draw crosswalk if enabled
    if has_crosswalk:
        crosswalk_y = h_road_top + 10
        stripe_width = 8
        stripe_gap = 8

        x = v_road_left + 10
        while x < v_road_right - 10:
            image[crosswalk_y:crosswalk_y+road_width-20, x:x+stripe_width] = (255, 255, 255)
            mask[crosswalk_y:crosswalk_y+road_width-20, x:x+stripe_width] = CROSSWALK
            x += stripe_width + stripe_gap

    # Draw center lane lines
    center_y = height // 2
    center_x = width // 2

    # Horizontal center line
    image[center_y-2:center_y+2, :v_road_left-20] = (255, 255, 0)
    image[center_y-2:center_y+2, v_road_right+20:] = (255, 255, 0)
    mask[center_y-2:center_y+2, :v_road_left-20] = LANE
    mask[center_y-2:center_y+2, v_road_right+20:] = LANE

    # Generate vehicle bounding boxes
    boxes = []
    vehicle_positions = [
        (v_road_left - 80, center_y + 20),  # Left of intersection
        (v_road_right + 20, center_y - 60),  # Right of intersection
    ]

    for i in range(min(num_vehicles, len(vehicle_positions))):
        veh_x, veh_y = vehicle_positions[i]
        veh_width, veh_height = 50, 70

        color = (200, 50, 50) if i % 2 == 0 else (50, 50, 200)

        # Clip to image bounds
        x1 = max(0, veh_x)
        y1 = max(0, veh_y)
        x2 = min(width, veh_x + veh_width)
        y2 = min(height, veh_y + veh_height)

        if x2 > x1 and y2 > y1:
            image[y1:y2, x1:x2] = color
            boxes.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class_id": 2,
                "class_name": "car",
                "confidence": 0.92,
            })

    metadata = {
        "scene_type": "intersection",
        "has_crosswalk": has_crosswalk,
        "num_vehicles": len(boxes),
        "seed": seed,
        "width": width,
        "height": height,
    }

    return SyntheticFixture(image=image, mask=mask, boxes=boxes, metadata=metadata)


def generate_all_fixtures(
    output_dir: Path = Path("tests/fixtures/synthetic"),
    seed: int = 42,
) -> None:
    """Generate all synthetic test fixtures.

    Args:
        output_dir: Output directory for fixtures.
        seed: Base random seed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fixtures = [
        ("highway_001", generate_highway_scene(seed=seed)),
        ("highway_002", generate_highway_scene(num_lanes=4, num_vehicles=5, seed=seed+1)),
        ("intersection_001", generate_intersection_scene(seed=seed+2)),
        ("intersection_002", generate_intersection_scene(has_crosswalk=False, seed=seed+3)),
        ("crosswalk_001", generate_intersection_scene(has_crosswalk=True, num_vehicles=1, seed=seed+4)),
    ]

    for name, fixture in fixtures:
        # Save image
        Image.fromarray(fixture.image).save(output_dir / f"{name}.jpg", quality=95)

        # Save mask as PNG (lossless)
        Image.fromarray(fixture.mask.astype(np.uint8)).save(output_dir / f"{name}_mask.png")

        # Save boxes as JSON
        with open(output_dir / f"{name}_boxes.json", "w") as f:
            json.dump({
                "boxes": fixture.boxes,
                "metadata": fixture.metadata,
            }, f, indent=2)

        print(f"Generated: {name}")

    # Create README
    readme_content = """# Synthetic Test Fixtures

Deterministic synthetic test data for Road Topology Segmentation.

## Files

Each scene has three files:
- `{name}.jpg` - RGB image (640x480)
- `{name}_mask.png` - Ground truth segmentation mask
- `{name}_boxes.json` - Vehicle bounding boxes and metadata

## Class Labels

- 0: Background
- 1: Road
- 2: Lane markings
- 3: Crosswalk
- 4: Sidewalk

## Regeneration

To regenerate fixtures:
```bash
python scripts/generate_test_fixtures.py
```

Fixtures are deterministic (fixed seed) for reproducible tests.
"""

    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)

    print(f"\nAll fixtures saved to: {output_dir}")


if __name__ == "__main__":
    import sys

    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures/synthetic")
    generate_all_fixtures(output)
