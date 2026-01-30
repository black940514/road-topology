"""Pytest configuration and shared fixtures for road_topology tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

if TYPE_CHECKING:
    from dataclasses import dataclass

    @dataclass
    class TestFixture:
        """Test fixture data container."""
        image: np.ndarray
        mask: np.ndarray
        boxes: list[dict]
        metadata: dict


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def synthetic_fixtures_dir(fixtures_dir: Path) -> Path:
    """Return path to synthetic test fixtures."""
    return fixtures_dir / "synthetic"


@pytest.fixture(scope="session")
def real_data_fixtures_dir(fixtures_dir: Path) -> Path:
    """Return path to real data test fixtures."""
    return fixtures_dir / "real_data"


@pytest.fixture
def load_synthetic_fixture(synthetic_fixtures_dir: Path):
    """Factory fixture to load synthetic test data by name.

    Usage:
        fixture = load_synthetic_fixture("highway_001")
        image = fixture.image  # np.ndarray (H, W, 3)
        mask = fixture.mask    # np.ndarray (H, W)
        boxes = fixture.boxes  # list[dict]
    """
    def _load(name: str):
        from dataclasses import dataclass

        @dataclass
        class TestFixture:
            image: np.ndarray
            mask: np.ndarray
            boxes: list[dict]
            metadata: dict

        image_path = synthetic_fixtures_dir / f"{name}.jpg"
        mask_path = synthetic_fixtures_dir / f"{name}_mask.png"
        boxes_path = synthetic_fixtures_dir / f"{name}_boxes.json"

        if not image_path.exists():
            raise FileNotFoundError(
                f"Fixture {name} not found. Run: python scripts/generate_test_fixtures.py"
            )

        # Load image as RGB
        image = np.array(Image.open(image_path).convert("RGB"))

        # Load mask as grayscale
        mask = np.array(Image.open(mask_path).convert("L"))

        # Load boxes and metadata
        with open(boxes_path) as f:
            data = json.load(f)

        return TestFixture(
            image=image,
            mask=mask,
            boxes=data["boxes"],
            metadata=data["metadata"],
        )

    return _load


@pytest.fixture
def sample_highway_scene(load_synthetic_fixture):
    """Load the default highway test scene."""
    return load_synthetic_fixture("highway_001")


@pytest.fixture
def sample_intersection_scene(load_synthetic_fixture):
    """Load the default intersection test scene."""
    return load_synthetic_fixture("intersection_001")


@pytest.fixture
def sample_image_rgb() -> np.ndarray:
    """Generate a simple test RGB image (100x100)."""
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask() -> np.ndarray:
    """Generate a simple test mask (100x100) with 5 classes."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 20:80] = 1  # Road
    mask[30:35, 20:80] = 2  # Lane
    mask[50:60, 30:70] = 3  # Crosswalk
    mask[10:20, 10:90] = 4  # Sidewalk
    return mask


@pytest.fixture
def sample_boxes() -> list[dict]:
    """Generate sample vehicle bounding boxes."""
    return [
        {
            "x1": 10,
            "y1": 20,
            "x2": 50,
            "y2": 80,
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.95,
        },
        {
            "x1": 60,
            "y1": 30,
            "x2": 90,
            "y2": 70,
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.88,
        },
    ]


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset numpy random seed before each test for reproducibility."""
    np.random.seed(42)
    yield
    # Cleanup if needed
