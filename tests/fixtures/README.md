# Test Fixtures

This directory contains test data for the Road Topology Segmentation project.

## Directory Structure

```
fixtures/
├── synthetic/          # Synthetic test data (generated)
│   ├── highway_001.jpg
│   ├── highway_001_mask.png
│   ├── highway_001_boxes.json
│   └── ...
├── real_data/          # Real-world test samples (optional)
│   └── .gitkeep
└── README.md           # This file
```

## Synthetic Fixtures

Deterministic synthetic test data is generated using the script:

```bash
# First, ensure dependencies are installed
pip install -r requirements.txt

# Generate fixtures
python scripts/generate_test_fixtures.py
```

### Generated Scenes

The generator creates 5 test scenes:

1. **highway_001** - Simple 2-lane highway with 3 vehicles
2. **highway_002** - Complex 4-lane highway with 5 vehicles
3. **intersection_001** - Intersection with crosswalk and 2 vehicles
4. **intersection_002** - Intersection without crosswalk and 2 vehicles
5. **crosswalk_001** - Intersection with prominent crosswalk, 1 vehicle

Each scene consists of:
- `{name}.jpg` - RGB image (640x480)
- `{name}_mask.png` - Ground truth segmentation mask
- `{name}_boxes.json` - Vehicle bounding boxes and metadata

### Class Labels

Segmentation masks use the following class indices:

| Index | Class |
|-------|-------|
| 0 | Background |
| 1 | Road |
| 2 | Lane markings |
| 3 | Crosswalk |
| 4 | Sidewalk |

## Real Data Fixtures

Place real-world test images in `real_data/` directory. These are not version controlled by default.

Recommended naming convention:
```
real_data/
├── sample_001.jpg
├── sample_001_mask.png   # Optional ground truth
└── sample_001_meta.json  # Optional metadata
```

## Usage in Tests

Test fixtures are loaded via pytest fixtures defined in `tests/conftest.py`:

```python
def test_highway_processing(sample_highway_scene):
    """Test using the sample highway scene fixture."""
    image = sample_highway_scene.image      # (H, W, 3)
    mask = sample_highway_scene.mask        # (H, W)
    boxes = sample_highway_scene.boxes      # list[dict]
    metadata = sample_highway_scene.metadata  # dict

    # Your test code here
    assert image.shape == (480, 640, 3)

def test_custom_fixture(load_synthetic_fixture):
    """Load a specific fixture by name."""
    fixture = load_synthetic_fixture("intersection_001")
    assert fixture.metadata["scene_type"] == "intersection"
```

## Regeneration

Fixtures are deterministic (seeded random generation) to ensure reproducible tests. Regenerate them if:
- The generator script is updated
- Fixtures are corrupted or missing
- New test scenarios are needed

```bash
# Regenerate all fixtures
python scripts/generate_test_fixtures.py

# Generate to custom location
python scripts/generate_test_fixtures.py tests/fixtures/synthetic_custom
```

## Git Configuration

By default:
- Synthetic fixtures **ARE** committed to version control (small, deterministic)
- Real data fixtures **ARE NOT** committed (add to `.gitignore` if large)

To exclude synthetic fixtures from git:
```bash
echo "tests/fixtures/synthetic/*.jpg" >> .gitignore
echo "tests/fixtures/synthetic/*.png" >> .gitignore
```
