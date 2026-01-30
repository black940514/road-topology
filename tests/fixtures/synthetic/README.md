# Synthetic Test Fixtures

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
