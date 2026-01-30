# CCTV Road Image EDA Report

## Overview
- **Total Images Analyzed**: 90
- **Ready for Training (score >= 70)**: 89 (98.9%)
- **Outliers Detected**: 9 (10.0%)

## Quality Metrics Summary

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Blur Score | 4902.28 | 2324.40 | 1494.77 | 13581.67 |
| Brightness | 120.19 | 13.58 | 64.06 | 156.42 |
| Readiness | 87.85 | 5.19 | 62.69 | 96.32 |

## Time of Day Distribution
- **Day**: 89 images (98.9%)
- **Twilight**: 1 images (1.1%)

## Weather Condition Distribution
- **Clear**: 33 images (36.7%)
- **Overcast**: 12 images (13.3%)
- **Variable**: 45 images (50.0%)

## Outlier Images
| Filename | Reasons | Readiness Score |
|----------|---------|----------------|
| 01CT000000134.jpg | blur_score_high | 79.7 |
| 01CT000000161.jpg | blur_score_high, contrast_high | 82.4 |
| 01CT000000197.jpg | road_coverage_ratio_low | 73.0 |
| 02CT000000148.jpg | contrast_high, road_coverage_ratio_low | 82.9 |
| 02CT000000185.jpg | blur_score_high, contrast_high | 84.5 |
| 03CT000000044.jpg | brightness_high | 78.5 |
| 03CT000000097.jpg | brightness_low, road_coverage_ratio_low | 71.4 |
| 04CT000000527.jpg | road_coverage_ratio_low | 70.5 |
| 05CT000000450.jpg | contrast_low, brightness_low | 62.7 |

## Recommendations
- **Good Overall Quality**: Dataset appears suitable for model training.

## Visualizations
See the `visualizations/` directory for detailed plots:
- `blur_distribution.png`
- `brightness_histogram.png`
- `color_histograms_per_camera.png`
- `road_coverage_estimate.png`
- `condition_classification.png`
- `outlier_scatter.png`
- `readiness_scores.png`
- `sample_grid.png`
