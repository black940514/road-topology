#!/usr/bin/env python3
"""
CCTV Road Image EDA Script
Analyzes CCTV road images for quality, color, road coverage, and readiness.
"""

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from tqdm import tqdm


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


@dataclass
class ImageMetrics:
    """Container for all image analysis metrics."""
    # Basic info
    filename: str = ""
    camera_prefix: str = ""

    # Quality metrics
    blur_score: float = 0.0
    noise_estimate: float = 0.0
    contrast: float = 0.0
    brightness: float = 0.0

    # Color metrics
    rgb_means: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    hsv_means: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dominant_colors: List[Tuple[int, int, int]] = field(default_factory=list)

    # Road metrics
    road_coverage_ratio: float = 0.0

    # Condition metrics
    time_of_day: str = "unknown"
    weather_condition: str = "unknown"

    # Outlier detection
    is_outlier: bool = False
    outlier_reasons: List[str] = field(default_factory=list)

    # Overall score
    readiness_score: float = 0.0


class QualityAnalyzer:
    """Analyzes image quality: blur, contrast, brightness, noise."""

    @staticmethod
    def compute_blur_score(gray: np.ndarray) -> float:
        """Compute blur score using Laplacian variance. Higher = sharper."""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    @staticmethod
    def compute_contrast(gray: np.ndarray) -> float:
        """Compute RMS contrast."""
        return float(gray.std())

    @staticmethod
    def compute_brightness(gray: np.ndarray) -> float:
        """Compute mean brightness."""
        return float(gray.mean())

    @staticmethod
    def estimate_noise(gray: np.ndarray) -> float:
        """Estimate noise using Laplacian-based method."""
        h, w = gray.shape
        # Use median absolute deviation of Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sigma = np.median(np.abs(laplacian)) / 0.6745
        return float(sigma)

    def analyze(self, image: np.ndarray) -> dict:
        """Analyze image quality metrics."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return {
            "blur_score": self.compute_blur_score(gray),
            "contrast": self.compute_contrast(gray),
            "brightness": self.compute_brightness(gray),
            "noise_estimate": self.estimate_noise(gray)
        }


class ColorAnalyzer:
    """Analyzes color properties: RGB/HSV means, dominant colors."""

    @staticmethod
    def compute_rgb_means(image: np.ndarray) -> Tuple[float, float, float]:
        """Compute mean RGB values."""
        b, g, r = cv2.split(image)
        return (float(r.mean()), float(g.mean()), float(b.mean()))

    @staticmethod
    def compute_hsv_means(image: np.ndarray) -> Tuple[float, float, float]:
        """Compute mean HSV values."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        return (float(h.mean()), float(s.mean()), float(v.mean()))

    @staticmethod
    def find_dominant_colors(image: np.ndarray, n_colors: int = 5) -> List[Tuple[int, int, int]]:
        """Find dominant colors using K-means clustering."""
        pixels = image.reshape(-1, 3).astype(np.float32)
        # Sample for efficiency
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]

        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        # Convert BGR to RGB
        colors_rgb = [(int(c[2]), int(c[1]), int(c[0])) for c in colors]
        return colors_rgb

    def analyze(self, image: np.ndarray) -> dict:
        """Analyze color metrics."""
        return {
            "rgb_means": self.compute_rgb_means(image),
            "hsv_means": self.compute_hsv_means(image),
            "dominant_colors": self.find_dominant_colors(image)
        }


class RoadCoverageAnalyzer:
    """Analyzes road coverage using heuristic gray-color detection."""

    @staticmethod
    def estimate_road_coverage(image: np.ndarray) -> float:
        """
        Estimate road coverage ratio in lower 60% of image.
        Roads typically appear as gray-ish areas with low saturation.
        """
        h, w = image.shape[:2]
        lower_region = image[int(h * 0.4):, :]  # Lower 60%

        hsv = cv2.cvtColor(lower_region, cv2.COLOR_BGR2HSV)

        # Road detection: low saturation, moderate value (gray tones)
        # Also detect darker asphalt
        lower_gray = np.array([0, 0, 30])
        upper_gray = np.array([180, 80, 200])

        mask = cv2.inRange(hsv, lower_gray, upper_gray)

        # Apply morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        road_pixels = np.sum(mask > 0)
        total_pixels = mask.size

        return float(road_pixels / total_pixels)

    def analyze(self, image: np.ndarray) -> dict:
        """Analyze road coverage."""
        return {
            "road_coverage_ratio": self.estimate_road_coverage(image)
        }


class ConditionClassifier:
    """Classifies time of day and weather conditions."""

    @staticmethod
    def classify_time_of_day(brightness: float, hsv_means: Tuple[float, float, float]) -> str:
        """
        Classify time of day based on brightness and color.
        - Night: very low brightness
        - Twilight: low brightness with some color
        - Day: normal to high brightness
        """
        _, saturation, value = hsv_means

        if brightness < 40:
            return "night"
        elif brightness < 80:
            return "twilight"
        else:
            return "day"

    @staticmethod
    def classify_weather(contrast: float, saturation: float, brightness: float) -> str:
        """
        Estimate weather condition heuristically.
        - Clear: good contrast, normal saturation
        - Overcast: reduced contrast, low saturation
        - Foggy: very low contrast, low saturation, high brightness
        - Rainy: low contrast, reflections (hard to detect)
        """
        if contrast < 30 and brightness > 150:
            return "foggy"
        elif contrast < 40 and saturation < 40:
            return "overcast"
        elif contrast > 50:
            return "clear"
        else:
            return "variable"

    def analyze(self, quality_metrics: dict, color_metrics: dict) -> dict:
        """Classify environmental conditions."""
        brightness = quality_metrics["brightness"]
        contrast = quality_metrics["contrast"]
        hsv_means = color_metrics["hsv_means"]
        saturation = hsv_means[1]

        return {
            "time_of_day": self.classify_time_of_day(brightness, hsv_means),
            "weather_condition": self.classify_weather(contrast, saturation, brightness)
        }


class OutlierDetector:
    """Detects outlier images using Isolation Forest."""

    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.feature_names = ["blur_score", "contrast", "brightness", "road_coverage_ratio"]

    def fit_predict(self, metrics_list: List[ImageMetrics]) -> List[Tuple[bool, List[str]]]:
        """Fit model and predict outliers."""
        features = np.array([
            [m.blur_score, m.contrast, m.brightness, m.road_coverage_ratio]
            for m in metrics_list
        ])

        # Normalize features
        means = features.mean(axis=0)
        stds = features.std(axis=0) + 1e-8
        features_norm = (features - means) / stds

        predictions = self.model.fit_predict(features_norm)

        results = []
        for i, (pred, feat) in enumerate(zip(predictions, features)):
            is_outlier = pred == -1
            reasons = []
            if is_outlier:
                # Determine why it's an outlier
                z_scores = (feat - means) / stds
                for j, (name, z) in enumerate(zip(self.feature_names, z_scores)):
                    if abs(z) > 2:
                        direction = "high" if z > 0 else "low"
                        reasons.append(f"{name}_{direction}")
            results.append((is_outlier, reasons))

        return results


class ReadinessScorer:
    """Computes readiness score for model training."""

    # Weights for different factors
    WEIGHTS = {
        "blur": 0.25,
        "contrast": 0.20,
        "brightness": 0.15,
        "road_coverage": 0.25,
        "not_outlier": 0.15
    }

    @staticmethod
    def normalize_blur(blur_score: float) -> float:
        """Normalize blur score to 0-1. Higher blur variance = better."""
        # Typical range: 10-500
        return min(1.0, blur_score / 200)

    @staticmethod
    def normalize_contrast(contrast: float) -> float:
        """Normalize contrast to 0-1."""
        # Typical range: 20-80
        return min(1.0, max(0, (contrast - 20) / 60))

    @staticmethod
    def normalize_brightness(brightness: float) -> float:
        """Normalize brightness to 0-1. Penalize extremes."""
        # Ideal range: 80-180
        if brightness < 40 or brightness > 220:
            return 0.3
        elif brightness < 80 or brightness > 180:
            return 0.7
        else:
            return 1.0

    def compute_score(self, metrics: ImageMetrics) -> float:
        """Compute weighted readiness score (0-100)."""
        blur_score = self.normalize_blur(metrics.blur_score)
        contrast_score = self.normalize_contrast(metrics.contrast)
        brightness_score = self.normalize_brightness(metrics.brightness)
        road_score = min(1.0, metrics.road_coverage_ratio / 0.5)  # Expect ~50% road
        outlier_score = 0.0 if metrics.is_outlier else 1.0

        weighted_sum = (
            self.WEIGHTS["blur"] * blur_score +
            self.WEIGHTS["contrast"] * contrast_score +
            self.WEIGHTS["brightness"] * brightness_score +
            self.WEIGHTS["road_coverage"] * road_score +
            self.WEIGHTS["not_outlier"] * outlier_score
        )

        return round(weighted_sum * 100, 2)


class VisualizationGenerator:
    """Generates all visualization plots."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_blur_distribution(self, metrics_list: List[ImageMetrics]):
        """Plot blur score distribution."""
        fig, ax = plt.subplots(figsize=(10, 6))
        blur_scores = [m.blur_score for m in metrics_list]
        ax.hist(blur_scores, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(blur_scores), color='red', linestyle='--', label=f'Mean: {np.mean(blur_scores):.1f}')
        ax.set_xlabel('Blur Score (Laplacian Variance)')
        ax.set_ylabel('Count')
        ax.set_title('Blur Score Distribution')
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'blur_distribution.png', dpi=150)
        plt.close()

    def plot_brightness_histogram(self, metrics_list: List[ImageMetrics]):
        """Plot brightness histogram."""
        fig, ax = plt.subplots(figsize=(10, 6))
        brightness = [m.brightness for m in metrics_list]
        ax.hist(brightness, bins=30, edgecolor='black', alpha=0.7, color='orange')
        ax.axvline(np.mean(brightness), color='red', linestyle='--', label=f'Mean: {np.mean(brightness):.1f}')
        ax.set_xlabel('Mean Brightness')
        ax.set_ylabel('Count')
        ax.set_title('Brightness Distribution')
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'brightness_histogram.png', dpi=150)
        plt.close()

    def plot_color_histograms_per_camera(self, metrics_list: List[ImageMetrics]):
        """Plot color histograms grouped by camera prefix."""
        cameras = {}
        for m in metrics_list:
            prefix = m.camera_prefix
            if prefix not in cameras:
                cameras[prefix] = []
            cameras[prefix].append(m)

        n_cameras = len(cameras)
        fig, axes = plt.subplots(n_cameras, 1, figsize=(12, 3 * n_cameras))
        if n_cameras == 1:
            axes = [axes]

        for ax, (prefix, cam_metrics) in zip(axes, sorted(cameras.items())):
            rgb_r = [m.rgb_means[0] for m in cam_metrics]
            rgb_g = [m.rgb_means[1] for m in cam_metrics]
            rgb_b = [m.rgb_means[2] for m in cam_metrics]

            x = np.arange(len(cam_metrics))
            width = 0.25
            ax.bar(x - width, rgb_r, width, label='R', color='red', alpha=0.7)
            ax.bar(x, rgb_g, width, label='G', color='green', alpha=0.7)
            ax.bar(x + width, rgb_b, width, label='B', color='blue', alpha=0.7)
            ax.set_ylabel('Mean Value')
            ax.set_title(f'Camera: {prefix}')
            ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'color_histograms_per_camera.png', dpi=150)
        plt.close()

    def plot_road_coverage_estimate(self, metrics_list: List[ImageMetrics]):
        """Plot road coverage estimates."""
        fig, ax = plt.subplots(figsize=(10, 6))
        coverage = [m.road_coverage_ratio * 100 for m in metrics_list]
        ax.hist(coverage, bins=20, edgecolor='black', alpha=0.7, color='gray')
        ax.axvline(np.mean(coverage), color='red', linestyle='--', label=f'Mean: {np.mean(coverage):.1f}%')
        ax.set_xlabel('Road Coverage (%)')
        ax.set_ylabel('Count')
        ax.set_title('Estimated Road Coverage Distribution')
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'road_coverage_estimate.png', dpi=150)
        plt.close()

    def plot_condition_classification(self, metrics_list: List[ImageMetrics]):
        """Plot condition classifications."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Time of day
        tod_counts = {}
        for m in metrics_list:
            tod = m.time_of_day
            tod_counts[tod] = tod_counts.get(tod, 0) + 1

        ax1.bar(tod_counts.keys(), tod_counts.values(), color=['navy', 'orange', 'gold'], edgecolor='black')
        ax1.set_xlabel('Time of Day')
        ax1.set_ylabel('Count')
        ax1.set_title('Time of Day Classification')

        # Weather
        weather_counts = {}
        for m in metrics_list:
            w = m.weather_condition
            weather_counts[w] = weather_counts.get(w, 0) + 1

        colors = {'clear': 'skyblue', 'overcast': 'gray', 'foggy': 'white', 'variable': 'lightgreen'}
        bar_colors = [colors.get(k, 'lightblue') for k in weather_counts.keys()]
        ax2.bar(weather_counts.keys(), weather_counts.values(), color=bar_colors, edgecolor='black')
        ax2.set_xlabel('Weather Condition')
        ax2.set_ylabel('Count')
        ax2.set_title('Weather Condition Classification')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'condition_classification.png', dpi=150)
        plt.close()

    def plot_outlier_scatter(self, metrics_list: List[ImageMetrics]):
        """Plot outlier scatter plot (blur vs brightness)."""
        fig, ax = plt.subplots(figsize=(10, 8))

        normal = [(m.blur_score, m.brightness) for m in metrics_list if not m.is_outlier]
        outliers = [(m.blur_score, m.brightness) for m in metrics_list if m.is_outlier]

        if normal:
            ax.scatter(*zip(*normal), c='blue', alpha=0.6, label='Normal', s=50)
        if outliers:
            ax.scatter(*zip(*outliers), c='red', alpha=0.8, label='Outlier', s=80, marker='x')

        ax.set_xlabel('Blur Score')
        ax.set_ylabel('Brightness')
        ax.set_title('Outlier Detection (Blur vs Brightness)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'outlier_scatter.png', dpi=150)
        plt.close()

    def plot_readiness_scores(self, metrics_list: List[ImageMetrics]):
        """Plot readiness score distribution."""
        fig, ax = plt.subplots(figsize=(10, 6))
        scores = [m.readiness_score for m in metrics_list]
        ax.hist(scores, bins=20, edgecolor='black', alpha=0.7, color='green')
        ax.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.1f}')
        ax.axvline(70, color='orange', linestyle=':', label='Threshold (70)')
        ax.set_xlabel('Readiness Score')
        ax.set_ylabel('Count')
        ax.set_title('Model Training Readiness Scores')
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'readiness_scores.png', dpi=150)
        plt.close()

    def plot_sample_grid(self, image_paths: List[Path], metrics_list: List[ImageMetrics], n_samples: int = 16):
        """Plot a grid of sample images with readiness scores."""
        # Sort by readiness score and sample evenly
        paired = list(zip(image_paths, metrics_list))
        paired.sort(key=lambda x: x[1].readiness_score)

        n = min(n_samples, len(paired))
        indices = np.linspace(0, len(paired) - 1, n, dtype=int)
        samples = [paired[i] for i in indices]

        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten() if n > 1 else [axes]

        for ax, (path, metrics) in zip(axes, samples):
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(f'{path.name}\nScore: {metrics.readiness_score:.1f}', fontsize=8)
            ax.axis('off')

        # Hide unused axes
        for ax in axes[len(samples):]:
            ax.axis('off')

        plt.suptitle('Sample Images by Readiness Score (Low to High)', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sample_grid.png', dpi=150)
        plt.close()

    def generate_all(self, image_paths: List[Path], metrics_list: List[ImageMetrics]):
        """Generate all visualizations."""
        self.plot_blur_distribution(metrics_list)
        self.plot_brightness_histogram(metrics_list)
        self.plot_color_histograms_per_camera(metrics_list)
        self.plot_road_coverage_estimate(metrics_list)
        self.plot_condition_classification(metrics_list)
        self.plot_outlier_scatter(metrics_list)
        self.plot_readiness_scores(metrics_list)
        self.plot_sample_grid(image_paths, metrics_list)


class ReportGenerator:
    """Generates JSON and Markdown reports."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_json(self, metrics_list: List[ImageMetrics]) -> Path:
        """Generate statistics.json with all metrics."""
        data = {
            "image_count": len(metrics_list),
            "images": [],
            "aggregates": {}
        }

        # Per-image metrics
        for m in metrics_list:
            img_data = asdict(m)
            data["images"].append(img_data)

        # Aggregate statistics
        blur_scores = [m.blur_score for m in metrics_list]
        brightness_vals = [m.brightness for m in metrics_list]
        contrast_vals = [m.contrast for m in metrics_list]
        road_coverage = [m.road_coverage_ratio for m in metrics_list]
        readiness = [m.readiness_score for m in metrics_list]

        data["aggregates"] = {
            "blur_score": {"mean": np.mean(blur_scores), "std": np.std(blur_scores), "min": np.min(blur_scores), "max": np.max(blur_scores)},
            "brightness": {"mean": np.mean(brightness_vals), "std": np.std(brightness_vals), "min": np.min(brightness_vals), "max": np.max(brightness_vals)},
            "contrast": {"mean": np.mean(contrast_vals), "std": np.std(contrast_vals), "min": np.min(contrast_vals), "max": np.max(contrast_vals)},
            "road_coverage_ratio": {"mean": np.mean(road_coverage), "std": np.std(road_coverage), "min": np.min(road_coverage), "max": np.max(road_coverage)},
            "readiness_score": {"mean": np.mean(readiness), "std": np.std(readiness), "min": np.min(readiness), "max": np.max(readiness)},
            "outlier_count": sum(1 for m in metrics_list if m.is_outlier),
            "time_of_day_distribution": {},
            "weather_distribution": {}
        }

        # Distributions
        for m in metrics_list:
            tod = m.time_of_day
            data["aggregates"]["time_of_day_distribution"][tod] = data["aggregates"]["time_of_day_distribution"].get(tod, 0) + 1
            w = m.weather_condition
            data["aggregates"]["weather_distribution"][w] = data["aggregates"]["weather_distribution"].get(w, 0) + 1

        output_path = self.output_dir / "statistics.json"
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

        return output_path

    def generate_markdown(self, metrics_list: List[ImageMetrics]) -> Path:
        """Generate quality_report.md."""
        n = len(metrics_list)
        blur_scores = [m.blur_score for m in metrics_list]
        brightness_vals = [m.brightness for m in metrics_list]
        readiness = [m.readiness_score for m in metrics_list]
        outliers = [m for m in metrics_list if m.is_outlier]
        ready_count = sum(1 for m in metrics_list if m.readiness_score >= 70)

        report = f"""# CCTV Road Image EDA Report

## Overview
- **Total Images Analyzed**: {n}
- **Ready for Training (score >= 70)**: {ready_count} ({ready_count/n*100:.1f}%)
- **Outliers Detected**: {len(outliers)} ({len(outliers)/n*100:.1f}%)

## Quality Metrics Summary

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Blur Score | {np.mean(blur_scores):.2f} | {np.std(blur_scores):.2f} | {np.min(blur_scores):.2f} | {np.max(blur_scores):.2f} |
| Brightness | {np.mean(brightness_vals):.2f} | {np.std(brightness_vals):.2f} | {np.min(brightness_vals):.2f} | {np.max(brightness_vals):.2f} |
| Readiness | {np.mean(readiness):.2f} | {np.std(readiness):.2f} | {np.min(readiness):.2f} | {np.max(readiness):.2f} |

## Time of Day Distribution
"""
        tod_counts = {}
        for m in metrics_list:
            tod_counts[m.time_of_day] = tod_counts.get(m.time_of_day, 0) + 1
        for tod, count in sorted(tod_counts.items()):
            report += f"- **{tod.capitalize()}**: {count} images ({count/n*100:.1f}%)\n"

        report += "\n## Weather Condition Distribution\n"
        weather_counts = {}
        for m in metrics_list:
            weather_counts[m.weather_condition] = weather_counts.get(m.weather_condition, 0) + 1
        for w, count in sorted(weather_counts.items()):
            report += f"- **{w.capitalize()}**: {count} images ({count/n*100:.1f}%)\n"

        if outliers:
            report += "\n## Outlier Images\n"
            report += "| Filename | Reasons | Readiness Score |\n"
            report += "|----------|---------|----------------|\n"
            for m in outliers[:10]:  # Show top 10
                reasons = ", ".join(m.outlier_reasons) if m.outlier_reasons else "N/A"
                report += f"| {m.filename} | {reasons} | {m.readiness_score:.1f} |\n"
            if len(outliers) > 10:
                report += f"\n*... and {len(outliers) - 10} more outliers*\n"

        report += "\n## Recommendations\n"
        if np.mean(blur_scores) < 100:
            report += "- **Low Average Blur Score**: Consider checking camera focus or image compression.\n"
        if len(outliers) > n * 0.15:
            report += "- **High Outlier Rate**: Review outlier images for data quality issues.\n"
        if ready_count < n * 0.7:
            report += "- **Low Readiness Rate**: Consider filtering low-quality images before training.\n"
        if np.mean(readiness) >= 70:
            report += "- **Good Overall Quality**: Dataset appears suitable for model training.\n"

        report += "\n## Visualizations\n"
        report += "See the `visualizations/` directory for detailed plots:\n"
        viz_files = [
            "blur_distribution.png", "brightness_histogram.png", "color_histograms_per_camera.png",
            "road_coverage_estimate.png", "condition_classification.png", "outlier_scatter.png",
            "readiness_scores.png", "sample_grid.png"
        ]
        for f in viz_files:
            report += f"- `{f}`\n"

        output_path = self.output_dir / "quality_report.md"
        with open(output_path, "w") as f:
            f.write(report)

        return output_path


def analyze_image(image_path: Path, quality: QualityAnalyzer, color: ColorAnalyzer,
                  road: RoadCoverageAnalyzer, condition: ConditionClassifier) -> ImageMetrics:
    """Analyze a single image and return metrics."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    metrics = ImageMetrics()
    metrics.filename = image_path.name

    # Extract camera prefix (e.g., "01CT" from "01CT000000123.jpg")
    name = image_path.stem
    metrics.camera_prefix = name[:4] if len(name) >= 4 else name

    # Quality analysis
    quality_result = quality.analyze(image)
    metrics.blur_score = quality_result["blur_score"]
    metrics.contrast = quality_result["contrast"]
    metrics.brightness = quality_result["brightness"]
    metrics.noise_estimate = quality_result["noise_estimate"]

    # Color analysis
    color_result = color.analyze(image)
    metrics.rgb_means = color_result["rgb_means"]
    metrics.hsv_means = color_result["hsv_means"]
    metrics.dominant_colors = color_result["dominant_colors"]

    # Road coverage
    road_result = road.analyze(image)
    metrics.road_coverage_ratio = road_result["road_coverage_ratio"]

    # Condition classification
    cond_result = condition.analyze(quality_result, color_result)
    metrics.time_of_day = cond_result["time_of_day"]
    metrics.weather_condition = cond_result["weather_condition"]

    return metrics


def main():
    parser = argparse.ArgumentParser(description="CCTV Road Image EDA")
    parser.add_argument("--input-dir", type=str, default="output_dir3", help="Input directory with images")
    parser.add_argument("--output-dir", type=str, default="results/eda_output_dir3", help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent.parent
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = script_dir / input_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir

    viz_dir = output_dir / "visualizations"

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")

    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ])

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images to analyze")

    # Initialize analyzers
    quality = QualityAnalyzer()
    color = ColorAnalyzer()
    road = RoadCoverageAnalyzer()
    condition = ConditionClassifier()
    outlier = OutlierDetector()
    scorer = ReadinessScorer()

    # Analyze all images
    metrics_list: List[ImageMetrics] = []
    for path in tqdm(image_paths, desc="Analyzing images"):
        try:
            metrics = analyze_image(path, quality, color, road, condition)
            metrics_list.append(metrics)
        except Exception as e:
            if args.verbose:
                print(f"Error processing {path}: {e}")

    if not metrics_list:
        print("No images could be analyzed")
        return

    # Outlier detection
    print("Detecting outliers...")
    outlier_results = outlier.fit_predict(metrics_list)
    for metrics, (is_outlier, reasons) in zip(metrics_list, outlier_results):
        metrics.is_outlier = is_outlier
        metrics.outlier_reasons = reasons

    # Compute readiness scores
    print("Computing readiness scores...")
    for metrics in metrics_list:
        metrics.readiness_score = scorer.compute_score(metrics)

    # Generate reports
    print("Generating reports...")
    report_gen = ReportGenerator(output_dir)
    json_path = report_gen.generate_json(metrics_list)
    md_path = report_gen.generate_markdown(metrics_list)

    # Generate visualizations
    print("Generating visualizations...")
    viz_gen = VisualizationGenerator(viz_dir)
    viz_gen.generate_all(image_paths, metrics_list)

    # Summary
    ready_count = sum(1 for m in metrics_list if m.readiness_score >= 70)
    outlier_count = sum(1 for m in metrics_list if m.is_outlier)

    print("\n" + "=" * 50)
    print("EDA COMPLETE")
    print("=" * 50)
    print(f"Images analyzed: {len(metrics_list)}")
    print(f"Ready for training (score >= 70): {ready_count} ({ready_count/len(metrics_list)*100:.1f}%)")
    print(f"Outliers detected: {outlier_count} ({outlier_count/len(metrics_list)*100:.1f}%)")
    print(f"Mean readiness score: {np.mean([m.readiness_score for m in metrics_list]):.1f}")
    print(f"\nOutputs:")
    print(f"  - {json_path}")
    print(f"  - {md_path}")
    print(f"  - {viz_dir}/")


if __name__ == "__main__":
    main()
