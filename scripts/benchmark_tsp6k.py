#!/usr/bin/env python3
"""
Road Segmentation Model Benchmark on TSP6K Dataset (CCTV Traffic Scenes)
Evaluates multiple models for road-related semantic segmentation
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

# TSP6K Class definitions (21 classes)
TSP6K_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'railing',
    'vegetation', 'terrain', 'sky', 'person', 'rider',
    'car', 'truck', 'bus', 'motorcycle', 'bicycle',
    'indication_line', 'lane_line', 'crosswalk', 'pole',
    'traffic_light', 'traffic_sign'
]

# Road-related class indices in TSP6K
ROAD_RELATED_CLASSES = {
    0: 'road',
    1: 'sidewalk',
    15: 'indication_line',
    16: 'lane_line',
    17: 'crosswalk'
}

# Cityscapes to TSP6K mapping (for Cityscapes-trained models)
CITYSCAPES_TO_TSP6K = {
    0: 0,   # road -> road
    1: 1,   # sidewalk -> sidewalk
    # Other classes don't map directly for road segmentation
}

# ADE20K to TSP6K mapping (for ADE20K-trained models)
ADE20K_TO_TSP6K = {
    6: 0,   # road -> road
    11: 1,  # sidewalk -> sidewalk
}


def get_device():
    """Get available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tsp6k_dataset(data_dir: str, split: str = "val") -> List[Tuple[str, str]]:
    """Load TSP6K dataset image and ground truth pairs"""
    release_dir = Path(data_dir) / "release"
    image_dir = release_dir / "image"
    label_dir = release_dir / "label_trainval"
    split_file = release_dir / "split" / f"{split}.txt"

    pairs = []
    with open(split_file, 'r') as f:
        for line in f:
            img_name = line.strip()
            img_path = image_dir / f"{img_name}.jpg"
            # Use semantic segmentation label (_sem.png)
            label_path = label_dir / f"{img_name}_sem.png"

            if img_path.exists() and label_path.exists():
                pairs.append((str(img_path), str(label_path)))

    return pairs


def parse_tsp6k_gt(gt_path: str, target_classes: List[int] = None) -> np.ndarray:
    """Parse TSP6K ground truth to binary mask for target classes

    Args:
        gt_path: Path to semantic label PNG
        target_classes: List of class IDs to include in mask (default: road-related)
    """
    if target_classes is None:
        target_classes = list(ROAD_RELATED_CLASSES.keys())

    gt_img = np.array(Image.open(gt_path))

    # Create binary mask for target classes
    mask = np.zeros_like(gt_img, dtype=np.uint8)
    for cls_id in target_classes:
        mask = mask | (gt_img == cls_id)

    return mask.astype(np.uint8)


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """Compute segmentation metrics: IoU, Precision, Recall, F1"""
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()

    iou = intersection / (union + 1e-8)

    tp = intersection
    fp = np.logical_and(pred_bool, ~gt_bool).sum()
    fn = np.logical_and(~pred_bool, gt_bool).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    correct = np.sum(pred_bool == gt_bool)
    total = pred_bool.size
    pixel_acc = correct / total

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pixel_acc": float(pixel_acc)
    }


def compute_per_class_metrics(pred: np.ndarray, gt: np.ndarray, num_classes: int = 21) -> Dict[str, float]:
    """Compute per-class IoU for all classes"""
    per_class_iou = {}

    for cls_id in range(num_classes):
        pred_mask = (pred == cls_id)
        gt_mask = (gt == cls_id)

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()

        if union > 0:
            iou = intersection / union
            per_class_iou[TSP6K_CLASSES[cls_id]] = float(iou)

    return per_class_iou


class SegFormerModel:
    """SegFormer model wrapper"""
    def __init__(self, variant: str = "b2", device: torch.device = None):
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.device = device or get_device()
        self.variant = variant

        model_id = f"nvidia/segformer-{variant}-finetuned-cityscapes-1024-1024"
        print(f"Loading SegFormer-{variant.upper()}...")

        self.processor = SegformerImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        self.name = f"SegFormer-{variant.upper()}"
        # Cityscapes road class IDs that map to TSP6K road-related classes
        self.road_class_ids = [0, 1]  # road, sidewalk in Cityscapes

    def predict(self, image: Image.Image) -> np.ndarray:
        """Predict segmentation mask"""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        upsampled = torch.nn.functional.interpolate(
            logits, size=image.size[::-1], mode="bilinear", align_corners=False
        )
        pred = upsampled.argmax(dim=1).squeeze().cpu().numpy()
        return pred

    def predict_road_mask(self, image: Image.Image) -> np.ndarray:
        """Predict binary road mask"""
        pred = self.predict(image)
        road_mask = np.isin(pred, self.road_class_ids).astype(np.uint8)
        return road_mask


class DPTModel:
    """DPT-Large model wrapper"""
    def __init__(self, device: torch.device = None):
        from transformers import DPTForSemanticSegmentation, DPTImageProcessor

        self.device = device or get_device()

        model_id = "Intel/dpt-large-ade"
        print("Loading DPT-Large...")

        self.processor = DPTImageProcessor.from_pretrained(model_id)
        self.model = DPTForSemanticSegmentation.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        self.name = "DPT-Large"
        # ADE20K road class IDs
        self.road_class_ids = [6, 11]  # road, sidewalk in ADE20K

    def predict(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        upsampled = torch.nn.functional.interpolate(
            logits, size=image.size[::-1], mode="bilinear", align_corners=False
        )
        pred = upsampled.argmax(dim=1).squeeze().cpu().numpy()
        return pred

    def predict_road_mask(self, image: Image.Image) -> np.ndarray:
        pred = self.predict(image)
        road_mask = np.isin(pred, self.road_class_ids).astype(np.uint8)
        return road_mask


def benchmark_model(model, data_pairs: List[Tuple[str, str]],
                   target_classes: List[int] = None,
                   max_images: int = None) -> Dict:
    """Benchmark a model on the TSP6K dataset"""
    if target_classes is None:
        # Default: road + sidewalk only (to match model capabilities)
        target_classes = [0, 1]  # road, sidewalk in TSP6K

    results = {
        "model_name": model.name,
        "device": str(model.device),
        "target_classes": [TSP6K_CLASSES[i] for i in target_classes],
        "metrics": [],
        "inference_times": [],
    }

    pairs = data_pairs[:max_images] if max_images else data_pairs

    for img_path, gt_path in tqdm(pairs, desc=f"Benchmarking {model.name}"):
        image = Image.open(img_path).convert("RGB")
        gt_mask = parse_tsp6k_gt(gt_path, target_classes)

        # Time inference
        start_time = time.time()
        pred_mask = model.predict_road_mask(image)
        inference_time = time.time() - start_time

        # Compute metrics
        metrics = compute_metrics(pred_mask, gt_mask)

        results["metrics"].append(metrics)
        results["inference_times"].append(inference_time)

    # Aggregate metrics
    avg_metrics = {}
    for key in results["metrics"][0].keys():
        values = [m[key] for m in results["metrics"]]
        avg_metrics[f"avg_{key}"] = float(np.mean(values))
        avg_metrics[f"std_{key}"] = float(np.std(values))

    results["aggregate"] = avg_metrics
    results["avg_inference_time"] = float(np.mean(results["inference_times"]))
    results["total_images"] = len(pairs)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark road segmentation on TSP6K")
    parser.add_argument("--data-dir", default="data/benchmark/tsp6k", help="TSP6K data directory")
    parser.add_argument("--output-dir", default="results/benchmark_tsp6k", help="Output directory")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to process")
    parser.add_argument("--models", nargs="+", default=["segformer-b2", "segformer-b5", "dpt"],
                        choices=["segformer-b2", "segformer-b5", "dpt"],
                        help="Models to benchmark")
    args = parser.parse_args()

    # Load dataset
    print(f"Loading TSP6K dataset ({args.split} split)...")
    data_pairs = load_tsp6k_dataset(args.data_dir, args.split)
    print(f"Found {len(data_pairs)} image-GT pairs")

    if len(data_pairs) == 0:
        print("Error: No data pairs found!")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    # Benchmark each model
    all_results = []

    model_classes = {
        "segformer-b2": lambda: SegFormerModel("b2", device),
        "segformer-b5": lambda: SegFormerModel("b5", device),
        "dpt": lambda: DPTModel(device),
    }

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Benchmarking {model_name} on TSP6K CCTV dataset")
        print('='*60)

        try:
            model = model_classes[model_name]()
            results = benchmark_model(model, data_pairs, max_images=args.max_images)
            all_results.append(results)

            # Print summary
            print(f"\n{model.name} Results on TSP6K:")
            print(f"  Target Classes: {results['target_classes']}")
            print(f"  IoU:        {results['aggregate']['avg_iou']:.4f} ± {results['aggregate']['std_iou']:.4f}")
            print(f"  Precision:  {results['aggregate']['avg_precision']:.4f}")
            print(f"  Recall:     {results['aggregate']['avg_recall']:.4f}")
            print(f"  F1 Score:   {results['aggregate']['avg_f1']:.4f}")
            print(f"  Pixel Acc:  {results['aggregate']['avg_pixel_acc']:.4f}")
            print(f"  Avg Time:   {results['avg_inference_time']:.3f}s/image")

            # Clean up GPU memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    results_file = output_dir / "tsp6k_benchmark_results.json"
    with open(results_file, "w") as f:
        # Remove per-image metrics to save space
        for r in all_results:
            r.pop("metrics", None)
            r.pop("inference_times", None)
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("TSP6K BENCHMARK SUMMARY (CCTV Traffic Scenes)")
    print('='*60)
    print(f"{'Model':<20} {'IoU':>10} {'F1':>10} {'Pixel Acc':>12} {'Time (s)':>10}")
    print('-'*62)
    for r in all_results:
        print(f"{r['model_name']:<20} {r['aggregate']['avg_iou']:>10.4f} {r['aggregate']['avg_f1']:>10.4f} {r['aggregate']['avg_pixel_acc']:>12.4f} {r['avg_inference_time']:>10.3f}")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
