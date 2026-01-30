#!/usr/bin/env python3
"""
Road Segmentation Model Benchmark on KITTI Road Dataset
Evaluates multiple models: SegFormer-B2, B5, Mask2Former, OneFormer, DPT-Large
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Cityscapes class mapping for road-related classes
CITYSCAPES_ROAD_CLASSES = {
    0: "road",
    1: "sidewalk",
    # Other classes will be treated as background
}

# ADE20K class mapping for road-related classes
ADE20K_ROAD_CLASSES = {
    6: "road",      # road, route
    11: "sidewalk", # sidewalk, pavement
    # Other classes are background
}


def get_device():
    """Get available device (MPS for Mac, CUDA for GPU, else CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_kitti_dataset(data_dir: str) -> List[Tuple[str, str]]:
    """Load KITTI Road dataset image and ground truth pairs"""
    training_dir = Path(data_dir) / "data_road" / "training"
    image_dir = training_dir / "image_2"
    gt_dir = training_dir / "gt_image_2"

    pairs = []
    for img_file in sorted(image_dir.glob("*.png")):
        # Find corresponding ground truth (road annotation)
        base_name = img_file.stem  # e.g., "um_000000"

        # KITTI GT naming: um_000000 -> um_road_000000.png
        # Parse prefix and number
        parts = base_name.split("_")
        prefix = parts[0]  # um, umm, uu
        number = parts[1]  # 000000

        gt_road = gt_dir / f"{prefix}_road_{number}.png"

        if gt_road.exists():
            pairs.append((str(img_file), str(gt_road)))

    return pairs


def parse_kitti_gt(gt_path: str) -> np.ndarray:
    """Parse KITTI ground truth to binary road mask
    KITTI GT colors:
    - Red channel 255: Road area
    - Blue channel 255: Lane markings (if separate file)
    """
    gt_img = np.array(Image.open(gt_path))

    # Road is marked in red channel
    if len(gt_img.shape) == 3:
        road_mask = gt_img[:, :, 2] > 0  # Blue channel for road in KITTI format
    else:
        road_mask = gt_img > 0

    return road_mask.astype(np.uint8)


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """Compute segmentation metrics: IoU, Precision, Recall, F1"""
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()

    # IoU
    iou = intersection / (union + 1e-8)

    # Precision and Recall
    tp = intersection
    fp = np.logical_and(pred_bool, ~gt_bool).sum()
    fn = np.logical_and(~pred_bool, gt_bool).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Pixel accuracy
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
        self.road_class_id = 0  # Cityscapes road class

    def predict(self, image: Image.Image) -> np.ndarray:
        """Predict road mask from image"""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        upsampled = torch.nn.functional.interpolate(
            logits, size=image.size[::-1], mode="bilinear", align_corners=False
        )
        pred = upsampled.argmax(dim=1).squeeze().cpu().numpy()

        # Road class (0 in Cityscapes)
        road_mask = (pred == self.road_class_id).astype(np.uint8)
        return road_mask


class Mask2FormerModel:
    """Mask2Former model wrapper"""
    def __init__(self, device: torch.device = None):
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

        self.device = device or get_device()

        model_id = "facebook/mask2former-swin-base-cityscapes-semantic"
        print("Loading Mask2Former...")

        self.processor = Mask2FormerImageProcessor.from_pretrained(model_id)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        self.name = "Mask2Former"
        self.road_class_id = 0

    def predict(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        pred = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0].cpu().numpy()

        road_mask = (pred == self.road_class_id).astype(np.uint8)
        return road_mask


class OneFormerModel:
    """OneFormer model wrapper (uses CPU due to MPS issues)"""
    def __init__(self, device: torch.device = None):
        from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

        # Force CPU for OneFormer due to MPS float64 issues
        self.device = torch.device("cpu")

        model_id = "shi-labs/oneformer_ade20k_swin_large"
        print("Loading OneFormer (CPU)...")

        self.processor = OneFormerProcessor.from_pretrained(model_id)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        self.name = "OneFormer"
        self.road_class_id = 6  # ADE20K road class

    def predict(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, task_inputs=["semantic"], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        pred = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0].cpu().numpy()

        road_mask = (pred == self.road_class_id).astype(np.uint8)
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
        self.road_class_id = 6  # ADE20K road class

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

        road_mask = (pred == self.road_class_id).astype(np.uint8)
        return road_mask


def benchmark_model(model, data_pairs: List[Tuple[str, str]], max_images: int = None) -> Dict:
    """Benchmark a model on the dataset"""
    results = {
        "model_name": model.name,
        "device": str(model.device),
        "metrics": [],
        "inference_times": [],
    }

    pairs = data_pairs[:max_images] if max_images else data_pairs

    for img_path, gt_path in tqdm(pairs, desc=f"Benchmarking {model.name}"):
        image = Image.open(img_path).convert("RGB")
        gt_mask = parse_kitti_gt(gt_path)

        # Time inference
        start_time = time.time()
        pred_mask = model.predict(image)
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

    parser = argparse.ArgumentParser(description="Benchmark road segmentation models")
    parser.add_argument("--data-dir", default=".", help="Directory containing KITTI data")
    parser.add_argument("--output-dir", default="results/benchmark", help="Output directory")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to process")
    parser.add_argument("--models", nargs="+", default=["segformer-b2", "segformer-b5", "mask2former", "dpt"],
                        choices=["segformer-b2", "segformer-b5", "mask2former", "oneformer", "dpt"],
                        help="Models to benchmark")
    args = parser.parse_args()

    # Load dataset
    print("Loading KITTI Road dataset...")
    data_pairs = load_kitti_dataset(args.data_dir)
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
        "mask2former": lambda: Mask2FormerModel(device),
        "oneformer": lambda: OneFormerModel(device),
        "dpt": lambda: DPTModel(device),
    }

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Benchmarking {model_name}")
        print('='*60)

        try:
            model = model_classes[model_name]()
            results = benchmark_model(model, data_pairs, args.max_images)
            all_results.append(results)

            # Print summary
            print(f"\n{model.name} Results:")
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
    results_file = output_dir / "benchmark_results.json"
    with open(results_file, "w") as f:
        # Remove per-image metrics to save space
        for r in all_results:
            r.pop("metrics", None)
            r.pop("inference_times", None)
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print('='*60)
    print(f"{'Model':<20} {'IoU':>10} {'F1':>10} {'Pixel Acc':>12} {'Time (s)':>10}")
    print('-'*62)
    for r in all_results:
        print(f"{r['model_name']:<20} {r['aggregate']['avg_iou']:>10.4f} {r['aggregate']['avg_f1']:>10.4f} {r['aggregate']['avg_pixel_acc']:>12.4f} {r['avg_inference_time']:>10.3f}")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
