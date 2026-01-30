#!/usr/bin/env python3
"""Test script to verify dataset loading after download.

This script demonstrates:
1. Loading datasets with RoadTopologyDataset
2. Combining multiple datasets
3. Basic statistics and visualization
4. DataLoader usage

Usage:
    python scripts/test_dataset_loading.py --dataset kitti
    python scripts/test_dataset_loading.py --dataset bdd100k
    python scripts/test_dataset_loading.py --all
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from rich.console import Console
from rich.table import Table
from torch.utils.data import ConcatDataset, DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from road_topology.segmentation.dataset import RoadTopologyDataset

app = typer.Typer(help="Test dataset loading")
console = Console()


def visualize_batch(images: torch.Tensor, masks: torch.Tensor, num_samples: int = 4) -> None:
    """Visualize a batch of images and masks.

    Args:
        images: Batch of images (B, C, H, W).
        masks: Batch of masks (B, H, W).
        num_samples: Number of samples to show.
    """
    num_samples = min(num_samples, len(images))

    # Class colors
    colors = np.array([
        [0, 0, 0],         # Background - black
        [128, 128, 128],   # Road - gray
        [255, 255, 255],   # Lane - white
        [255, 255, 0],     # Crosswalk - yellow
        [0, 255, 0],       # Sidewalk - green
    ])

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx in range(num_samples):
        # Convert image to numpy
        img = images[idx].permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)

        # Convert mask to numpy
        mask = masks[idx].numpy().astype(np.uint8)

        # Create colored mask
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id in range(5):
            colored_mask[mask == class_id] = colors[class_id]

        # Plot
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f"Image {idx + 1}")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(colored_mask)
        axes[idx, 1].set_title(f"Mask {idx + 1}")
        axes[idx, 1].axis("off")

    plt.tight_layout()
    plt.savefig("dataset_samples.png", dpi=150, bbox_inches="tight")
    console.print("[green]Visualization saved to: dataset_samples.png[/green]")

    try:
        plt.show()
    except:
        pass


def compute_class_distribution(dataset: RoadTopologyDataset) -> dict[str, dict[str, float]]:
    """Compute class distribution across dataset.

    Args:
        dataset: Dataset to analyze.

    Returns:
        Dictionary with class statistics.
    """
    console.print(f"[yellow]Computing class distribution for {len(dataset)} images...[/yellow]")

    class_counts = np.zeros(5, dtype=np.int64)
    total_pixels = 0

    # Sample subset if dataset is large
    num_samples = min(100, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    for idx in indices:
        sample = dataset[idx]
        mask = sample["mask"].numpy()

        unique, counts = np.unique(mask, return_counts=True)
        for class_id, count in zip(unique, counts):
            if class_id < 5:
                class_counts[class_id] += count
        total_pixels += mask.size

    # Compute percentages
    class_names = ["Background", "Road", "Lane", "Crosswalk", "Sidewalk"]
    stats = {}

    for class_id, name in enumerate(class_names):
        count = int(class_counts[class_id])
        percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
        stats[name] = {
            "count": count,
            "percentage": percentage,
        }

    return stats


@app.command()
def test_dataset(
    dataset_name: str = typer.Argument(..., help="Dataset name (kitti, bdd100k, cityscapes)"),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", "-d", help="Data directory"),
    split: str = typer.Option("train", "--split", "-s", help="Dataset split"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Batch size"),
    visualize: bool = typer.Option(True, "--visualize/--no-visualize", help="Visualize samples"),
) -> None:
    """Test loading a single dataset.

    Args:
        dataset_name: Name of dataset to test.
        data_dir: Base data directory.
        split: Dataset split to load.
        batch_size: Batch size for DataLoader.
        visualize: Whether to visualize samples.
    """
    console.print(f"[bold blue]Testing {dataset_name.upper()} Dataset[/bold blue]")

    dataset_path = data_dir / dataset_name
    if not dataset_path.exists():
        console.print(f"[red]Dataset not found at: {dataset_path}[/red]")
        console.print("Run download_datasets.py first!")
        return

    # Load dataset
    try:
        dataset = RoadTopologyDataset(root=dataset_path, split=split)
    except Exception as e:
        console.print(f"[red]Failed to load dataset: {e}[/red]")
        return

    console.print(f"[green]Loaded {len(dataset)} images from {split} split[/green]")

    if len(dataset) == 0:
        console.print("[red]Dataset is empty![/red]")
        return

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Get one batch
    try:
        batch = next(iter(loader))
    except Exception as e:
        console.print(f"[red]Failed to load batch: {e}[/red]")
        return

    images = batch["image"]
    masks = batch["mask"]

    console.print(f"\n[cyan]Batch Statistics:[/cyan]")
    console.print(f"  Images shape: {images.shape}")
    console.print(f"  Masks shape: {masks.shape}")
    console.print(f"  Image dtype: {images.dtype}")
    console.print(f"  Mask dtype: {masks.dtype}")
    console.print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    console.print(f"  Mask classes: {masks.unique().tolist()}")

    # Compute class distribution
    stats = compute_class_distribution(dataset)

    table = Table(title=f"{dataset_name.upper()} Class Distribution")
    table.add_column("Class", style="cyan")
    table.add_column("Pixels", style="magenta")
    table.add_column("Percentage", style="green")

    for class_name, class_stats in stats.items():
        count = class_stats["count"]
        percentage = class_stats["percentage"]
        table.add_row(class_name, f"{count:,}", f"{percentage:.2f}%")

    console.print("\n", table)

    # Visualize
    if visualize:
        console.print("\n[yellow]Creating visualization...[/yellow]")
        visualize_batch(images, masks)


@app.command()
def test_all(
    data_dir: Path = typer.Option(Path("data"), "--data-dir", "-d", help="Data directory"),
) -> None:
    """Test all available datasets.

    Args:
        data_dir: Base data directory.
    """
    console.print("[bold blue]Testing All Datasets[/bold blue]\n")

    datasets = ["kitti", "bdd100k", "cityscapes"]
    available = []

    for name in datasets:
        dataset_path = data_dir / name
        if dataset_path.exists():
            try:
                dataset = RoadTopologyDataset(root=dataset_path, split="train")
                if len(dataset) > 0:
                    available.append((name, dataset))
                    console.print(f"[green]✓ {name.upper()}: {len(dataset)} images[/green]")
                else:
                    console.print(f"[yellow]⚠ {name.upper()}: Empty[/yellow]")
            except Exception as e:
                console.print(f"[red]✗ {name.upper()}: Error - {e}[/red]")
        else:
            console.print(f"[red]✗ {name.upper()}: Not found[/red]")

    if not available:
        console.print("\n[red]No datasets available![/red]")
        console.print("Run download_datasets.py first!")
        return

    # Create combined dataset
    console.print(f"\n[cyan]Combining {len(available)} datasets...[/cyan]")
    combined = ConcatDataset([ds for _, ds in available])
    console.print(f"[green]Combined dataset: {len(combined)} images[/green]")

    # Test DataLoader
    loader = DataLoader(combined, batch_size=8, shuffle=True, num_workers=0)

    try:
        batch = next(iter(loader))
        console.print(f"\n[green]Successfully loaded batch of {len(batch['image'])} images[/green]")
        console.print(f"  Image shape: {batch['image'].shape}")
        console.print(f"  Mask shape: {batch['mask'].shape}")
    except Exception as e:
        console.print(f"\n[red]Failed to load batch: {e}[/red]")


@app.command()
def benchmark(
    dataset_name: str = typer.Argument("kitti", help="Dataset to benchmark"),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", "-d", help="Data directory"),
    batch_size: int = typer.Option(16, "--batch-size", "-b", help="Batch size"),
    num_workers: int = typer.Option(4, "--num-workers", "-w", help="DataLoader workers"),
) -> None:
    """Benchmark dataset loading speed.

    Args:
        dataset_name: Name of dataset to benchmark.
        data_dir: Base data directory.
        batch_size: Batch size for DataLoader.
        num_workers: Number of DataLoader workers.
    """
    import time

    console.print(f"[bold blue]Benchmarking {dataset_name.upper()} Dataset[/bold blue]")

    dataset_path = data_dir / dataset_name
    if not dataset_path.exists():
        console.print(f"[red]Dataset not found at: {dataset_path}[/red]")
        return

    dataset = RoadTopologyDataset(root=dataset_path, split="train")
    console.print(f"Dataset size: {len(dataset)} images")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Warmup
    console.print("[yellow]Warming up...[/yellow]")
    for _ in range(2):
        _ = next(iter(loader))

    # Benchmark
    console.print("[yellow]Benchmarking...[/yellow]")
    num_batches = min(50, len(loader))

    start_time = time.time()
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
    elapsed = time.time() - start_time

    images_per_sec = (num_batches * batch_size) / elapsed
    time_per_batch = elapsed / num_batches

    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Batch size", str(batch_size))
    table.add_row("Num workers", str(num_workers))
    table.add_row("Batches processed", str(num_batches))
    table.add_row("Total time", f"{elapsed:.2f}s")
    table.add_row("Time per batch", f"{time_per_batch*1000:.1f}ms")
    table.add_row("Images/second", f"{images_per_sec:.1f}")

    console.print("\n", table)


if __name__ == "__main__":
    app()
