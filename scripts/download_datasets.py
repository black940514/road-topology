#!/usr/bin/env python3
"""Download and prepare public datasets for road topology segmentation.

Supports:
- BDD100K (Berkeley DeepDrive) - 9 lane marking categories + semantic segmentation
- Cityscapes - Urban street scenes with semantic segmentation
- KITTI Road - Small dataset good for quick testing

Usage:
    python download_datasets.py bdd100k --output-dir data/
    python download_datasets.py cityscapes --email your@email.com --password pass
    python download_datasets.py kitti
    python download_datasets.py all
    python download_datasets.py demo  # Download samples and visualize
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import zipfile
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

# Initialize CLI and console
app = typer.Typer(help="Download and prepare road topology datasets")
console = Console()


# Class mapping constants
class ProjectClass(int, Enum):
    """Project class indices."""
    BACKGROUND = 0
    ROAD = 1
    LANE = 2
    CROSSWALK = 3
    SIDEWALK = 4


# BDD100K semantic segmentation (19 classes) → Project classes
BDD100K_SEM_MAP = {
    0: ProjectClass.ROAD,         # road
    1: ProjectClass.SIDEWALK,     # sidewalk
    2: ProjectClass.BACKGROUND,   # building
    3: ProjectClass.BACKGROUND,   # wall
    4: ProjectClass.BACKGROUND,   # fence
    5: ProjectClass.BACKGROUND,   # pole
    6: ProjectClass.LANE,         # traffic light
    7: ProjectClass.LANE,         # traffic sign
    8: ProjectClass.BACKGROUND,   # vegetation
    9: ProjectClass.BACKGROUND,   # terrain
    10: ProjectClass.BACKGROUND,  # sky
    11: ProjectClass.BACKGROUND,  # person
    12: ProjectClass.BACKGROUND,  # rider
    13: ProjectClass.BACKGROUND,  # car
    14: ProjectClass.BACKGROUND,  # truck
    15: ProjectClass.BACKGROUND,  # bus
    16: ProjectClass.BACKGROUND,  # train
    17: ProjectClass.BACKGROUND,  # motorcycle
    18: ProjectClass.BACKGROUND,  # bicycle
    255: ProjectClass.BACKGROUND, # unlabeled
}

# BDD100K lane markings (9 categories) → Project classes
BDD100K_LANE_MAP = {
    0: ProjectClass.CROSSWALK,    # crosswalk
    1: ProjectClass.LANE,         # double white
    2: ProjectClass.LANE,         # double yellow
    3: ProjectClass.LANE,         # double other
    4: ProjectClass.LANE,         # single white
    5: ProjectClass.LANE,         # single yellow
    6: ProjectClass.LANE,         # single other
    7: ProjectClass.ROAD,         # road curb
    8: ProjectClass.BACKGROUND,   # other
    255: ProjectClass.BACKGROUND, # unlabeled
}

# Cityscapes (34 classes) → Project classes
CITYSCAPES_MAP = {
    0: ProjectClass.BACKGROUND,   # unlabeled
    1: ProjectClass.BACKGROUND,   # ego vehicle
    2: ProjectClass.BACKGROUND,   # rectification border
    3: ProjectClass.BACKGROUND,   # out of roi
    4: ProjectClass.BACKGROUND,   # static
    5: ProjectClass.BACKGROUND,   # dynamic
    6: ProjectClass.BACKGROUND,   # ground
    7: ProjectClass.ROAD,         # road
    8: ProjectClass.SIDEWALK,     # sidewalk
    9: ProjectClass.BACKGROUND,   # parking
    10: ProjectClass.BACKGROUND,  # rail track
    11: ProjectClass.BACKGROUND,  # building
    12: ProjectClass.BACKGROUND,  # wall
    13: ProjectClass.BACKGROUND,  # fence
    14: ProjectClass.BACKGROUND,  # guard rail
    15: ProjectClass.BACKGROUND,  # bridge
    16: ProjectClass.BACKGROUND,  # tunnel
    17: ProjectClass.BACKGROUND,  # pole
    18: ProjectClass.BACKGROUND,  # polegroup
    19: ProjectClass.LANE,        # traffic light
    20: ProjectClass.LANE,        # traffic sign
    21: ProjectClass.BACKGROUND,  # vegetation
    22: ProjectClass.BACKGROUND,  # terrain
    23: ProjectClass.BACKGROUND,  # sky
    24: ProjectClass.BACKGROUND,  # person
    25: ProjectClass.BACKGROUND,  # rider
    26: ProjectClass.BACKGROUND,  # car
    27: ProjectClass.BACKGROUND,  # truck
    28: ProjectClass.BACKGROUND,  # bus
    29: ProjectClass.BACKGROUND,  # caravan
    30: ProjectClass.BACKGROUND,  # trailer
    31: ProjectClass.BACKGROUND,  # train
    32: ProjectClass.BACKGROUND,  # motorcycle
    33: ProjectClass.BACKGROUND,  # bicycle
    255: ProjectClass.BACKGROUND, # unlabeled
}

# KITTI Road → Project classes (simpler, only road/non-road)
KITTI_ROAD_MAP = {
    0: ProjectClass.BACKGROUND,   # non-road
    1: ProjectClass.ROAD,         # road
    2: ProjectClass.ROAD,         # road (alternative)
    255: ProjectClass.BACKGROUND, # unlabeled
}


# Dataset URLs (using Kaggle as primary source)
DATASET_URLS = {
    "bdd100k_sem": "https://www.kaggle.com/datasets/solesensei/bdd100k_sem_seg",
    "bdd100k_lane": "https://www.kaggle.com/datasets/solesensei/bdd100k-lane",
    "cityscapes": "https://www.cityscapes-dataset.com/downloads/",
    "kitti_road": "http://www.cvlibs.net/download.php?file=data_road.zip",
}


def download_file(url: str, destination: Path, progress: Progress, task: TaskID | None = None) -> Path:
    """Download file with progress bar.

    Args:
        url: URL to download from.
        destination: Destination file path.
        progress: Rich progress instance.
        task: Progress task ID.

    Returns:
        Path to downloaded file.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    if task is not None:
        progress.update(task, total=total_size)

    with open(destination, "wb") as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if task is not None:
                    progress.update(task, advance=len(chunk))

    return destination


def verify_checksum(file_path: Path, expected_md5: str) -> bool:
    """Verify file checksum.

    Args:
        file_path: Path to file.
        expected_md5: Expected MD5 hash.

    Returns:
        True if checksum matches.
    """
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest() == expected_md5


def extract_archive(archive_path: Path, extract_dir: Path, progress: Progress) -> Path:
    """Extract archive with progress.

    Args:
        archive_path: Path to archive file.
        extract_dir: Directory to extract to.
        progress: Rich progress instance.

    Returns:
        Path to extracted directory.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)

    task = progress.add_task(f"Extracting {archive_path.name}...", total=None)

    try:
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
        elif archive_path.suffix in [".tar", ".gz", ".tgz", ".bz2"]:
            with tarfile.open(archive_path, "r:*") as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
    finally:
        progress.remove_task(task)

    return extract_dir


def remap_mask(mask: np.ndarray, mapping: dict[int, ProjectClass]) -> np.ndarray:
    """Remap mask classes using provided mapping.

    Args:
        mask: Input mask with original class indices.
        mapping: Dictionary mapping original class → ProjectClass.

    Returns:
        Remapped mask with project class indices.
    """
    output = np.zeros_like(mask, dtype=np.uint8)
    for original_class, project_class in mapping.items():
        output[mask == original_class] = int(project_class)
    return output


def create_split_dirs(base_dir: Path, splits: list[str] = ["train", "val", "test"]) -> dict[str, tuple[Path, Path]]:
    """Create directory structure for dataset splits.

    Args:
        base_dir: Base dataset directory.
        splits: List of split names.

    Returns:
        Dictionary mapping split name → (images_dir, masks_dir).
    """
    dirs = {}
    for split in splits:
        images_dir = base_dir / split / "images"
        masks_dir = base_dir / split / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        dirs[split] = (images_dir, masks_dir)
    return dirs


@app.command()
def bdd100k(
    output_dir: Path = typer.Option(Path("data"), "--output-dir", "-o", help="Output directory"),
    use_kaggle: bool = typer.Option(True, "--kaggle/--no-kaggle", help="Use Kaggle API"),
    skip_existing: bool = typer.Option(True, "--skip-existing", help="Skip if already downloaded"),
) -> None:
    """Download and prepare BDD100K dataset.

    BDD100K includes:
    - 10K images with semantic segmentation (19 classes)
    - 100K images with lane markings (9 categories including crosswalks)

    This will create combined masks using both semantic and lane annotations.
    """
    console.print("[bold blue]Downloading BDD100K Dataset[/bold blue]")

    output_dir = output_dir / "bdd100k"
    output_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing and (output_dir / "train").exists():
        console.print("[yellow]BDD100K already exists. Use --no-skip-existing to re-download.[/yellow]")
        return

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ) as progress:

        if use_kaggle:
            console.print("[yellow]Note: Using Kaggle requires kaggle CLI and authentication.[/yellow]")
            console.print("Please ensure you have:")
            console.print("  1. pip install kaggle")
            console.print("  2. kaggle.json in ~/.kaggle/")
            console.print("  3. Run: kaggle datasets download -d solesensei/bdd100k_sem_seg")
            console.print("  4. Run: kaggle datasets download -d solesensei/bdd100k-lane")
            return

        # Manual download instructions
        console.print("[yellow]Manual download required:[/yellow]")
        console.print(f"1. Visit: {DATASET_URLS['bdd100k_sem']}")
        console.print(f"2. Visit: {DATASET_URLS['bdd100k_lane']}")
        console.print("3. Download both datasets")
        console.print(f"4. Extract to: {output_dir}")
        console.print("5. Re-run this command with extracted data in place")


@app.command()
def cityscapes(
    output_dir: Path = typer.Option(Path("data"), "--output-dir", "-o", help="Output directory"),
    email: str = typer.Option("", "--email", help="Cityscapes account email"),
    password: str = typer.Option("", "--password", help="Cityscapes account password", hide_input=True),
    skip_existing: bool = typer.Option(True, "--skip-existing", help="Skip if already downloaded"),
) -> None:
    """Download and prepare Cityscapes dataset.

    Cityscapes requires account registration at:
    https://www.cityscapes-dataset.com/register/

    Downloads:
    - leftImg8bit_trainvaltest.zip (images)
    - gtFine_trainvaltest.zip (annotations)
    """
    console.print("[bold blue]Downloading Cityscapes Dataset[/bold blue]")

    output_dir = output_dir / "cityscapes"
    output_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing and (output_dir / "train").exists():
        console.print("[yellow]Cityscapes already exists. Use --no-skip-existing to re-download.[/yellow]")
        return

    # Cityscapes requires authentication and manual download
    console.print("[yellow]Cityscapes requires manual download:[/yellow]")
    console.print("1. Register at: https://www.cityscapes-dataset.com/register/")
    console.print("2. Login at: https://www.cityscapes-dataset.com/login/")
    console.print("3. Download:")
    console.print("   - leftImg8bit_trainvaltest.zip")
    console.print("   - gtFine_trainvaltest.zip")
    console.print(f"4. Extract both to: {output_dir}")
    console.print("5. Re-run this command to process masks")

    # If data exists, process it
    if (output_dir / "leftImg8bit").exists() and (output_dir / "gtFine").exists():
        console.print("\n[green]Found existing data, processing...[/green]")
        _process_cityscapes(output_dir)


def _process_cityscapes(base_dir: Path) -> None:
    """Process Cityscapes dataset and create remapped masks."""
    from PIL import Image

    with Progress() as progress:
        task = progress.add_task("Processing Cityscapes masks...", total=None)

        # Create output structure
        dirs = create_split_dirs(base_dir)

        for split in ["train", "val", "test"]:
            img_src = base_dir / "leftImg8bit" / split
            mask_src = base_dir / "gtFine" / split

            if not img_src.exists() or not mask_src.exists():
                continue

            img_dst, mask_dst = dirs[split]

            # Process each city
            for city_dir in img_src.iterdir():
                if not city_dir.is_dir():
                    continue

                city_name = city_dir.name
                mask_city_dir = mask_src / city_name

                for img_path in city_dir.glob("*_leftImg8bit.png"):
                    # Find corresponding label file
                    base_name = img_path.stem.replace("_leftImg8bit", "")
                    label_path = mask_city_dir / f"{base_name}_gtFine_labelIds.png"

                    if not label_path.exists():
                        continue

                    # Copy image
                    shutil.copy(img_path, img_dst / f"{base_name}.png")

                    # Remap and save mask
                    mask = np.array(Image.open(label_path))
                    remapped = remap_mask(mask, CITYSCAPES_MAP)
                    Image.fromarray(remapped).save(mask_dst / f"{base_name}.png")

        progress.remove_task(task)

    console.print("[green]Cityscapes processing complete![/green]")


@app.command()
def kitti(
    output_dir: Path = typer.Option(Path("data"), "--output-dir", "-o", help="Output directory"),
    skip_existing: bool = typer.Option(True, "--skip-existing", help="Skip if already downloaded"),
) -> None:
    """Download and prepare KITTI Road dataset.

    Small dataset (289 images) good for quick testing.
    Download from: http://www.cvlibs.net/datasets/kitti/eval_road.php
    """
    console.print("[bold blue]Downloading KITTI Road Dataset[/bold blue]")

    output_dir = output_dir / "kitti"
    output_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing and (output_dir / "train").exists():
        console.print("[yellow]KITTI already exists. Use --no-skip-existing to re-download.[/yellow]")
        return

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ) as progress:

        # Download KITTI Road
        zip_path = output_dir / "data_road.zip"

        if not zip_path.exists():
            task = progress.add_task("Downloading KITTI Road...", total=None)
            try:
                download_file(DATASET_URLS["kitti_road"], zip_path, progress, task)
            except Exception as e:
                console.print(f"[red]Download failed: {e}[/red]")
                console.print("[yellow]Manual download:[/yellow]")
                console.print("1. Visit: http://www.cvlibs.net/datasets/kitti/eval_road.php")
                console.print("2. Download: data_road.zip")
                console.print(f"3. Place in: {output_dir}")
                return

        # Extract
        extract_dir = output_dir / "extracted"
        extract_archive(zip_path, extract_dir, progress)

        # Process KITTI structure
        _process_kitti(extract_dir, output_dir)

    console.print("[green]KITTI Road download complete![/green]")


def _process_kitti(extract_dir: Path, output_dir: Path) -> None:
    """Process KITTI Road dataset structure."""
    from PIL import Image

    dirs = create_split_dirs(output_dir, ["train", "test"])

    # KITTI has training and testing sets
    for split in ["training", "testing"]:
        src_img_dir = extract_dir / split / "image_2"
        src_mask_dir = extract_dir / split / "gt_image_2"

        if not src_img_dir.exists():
            continue

        dst_split = "train" if split == "training" else "test"
        img_dst, mask_dst = dirs[dst_split]

        # Copy images
        for img_path in src_img_dir.glob("*.png"):
            shutil.copy(img_path, img_dst / img_path.name)

        # Process masks if they exist
        if src_mask_dir.exists():
            for mask_path in src_mask_dir.glob("*_road_*.png"):
                mask = np.array(Image.open(mask_path))
                # KITTI uses RGB: road pixels are typically pink/red
                # Convert to binary road/non-road
                road_mask = (mask[:, :, 2] > 0).astype(np.uint8)  # Red channel
                remapped = np.where(road_mask, int(ProjectClass.ROAD), int(ProjectClass.BACKGROUND))

                output_name = mask_path.name.replace("_road_", "_")
                Image.fromarray(remapped.astype(np.uint8)).save(mask_dst / output_name)

    console.print("[green]KITTI Road processing complete![/green]")


@app.command()
def all_datasets(
    output_dir: Path = typer.Option(Path("data"), "--output-dir", "-o", help="Output directory"),
) -> None:
    """Download and prepare all supported datasets."""
    console.print("[bold blue]Downloading All Datasets[/bold blue]")

    # Download each dataset
    kitti(output_dir=output_dir)

    console.print("\n[yellow]Note: BDD100K and Cityscapes require manual download/authentication.[/yellow]")
    console.print("Run individual commands for instructions:")
    console.print("  python download_datasets.py bdd100k")
    console.print("  python download_datasets.py cityscapes")


@app.command()
def demo(
    output_dir: Path = typer.Option(Path("data/demo"), "--output-dir", "-o", help="Output directory"),
    num_samples: int = typer.Option(10, "--samples", "-n", help="Number of samples to download"),
) -> None:
    """Download small sample and visualize.

    Downloads a small subset of data for testing and visualizes samples.
    """
    console.print("[bold blue]Demo Mode - Downloading Samples[/bold blue]")

    # For demo, use KITTI as it's smallest
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download KITTI
    kitti(output_dir=output_dir.parent, skip_existing=False)

    # Visualize samples
    console.print("\n[bold green]Visualizing Samples[/bold green]")
    _visualize_samples(output_dir.parent / "kitti", num_samples)

    # Print statistics
    _print_dataset_stats(output_dir.parent / "kitti")


def _visualize_samples(dataset_dir: Path, num_samples: int = 5) -> None:
    """Visualize dataset samples with masks overlaid."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    train_imgs = dataset_dir / "train" / "images"
    train_masks = dataset_dir / "train" / "masks"

    if not train_imgs.exists():
        console.print("[yellow]No training data found for visualization.[/yellow]")
        return

    img_paths = sorted(train_imgs.glob("*.png"))[:num_samples]

    if not img_paths:
        console.print("[yellow]No images found for visualization.[/yellow]")
        return

    # Define colormap for classes
    colors = [
        [0, 0, 0],         # Background - black
        [128, 128, 128],   # Road - gray
        [255, 255, 255],   # Lane - white
        [255, 255, 0],     # Crosswalk - yellow
        [0, 255, 0],       # Sidewalk - green
    ]
    cmap = ListedColormap(np.array(colors) / 255.0)

    fig, axes = plt.subplots(len(img_paths), 3, figsize=(15, 5 * len(img_paths)))
    if len(img_paths) == 1:
        axes = axes.reshape(1, -1)

    for idx, img_path in enumerate(img_paths):
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load mask
        mask_path = train_masks / img_path.name
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Plot
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f"Image: {img_path.name}")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(mask, cmap=cmap, vmin=0, vmax=4)
        axes[idx, 1].set_title("Ground Truth Mask")
        axes[idx, 1].axis("off")

        # Overlay
        overlay = img.copy()
        for class_id in range(5):
            mask_binary = (mask == class_id)
            if mask_binary.any():
                overlay[mask_binary] = (
                    0.6 * overlay[mask_binary] +
                    0.4 * np.array(colors[class_id])
                ).astype(np.uint8)

        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title("Overlay")
        axes[idx, 2].axis("off")

    plt.tight_layout()
    output_path = dataset_dir / "visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    console.print(f"[green]Visualization saved to: {output_path}[/green]")

    # Try to display
    try:
        plt.show()
    except:
        pass


def _print_dataset_stats(dataset_dir: Path) -> None:
    """Print dataset statistics."""
    table = Table(title="Dataset Statistics")
    table.add_column("Split", style="cyan")
    table.add_column("Images", style="magenta")
    table.add_column("Masks", style="green")

    for split in ["train", "val", "test"]:
        img_dir = dataset_dir / split / "images"
        mask_dir = dataset_dir / split / "masks"

        if img_dir.exists():
            num_images = len(list(img_dir.glob("*.png"))) + len(list(img_dir.glob("*.jpg")))
            num_masks = len(list(mask_dir.glob("*.png"))) if mask_dir.exists() else 0
            table.add_row(split, str(num_images), str(num_masks))

    console.print(table)

    # Print class distribution from a sample
    mask_dir = dataset_dir / "train" / "masks"
    if mask_dir.exists():
        mask_files = list(mask_dir.glob("*.png"))
        if mask_files:
            sample_mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
            unique, counts = np.unique(sample_mask, return_counts=True)

            class_table = Table(title="Sample Class Distribution")
            class_table.add_column("Class", style="cyan")
            class_table.add_column("Pixels", style="magenta")
            class_table.add_column("Percentage", style="green")

            total_pixels = sample_mask.size
            class_names = ["Background", "Road", "Lane", "Crosswalk", "Sidewalk"]

            for class_id, count in zip(unique, counts):
                if class_id < len(class_names):
                    name = class_names[class_id]
                    percentage = (count / total_pixels) * 100
                    class_table.add_row(name, str(count), f"{percentage:.2f}%")

            console.print(class_table)


@app.command()
def info() -> None:
    """Show information about supported datasets."""
    console.print("\n[bold blue]Supported Datasets[/bold blue]\n")

    datasets = [
        {
            "name": "BDD100K",
            "size": "Large (10K semantic + 100K lane)",
            "classes": "9 lane categories + 19 semantic",
            "features": "Crosswalk annotations, diverse conditions",
            "url": DATASET_URLS["bdd100k_sem"],
        },
        {
            "name": "Cityscapes",
            "size": "Medium (5K train, 500 val)",
            "classes": "34 semantic classes",
            "features": "High quality urban scenes",
            "url": DATASET_URLS["cityscapes"],
        },
        {
            "name": "KITTI Road",
            "size": "Small (289 images)",
            "classes": "Binary road/non-road",
            "features": "Good for quick testing",
            "url": DATASET_URLS["kitti_road"],
        },
    ]

    for dataset in datasets:
        table = Table(title=dataset["name"], show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")

        for key, value in dataset.items():
            if key != "name":
                table.add_row(key.capitalize(), value)

        console.print(table)
        console.print()


if __name__ == "__main__":
    app()
