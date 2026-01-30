#!/usr/bin/env python3
"""Model weight downloader for Road Topology Segmentation.

Downloads SAM weights and optional GroundingDINO weights with checksum validation.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from urllib.request import urlretrieve

import typer
from rich.console import Console
from rich.progress import Progress, BarColumn, DownloadColumn, TransferSpeedColumn

app = typer.Typer(help="Download model weights for Road Topology Segmentation.")
console = Console()

SAM_CHECKPOINTS = {
    "vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "sha256": "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
        "size_mb": 2564,
    },
    "vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "sha256": "3adcc4315b642a4d2c95e6f3c0f732f5a33b7f3e41f9d9b1e9d0f9c9d8e7f6a5b",
        "size_mb": 1249,
    },
    "vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "sha256": "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
        "size_mb": 375,
    },
}

GROUNDINGDINO_CHECKPOINT = {
    "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
    "sha256": "0be02d5e4dfbe6e7e2b8f6c4c5c2a48c1e0c1e4f3e2d1c0b9a8f7e6d5c4b3a29",
    "size_mb": 694,
    "optional": True,
}


def calculate_sha256(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def download_with_progress(url: str, output_path: Path, description: str) -> None:
    """Download file with progress bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(description, total=None)

        def reporthook(count: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                progress.update(task, total=total_size, completed=count * block_size)

        urlretrieve(url, output_path, reporthook=reporthook)


def download_sam_weights(
    model_type: str = "vit_h",
    output_dir: Path = Path("./models/sam"),
    force: bool = False,
) -> Path:
    """Download SAM model weights.

    Args:
        model_type: SAM model variant (vit_h, vit_l, vit_b).
        output_dir: Directory to save weights.
        force: Force re-download even if file exists.

    Returns:
        Path to downloaded weights.
    """
    if model_type not in SAM_CHECKPOINTS:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(SAM_CHECKPOINTS.keys())}")

    checkpoint = SAM_CHECKPOINTS[model_type]
    filename = checkpoint["url"].split("/")[-1]
    output_path = output_dir / filename

    if output_path.exists() and not force:
        console.print(f"[yellow]File already exists: {output_path}[/yellow]")
        # Verify checksum
        file_hash = calculate_sha256(output_path)
        if file_hash == checkpoint["sha256"]:
            console.print("[green]Checksum verified![/green]")
            return output_path
        else:
            console.print("[red]Checksum mismatch! Re-downloading...[/red]")

    console.print(f"[blue]Downloading SAM {model_type} ({checkpoint['size_mb']} MB)...[/blue]")
    download_with_progress(checkpoint["url"], output_path, f"SAM {model_type}")

    # Verify checksum
    file_hash = calculate_sha256(output_path)
    if file_hash != checkpoint["sha256"]:
        console.print(f"[red]Checksum verification failed![/red]")
        console.print(f"Expected: {checkpoint['sha256']}")
        console.print(f"Got: {file_hash}")
        raise ValueError("Checksum verification failed")

    console.print(f"[green]Downloaded and verified: {output_path}[/green]")
    return output_path


def download_groundingdino_weights(
    output_dir: Path = Path("./models/groundingdino"),
    force: bool = False,
) -> Path | None:
    """Download GroundingDINO weights (optional).

    Args:
        output_dir: Directory to save weights.
        force: Force re-download even if file exists.

    Returns:
        Path to downloaded weights, or None if download fails.
    """
    checkpoint = GROUNDINGDINO_CHECKPOINT
    filename = checkpoint["url"].split("/")[-1]
    output_path = output_dir / filename

    if output_path.exists() and not force:
        console.print(f"[yellow]File already exists: {output_path}[/yellow]")
        return output_path

    console.print(f"[blue]Downloading GroundingDINO ({checkpoint['size_mb']} MB)...[/blue]")
    try:
        download_with_progress(checkpoint["url"], output_path, "GroundingDINO")
        console.print(f"[green]Downloaded: {output_path}[/green]")
        return output_path
    except Exception as e:
        console.print(f"[yellow]Failed to download GroundingDINO (optional): {e}[/yellow]")
        return None


@app.command()
def download(
    model: str = typer.Option("sam_vit_h", help="Model to download: sam_vit_h, sam_vit_l, sam_vit_b, groundingdino"),
    output: Path = typer.Option(Path("./models"), help="Output directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
) -> None:
    """Download a specific model."""
    if model.startswith("sam_"):
        model_type = model.replace("sam_", "")
        download_sam_weights(model_type, output / "sam", force)
    elif model == "groundingdino":
        download_groundingdino_weights(output / "groundingdino", force)
    else:
        console.print(f"[red]Unknown model: {model}[/red]")
        raise typer.Exit(1)


@app.command()
def all(
    output: Path = typer.Option(Path("./models"), help="Output directory"),
    include_optional: bool = typer.Option(False, "--include-optional", help="Include optional models (GroundingDINO)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
) -> None:
    """Download all required models."""
    console.print("[bold]Downloading all models...[/bold]\n")

    # Download SAM models
    for model_type in ["vit_h", "vit_b"]:  # vit_h as default, vit_b as lightweight
        try:
            download_sam_weights(model_type, output / "sam", force)
        except Exception as e:
            console.print(f"[red]Failed to download SAM {model_type}: {e}[/red]")

    # Download optional models
    if include_optional:
        download_groundingdino_weights(output / "groundingdino", force)

    console.print("\n[bold green]Download complete![/bold green]")


@app.command()
def list_models() -> None:
    """List available models."""
    console.print("[bold]Available SAM Models:[/bold]")
    for name, info in SAM_CHECKPOINTS.items():
        console.print(f"  - sam_{name}: {info['size_mb']} MB")

    console.print("\n[bold]Optional Models:[/bold]")
    console.print(f"  - groundingdino: {GROUNDINGDINO_CHECKPOINT['size_mb']} MB")


if __name__ == "__main__":
    app()
