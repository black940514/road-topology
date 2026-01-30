"""Evaluation CLI commands."""
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from road_topology.core.device import get_device

app = typer.Typer(help="Model evaluation commands")
console = Console()


@app.command()
def metrics(
    model: Path = typer.Option(..., "--model", "-m", help="Model checkpoint"),
    data: Path = typer.Option(..., "--data", "-d", help="Dataset directory"),
    split: str = typer.Option("val", "--split", "-s", help="Dataset split (train/val/test)"),
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="Batch size"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results to JSON"),
):
    """Compute evaluation metrics on dataset."""
    import json
    import torch
    from torch.utils.data import DataLoader

    from road_topology.segmentation.dataset import RoadTopologyDataset
    from road_topology.segmentation.transforms import get_val_transforms
    from road_topology.segmentation.models import SegFormerModel
    from road_topology.evaluation.metrics import compute_miou, compute_per_class_iou, compute_f1_score

    console.print(f"[blue]Evaluating model on {split} split[/blue]")

    # Load model
    seg_model = SegFormerModel.load(model)
    seg_model.eval()
    seg_model.to(get_device())

    # Create dataset
    dataset = RoadTopologyDataset(data, split, get_val_transforms())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Collect predictions
    all_preds = []
    all_targets = []

    console.print(f"Running inference on {len(dataset)} images...")

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(seg_model.device)
            masks = batch["mask"]

            preds = seg_model.predict(images)
            all_preds.append(preds.cpu())
            all_targets.append(masks)

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    # Compute metrics
    miou = compute_miou(preds, targets)
    class_iou = compute_per_class_iou(preds, targets)
    f1_scores = compute_f1_score(preds, targets)

    # Display results
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Overall mIoU", f"{miou:.4f}")

    console.print(table)

    # Per-class results
    class_table = Table(title="Per-Class Metrics")
    class_table.add_column("Class", style="cyan")
    class_table.add_column("IoU", style="magenta")
    class_table.add_column("F1", style="green")

    for class_name in class_iou.keys():
        iou = class_iou[class_name]
        f1 = f1_scores.get(class_name, 0.0)
        class_table.add_row(class_name, f"{iou:.4f}", f"{f1:.4f}")

    console.print(class_table)

    # Save to JSON
    if output:
        results = {
            "miou": float(miou),
            "class_iou": {k: float(v) for k, v in class_iou.items()},
            "f1_scores": {k: float(v) for k, v in f1_scores.items()},
            "num_images": len(dataset),
        }

        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(results, f, indent=2)

        console.print(f"\n[green]Results saved to {output}[/green]")


@app.command()
def compare(
    models: list[Path] = typer.Option(..., "--model", "-m", help="Model checkpoints to compare"),
    data: Path = typer.Option(..., "--data", "-d", help="Dataset directory"),
    split: str = typer.Option("val", "--split", "-s", help="Dataset split"),
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="Batch size"),
):
    """Compare multiple models on the same dataset."""
    import torch
    from torch.utils.data import DataLoader

    from road_topology.segmentation.dataset import RoadTopologyDataset
    from road_topology.segmentation.transforms import get_val_transforms
    from road_topology.segmentation.models import SegFormerModel
    from road_topology.evaluation.metrics import compute_miou, compute_per_class_iou

    console.print(f"[blue]Comparing {len(models)} models[/blue]")

    # Create dataset once
    dataset = RoadTopologyDataset(data, split, get_val_transforms())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Get ground truth once
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            all_targets.append(batch["mask"])
    targets = torch.cat(all_targets).numpy()

    # Evaluate each model
    results = {}

    for model_path in models:
        console.print(f"\nEvaluating: {model_path.name}")

        seg_model = SegFormerModel.load(model_path)
        seg_model.eval()
        seg_model.to(get_device())

        all_preds = []
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(seg_model.device)
                preds = seg_model.predict(images)
                all_preds.append(preds.cpu())

        preds = torch.cat(all_preds).numpy()

        miou = compute_miou(preds, targets)
        class_iou = compute_per_class_iou(preds, targets)

        results[model_path.name] = {
            "miou": miou,
            "class_iou": class_iou,
        }

    # Display comparison table
    table = Table(title="Model Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("mIoU", style="magenta")

    for model_name, metrics in results.items():
        table.add_row(model_name, f"{metrics['miou']:.4f}")

    console.print(table)


@app.command()
def confusion_matrix(
    model: Path = typer.Option(..., "--model", "-m", help="Model checkpoint"),
    data: Path = typer.Option(..., "--data", "-d", help="Dataset directory"),
    split: str = typer.Option("val", "--split", "-s", help="Dataset split"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save confusion matrix plot"),
):
    """Generate confusion matrix for model predictions."""
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    from road_topology.segmentation.dataset import RoadTopologyDataset
    from road_topology.segmentation.transforms import get_val_transforms
    from road_topology.segmentation.models import SegFormerModel
    from road_topology.core.types import CLASS_NAMES

    console.print(f"[blue]Generating confusion matrix[/blue]")

    # Load model
    seg_model = SegFormerModel.load(model)
    seg_model.eval()
    seg_model.to(get_device())

    # Create dataset
    dataset = RoadTopologyDataset(data, split, get_val_transforms())
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    # Collect predictions
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(seg_model.device)
            masks = batch["mask"]

            preds = seg_model.predict(images)
            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(masks.numpy().flatten())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # Compute confusion matrix
    cm = confusion_matrix(targets, preds)

    # Normalize
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=300)
        console.print(f"[green]Saved confusion matrix to {output}[/green]")
    else:
        plt.show()


@app.command()
def export(
    model: Path = typer.Option(..., "--model", "-m", help="Model checkpoint"),
    output: Path = typer.Option(..., "--output", "-o", help="Output path for exported model"),
    format: str = typer.Option("onnx", "--format", "-f", help="Export format (onnx/torchscript)"),
):
    """Export model to ONNX or TorchScript format."""
    import torch

    from road_topology.segmentation.models import SegFormerModel

    console.print(f"[blue]Exporting model to {format}[/blue]")

    # Load model
    seg_model = SegFormerModel.load(model)
    seg_model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 512, 512)

    if format == "onnx":
        torch.onnx.export(
            seg_model,
            dummy_input,
            output,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "output": {0: "batch_size", 2: "height", 3: "width"},
            },
        )
        console.print(f"[green]Exported to ONNX: {output}[/green]")

    elif format == "torchscript":
        traced = torch.jit.trace(seg_model, dummy_input)
        traced.save(str(output))
        console.print(f"[green]Exported to TorchScript: {output}[/green]")

    else:
        console.print(f"[red]Unsupported format: {format}[/red]")
        raise typer.Exit(1)
