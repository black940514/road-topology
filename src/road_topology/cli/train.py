"""Training CLI commands."""
from pathlib import Path

import typer
from rich.console import Console

from road_topology.core.device import get_device

app = typer.Typer(help="Model training commands")
console = Console()


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Training config YAML"),
    data: Path = typer.Option(..., "--data", "-d", help="Dataset root directory"),
    output: Path = typer.Option(Path("./outputs"), "--output", "-o", help="Output directory"),
    epochs: int = typer.Option(None, "--epochs", help="Override epochs"),
    batch_size: int = typer.Option(None, "--batch-size", "-b", help="Override batch size"),
    resume: Path = typer.Option(None, "--resume", "-r", help="Resume from checkpoint"),
):
    """Train segmentation model."""
    import torch
    from torch.utils.data import DataLoader

    from road_topology.core.config import load_config
    from road_topology.segmentation.dataset import RoadTopologyDataset
    from road_topology.segmentation.transforms import get_train_transforms, get_val_transforms
    from road_topology.segmentation.models import SegFormerModel
    from road_topology.segmentation.trainer import SegmentationTrainer

    cfg = load_config(config)

    if epochs:
        cfg.training.epochs = epochs
    if batch_size:
        cfg.training.batch_size = batch_size

    console.print(f"[blue]Training with config: {config}[/blue]")
    console.print(f"  Device: {cfg.device}")
    console.print(f"  Epochs: {cfg.training.epochs}")
    console.print(f"  Batch size: {cfg.training.batch_size}")

    # Create datasets
    train_transforms = get_train_transforms(cfg.segmentation.image_size)
    val_transforms = get_val_transforms(cfg.segmentation.image_size)

    train_dataset = RoadTopologyDataset(data, "train", train_transforms)
    val_dataset = RoadTopologyDataset(data, "val", val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    # Create model
    model = SegFormerModel(
        backbone=cfg.segmentation.backbone,
        num_classes=cfg.segmentation.num_classes,
    )
    model.to(cfg.device)

    # Create trainer
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg.training,
        output_dir=output,
    )

    if resume:
        trainer.load_checkpoint(resume)

    # Train
    history = trainer.train(cfg.training.epochs)

    console.print(f"[green]Training complete! Best mIoU: {history.get('best_miou', 'N/A')}[/green]")


@app.command()
def validate(
    model: Path = typer.Option(..., "--model", "-m", help="Model checkpoint"),
    data: Path = typer.Option(..., "--data", "-d", help="Validation dataset"),
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="Batch size"),
):
    """Validate model on dataset."""
    from torch.utils.data import DataLoader

    from road_topology.segmentation.dataset import RoadTopologyDataset
    from road_topology.segmentation.transforms import get_val_transforms
    from road_topology.segmentation.models import SegFormerModel
    from road_topology.evaluation.metrics import compute_miou, compute_per_class_iou

    console.print(f"[blue]Validating model: {model}[/blue]")

    # Load model
    seg_model = SegFormerModel.load(model)
    seg_model.eval()

    # Create dataset
    val_dataset = RoadTopologyDataset(data, "val", get_val_transforms())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate
    all_preds = []
    all_targets = []

    import torch
    device = get_device()
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"]

            preds = seg_model.predict(images)
            all_preds.append(preds.cpu())
            all_targets.append(masks)

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    miou = compute_miou(preds, targets)
    class_iou = compute_per_class_iou(preds, targets)

    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  mIoU: {miou:.4f}")
    for name, iou in class_iou.items():
        console.print(f"  {name}: {iou:.4f}")
