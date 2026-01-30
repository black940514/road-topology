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


@app.command()
def generate_instances(
    semantic_dir: Path = typer.Option(..., "--semantic", "-s", help="Semantic mask directory"),
    output_dir: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    min_size: int = typer.Option(100, "--min-size", help="Minimum instance size (pixels)"),
    visualize: bool = typer.Option(False, "--visualize/--no-visualize", help="Generate visualizations"),
):
    """Generate lane instance masks from semantic segmentation."""
    import cv2
    import numpy as np
    from rich.progress import Progress

    console.print(f"[blue]Generating instance masks from: {semantic_dir}[/blue]")

    # Find semantic masks
    semantic_paths = list(semantic_dir.glob("*.png"))
    console.print(f"Found {len(semantic_paths)} semantic masks")

    if not semantic_paths:
        console.print("[yellow]No semantic masks found![/yellow]")
        return

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    if visualize:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

    with Progress() as progress:
        task = progress.add_task("Processing masks...", total=len(semantic_paths))

        for semantic_path in semantic_paths:
            # Load semantic mask
            semantic_mask = cv2.imread(str(semantic_path), cv2.IMREAD_GRAYSCALE)

            # Extract lane class (assuming class 1 or specific lane class ID)
            # Adjust this based on your class definitions
            lane_class_id = 1  # Modify as needed
            lane_mask = (semantic_mask == lane_class_id).astype(np.uint8)

            # Connected components analysis
            num_labels, labels = cv2.connectedComponents(lane_mask)

            # Filter by minimum size
            instance_mask = np.zeros_like(labels, dtype=np.uint8)
            instance_id = 1

            for label_id in range(1, num_labels):
                component_mask = (labels == label_id)
                size = component_mask.sum()

                if size >= min_size:
                    instance_mask[component_mask] = instance_id
                    instance_id += 1

            # Save instance mask
            output_path = output_dir / semantic_path.name
            cv2.imwrite(str(output_path), instance_mask)

            # Optionally visualize
            if visualize:
                from road_topology.evaluation.visualize import visualize_instance_masks

                # Load original image if available
                image_path = semantic_path.parent.parent / "images" / semantic_path.name
                if image_path.exists():
                    image = cv2.imread(str(image_path))
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    # Use semantic mask as placeholder
                    image_rgb = np.stack([semantic_mask] * 3, axis=-1)

                vis_path = vis_dir / f"{semantic_path.stem}_instances.png"
                visualize_instance_masks(image_rgb, semantic_mask, instance_mask, vis_path)

            progress.update(task, advance=1)

    # Compute statistics
    from road_topology.evaluation.visualize import compute_instance_mask_stats

    stats = compute_instance_mask_stats(output_dir)

    console.print("\n[bold]Statistics:[/bold]")
    console.print(f"  Images processed: {stats['num_images']}")
    console.print(f"  Avg instances/image: {stats['avg_instances_per_image']:.2f}")
    console.print(f"  Max instances: {stats['max_instances']}")
    console.print(f"  Min instances: {stats['min_instances']}")
    console.print(f"  Total instances: {stats['total_instances']}")

    console.print(f"\n[green]Instance masks saved to {output_dir}[/green]")


@app.command()
def lane(
    config: Path = typer.Option(..., "--config", "-c", help="Training config YAML"),
    data: Path = typer.Option(..., "--data", "-d", help="Dataset root directory"),
    output: Path = typer.Option(Path("./outputs"), "--output", "-o", help="Output directory"),
    epochs: int = typer.Option(None, "--epochs", help="Override epochs"),
    batch_size: int = typer.Option(None, "--batch-size", "-b", help="Override batch size"),
    resume: Path = typer.Option(None, "--resume", "-r", help="Resume from checkpoint"),
):
    """Train lane segmentation model with instance separation."""
    import torch
    from torch.utils.data import DataLoader

    from road_topology.core.config import load_config, LaneSegmentationConfig
    from road_topology.segmentation.dataset import LaneInstanceDataset
    from road_topology.segmentation.transforms import get_train_transforms, get_val_transforms
    from road_topology.segmentation.models import SegFormerLaneModel
    from road_topology.segmentation.trainer_lane import LaneSegmentationTrainer
    from road_topology.segmentation.losses import CombinedLoss, DiscriminativeLoss

    cfg = load_config(config)
    lane_cfg = LaneSegmentationConfig()

    if epochs:
        cfg.training.epochs = epochs
    if batch_size:
        cfg.training.batch_size = batch_size

    console.print(f"[blue]Training lane segmentation model (SegFormer-B5)[/blue]")
    console.print(f"  Config: {config}")
    console.print(f"  Device: {cfg.device}")
    console.print(f"  Epochs: {cfg.training.epochs}")
    console.print(f"  Batch size: {cfg.training.batch_size}")
    console.print(f"  Semantic classes: {lane_cfg.num_semantic_classes}")
    console.print(f"  Embedding dim: {lane_cfg.embedding_dim}")

    # Create datasets with instance masks
    train_transforms = get_train_transforms(cfg.segmentation.image_size)
    val_transforms = get_val_transforms(cfg.segmentation.image_size)

    train_dataset = LaneInstanceDataset(data, "train", train_transforms)
    val_dataset = LaneInstanceDataset(data, "val", val_transforms)

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

    # Create dual-head model
    model = SegFormerLaneModel(
        backbone=lane_cfg.backbone,
        num_semantic_classes=lane_cfg.num_semantic_classes,
        embedding_dim=lane_cfg.embedding_dim,
        pretrained=True,
    )
    device = torch.device(cfg.device)
    model.to(device)

    # Create losses
    semantic_criterion = CombinedLoss()
    instance_criterion = DiscriminativeLoss(
        delta_var=lane_cfg.delta_var,
        delta_dist=lane_cfg.delta_dist,
    )

    # Create optimizer with different learning rates
    param_groups = model.get_trainable_params()
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.training.learning_rate)

    # Create trainer
    trainer = LaneSegmentationTrainer(
        model=model,
        optimizer=optimizer,
        semantic_criterion=semantic_criterion,
        instance_criterion=instance_criterion,
        semantic_weight=lane_cfg.semantic_weight,
        instance_weight=lane_cfg.instance_weight,
        device=device,
    )

    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        console.print(f"[yellow]Resumed from {resume}[/yellow]")

    # Train
    best_miou = 0.0
    output.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.training.epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.validate(val_loader)

        console.print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f} - "
            f"Val mIoU: {val_metrics['miou']:.4f}"
        )

        # Save best model
        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "miou": best_miou,
            }, output / "best_model.pth")

    console.print(f"[green]Training complete![/green]")
    console.print(f"  Best mIoU: {best_miou:.4f}")
