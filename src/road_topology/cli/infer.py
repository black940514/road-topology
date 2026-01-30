"""Inference CLI commands."""
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress

from road_topology.core.device import get_device

app = typer.Typer(help="Inference commands")
console = Console()


@app.command()
def image(
    model: Path = typer.Option(..., "--model", "-m", help="Model checkpoint"),
    input: Path = typer.Option(..., "--input", "-i", help="Input image path"),
    output: Path = typer.Option(None, "--output", "-o", help="Output path"),
    visualize: bool = typer.Option(True, "--visualize/--no-visualize", help="Visualize result"),
    overlay: bool = typer.Option(True, "--overlay/--no-overlay", help="Overlay on original image"),
):
    """Run inference on a single image."""
    import cv2
    import torch
    import numpy as np

    from road_topology.segmentation.models import SegFormerModel
    from road_topology.segmentation.transforms import get_val_transforms
    from road_topology.core.types import CLASS_COLORS

    console.print(f"[blue]Running inference on: {input}[/blue]")

    # Load model
    seg_model = SegFormerModel.load(model)
    seg_model.eval()
    seg_model.to(get_device())

    # Load and preprocess image
    image = cv2.imread(str(input))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transforms = get_val_transforms()
    transformed = transforms(image=image_rgb)
    input_tensor = transformed["image"].unsqueeze(0)

    # Run inference
    with torch.no_grad():
        input_tensor = input_tensor.to(seg_model.device)
        pred_mask = seg_model.predict(input_tensor)[0]

    # Create visualization
    h, w = pred_mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS.items():
        vis[pred_mask == class_id] = color

    if overlay:
        resized_image = cv2.resize(image, (w, h))
        vis = cv2.addWeighted(resized_image, 0.5, vis, 0.5, 0)

    # Save or display
    if output:
        cv2.imwrite(str(output), vis)
        console.print(f"[green]Saved result to {output}[/green]")

    if visualize:
        cv2.imshow("Segmentation Result", vis)
        console.print("[yellow]Press any key to close...[/yellow]")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Print statistics
    unique_classes = np.unique(pred_mask)
    console.print(f"\n[bold]Detected classes:[/bold]")
    for class_id in unique_classes:
        pixel_count = np.sum(pred_mask == class_id)
        percentage = 100 * pixel_count / pred_mask.size
        console.print(f"  Class {class_id}: {percentage:.2f}%")


@app.command()
def video(
    model: Path = typer.Option(..., "--model", "-m", help="Model checkpoint"),
    input: Path = typer.Option(..., "--input", "-i", help="Input video path"),
    output: Path = typer.Option(..., "--output", "-o", help="Output video path"),
    fps: Optional[int] = typer.Option(None, "--fps", help="Output FPS (default: same as input)"),
    overlay: bool = typer.Option(True, "--overlay/--no-overlay", help="Overlay on original frames"),
):
    """Run inference on a video."""
    import cv2
    import torch
    import numpy as np

    from road_topology.segmentation.models import SegFormerModel
    from road_topology.segmentation.transforms import get_val_transforms
    from road_topology.core.types import CLASS_COLORS

    console.print(f"[blue]Processing video: {input}[/blue]")

    # Load model
    seg_model = SegFormerModel.load(model)
    seg_model.eval()
    seg_model.to(get_device())

    # Open video
    cap = cv2.VideoCapture(str(input))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_fps = fps or input_fps

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output), fourcc, output_fps, (width, height))

    transforms = get_val_transforms()

    with Progress() as progress:
        task = progress.add_task("Processing frames...", total=total_frames)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            transformed = transforms(image=frame_rgb)
            input_tensor = transformed["image"].unsqueeze(0)

            # Inference
            with torch.no_grad():
                input_tensor = input_tensor.to(seg_model.device)
                pred_mask = seg_model.predict(input_tensor)[0]

            # Create visualization
            vis = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
            for class_id, color in CLASS_COLORS.items():
                vis[pred_mask == class_id] = color

            # Resize to original frame size
            vis = cv2.resize(vis, (width, height))

            if overlay:
                vis = cv2.addWeighted(frame, 0.5, vis, 0.5, 0)

            out.write(vis)

            frame_idx += 1
            progress.update(task, completed=frame_idx)

    cap.release()
    out.release()

    console.print(f"[green]Saved output video to {output}[/green]")
    console.print(f"  Processed {frame_idx} frames")


@app.command()
def batch(
    model: Path = typer.Option(..., "--model", "-m", help="Model checkpoint"),
    input_dir: Path = typer.Option(..., "--input", "-i", help="Input directory"),
    output_dir: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    pattern: str = typer.Option("*.jpg", "--pattern", "-p", help="File pattern to match"),
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="Batch size"),
):
    """Run inference on a batch of images."""
    import cv2
    import torch
    import numpy as np
    from torch.utils.data import DataLoader

    from road_topology.segmentation.models import SegFormerModel
    from road_topology.segmentation.transforms import get_val_transforms
    from road_topology.core.types import CLASS_COLORS

    console.print(f"[blue]Processing images in: {input_dir}[/blue]")

    # Find images
    image_paths = list(input_dir.glob(pattern))
    console.print(f"Found {len(image_paths)} images")

    if not image_paths:
        console.print("[yellow]No images found![/yellow]")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    seg_model = SegFormerModel.load(model)
    seg_model.eval()
    seg_model.to(get_device())

    transforms = get_val_transforms()

    with Progress() as progress:
        task = progress.add_task("Processing images...", total=len(image_paths))

        for img_path in image_paths:
            # Load and preprocess
            image = cv2.imread(str(img_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            transformed = transforms(image=image_rgb)
            input_tensor = transformed["image"].unsqueeze(0)

            # Inference
            with torch.no_grad():
                input_tensor = input_tensor.to(seg_model.device)
                pred_mask = seg_model.predict(input_tensor)[0]

            # Create visualization
            h, w = pred_mask.shape
            vis = np.zeros((h, w, 3), dtype=np.uint8)

            for class_id, color in CLASS_COLORS.items():
                vis[pred_mask == class_id] = color

            # Resize and overlay
            resized_image = cv2.resize(image, (w, h))
            vis = cv2.addWeighted(resized_image, 0.5, vis, 0.5, 0)

            # Save
            output_path = output_dir / f"{img_path.stem}_seg.png"
            cv2.imwrite(str(output_path), vis)

            progress.update(task, advance=1)

    console.print(f"[green]Processed {len(image_paths)} images[/green]")
    console.print(f"Results saved to {output_dir}")


@app.command()
def lane(
    model: Path = typer.Option(..., "--model", "-m", help="Lane model checkpoint"),
    input: Path = typer.Option(..., "--input", "-i", help="Input image or directory"),
    output: Path = typer.Option(..., "--output", "-o", help="Output path or directory"),
    visualize: bool = typer.Option(True, "--visualize/--no-visualize", help="Visualize results"),
):
    """Run lane segmentation inference with instance separation."""
    import cv2
    import numpy as np

    from road_topology.inference import LanePredictor
    from road_topology.evaluation.visualize import visualize_lane_instances

    console.print(f"[blue]Running lane instance inference[/blue]")

    # Load predictor
    predictor = LanePredictor(model_path=str(model), device="auto")
    console.print(f"  Model loaded from: {model}")
    console.print(f"  Device: {predictor.device}")

    # Check if input is file or directory
    if input.is_file():
        # Single image inference
        image = cv2.imread(str(input))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run inference
        results = predictor.predict(image_rgb)

        semantic_mask = results["semantic_mask"]
        instance_mask = results["lane_instances"]
        lane_count = results["lane_count"]

        # Visualize
        if visualize:
            vis = visualize_lane_instances(image_rgb, semantic_mask, instance_mask, alpha=0.5)

            # Save
            output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            console.print(f"[green]Saved result to {output}[/green]")

        # Print statistics
        console.print(f"\n[bold]Detected {lane_count} lane instances[/bold]")

        for instance_id in range(1, lane_count + 1):
            pixel_count = int((instance_mask == instance_id).sum())
            console.print(f"  Lane {instance_id}: {pixel_count} pixels")

        # Show crosswalk info
        crosswalk_pixels = int((semantic_mask == 3).sum())  # crosswalk class = 3
        if crosswalk_pixels > 0:
            console.print(f"\n[bold]Crosswalk detected:[/bold] {crosswalk_pixels} pixels")

    else:
        # Batch processing
        image_paths = list(input.glob("*.jpg")) + list(input.glob("*.png"))
        console.print(f"Found {len(image_paths)} images")

        if not image_paths:
            console.print("[yellow]No images found![/yellow]")
            return

        output.mkdir(parents=True, exist_ok=True)

        with Progress() as progress:
            task = progress.add_task("Processing images...", total=len(image_paths))

            for img_path in image_paths:
                # Load image
                image = cv2.imread(str(img_path))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Inference
                results = predictor.predict(image_rgb)

                semantic_mask = results["semantic_mask"]
                instance_mask = results["lane_instances"]

                # Visualize
                if visualize:
                    vis = visualize_lane_instances(image_rgb, semantic_mask, instance_mask, alpha=0.5)
                    output_path = output / f"{img_path.stem}_lane_instances.png"
                    cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

                # Save instance mask
                mask_path = output / f"{img_path.stem}_instances.png"
                cv2.imwrite(str(mask_path), instance_mask.astype(np.uint8))

                progress.update(task, advance=1)

        console.print(f"[green]Processed {len(image_paths)} images[/green]")
        console.print(f"Results saved to {output}")
