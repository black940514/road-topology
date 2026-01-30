"""Pseudo-label generation CLI commands."""
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress

app = typer.Typer(help="Pseudo-label generation commands")
console = Console()


@app.command()
def generate(
    video: Path = typer.Option(..., "--video", "-v", help="Input video path"),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    config: Path = typer.Option(None, "--config", "-c", help="Config file"),
    trajectory_width: int = typer.Option(50, help="Trajectory width in pixels"),
    threshold: float = typer.Option(0.1, help="Mask threshold"),
    frame_skip: int = typer.Option(0, help="Frames to skip between processing"),
):
    """Generate pseudo-labels from video."""
    from road_topology.core.config import load_config
    from road_topology.pseudolabel.generator import create_generator, PseudoLabelConfig

    cfg = load_config(config)

    pl_config = PseudoLabelConfig(
        trajectory_width=trajectory_width,
        mask_threshold=threshold,
        frame_skip=frame_skip,
    )

    console.print(f"[blue]Processing video: {video}[/blue]")

    generator = create_generator(cfg)

    with Progress() as progress:
        task = progress.add_task("Generating pseudo-labels...", total=100)

        def update_progress(current, total):
            progress.update(task, completed=int(100 * current / total))

        result = generator.process_video(video, progress_callback=update_progress)

    paths = generator.save_result(result, output)

    console.print(f"[green]Saved pseudo-labels to {output}[/green]")
    console.print(f"  Trajectories: {len(result.trajectories)}")
    console.print(f"  Coverage: {result.metadata.get('coverage_ratio', 0):.1%}")


@app.command()
def visualize(
    mask: Path = typer.Option(..., "--mask", "-m", help="Mask file path"),
    video: Path = typer.Option(None, "--video", "-v", help="Optional video for overlay"),
    output: Path = typer.Option(None, "--output", "-o", help="Output path"),
):
    """Visualize pseudo-label mask."""
    import cv2
    import numpy as np
    from road_topology.core.types import CLASS_COLORS

    mask_img = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)

    # Create colored visualization
    h, w = mask_img.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS.items():
        vis[mask_img == class_id] = color

    if video:
        cap = cv2.VideoCapture(str(video))
        ret, frame = cap.read()
        cap.release()
        if ret:
            vis = cv2.addWeighted(frame, 0.5, vis, 0.5, 0)

    if output:
        cv2.imwrite(str(output), vis)
        console.print(f"[green]Saved visualization to {output}[/green]")
    else:
        cv2.imshow("Pseudo-label Visualization", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
