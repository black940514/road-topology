"""CLI entry point for Road Topology Segmentation."""
from pathlib import Path

import typer
from rich.console import Console

from road_topology.cli import pseudolabel, train, infer, evaluate

app = typer.Typer(
    name="road-topo",
    help="Road Topology Segmentation CLI",
    add_completion=False,
)

console = Console()

# Add subcommands
app.add_typer(pseudolabel.app, name="pseudolabel", help="Generate pseudo-labels")
app.add_typer(train.app, name="train", help="Train segmentation models")
app.add_typer(infer.app, name="infer", help="Run inference")
app.add_typer(evaluate.app, name="evaluate", help="Evaluate models")


@app.command()
def version():
    """Show version information."""
    from road_topology import __version__
    console.print(f"road-topology version {__version__}")


@app.command()
def info():
    """Show system and configuration information."""
    import torch
    from road_topology.core.device import get_device_info

    device_info = get_device_info()

    console.print("[bold]Road Topology Segmentation[/bold]")
    console.print(f"PyTorch version: {torch.__version__}")
    console.print(f"Default device: {device_info.name} ({device_info.type})")
    console.print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        console.print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    console.print(f"MPS available: {torch.backends.mps.is_available()}")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
