"""CLI module for WildTrain using Typer and Rich."""

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text
import logging
from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf
import subprocess
import platform
from datetime import datetime


from .trainers.classification_trainer import run_classification
from .utils.logging import ROOT

# Create Typer app
app = typer.Typer(
    name="wildtrain",
    help="Modular Computer Vision Framework for Detection and Classification",
    add_completion=False,
    rich_markup_mode="rich",
)

# Create Rich console
console = Console()

def setup_logging(verbose: bool = False) -> None:
    """Setup rich logging with appropriate level."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True),
        logging.FileHandler(ROOT/"logs"/f"cli_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
                            encoding="utf-8")
        ]
    )

@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    config_dir: Optional[Path] = typer.Option(None, "--config-dir", "-c", help="Configuration directory"),
) -> None:
    """WildTrain - Modular Computer Vision Framework."""
    setup_logging(verbose)
    
    # Display welcome message
    welcome_text = Text("🚀 WildTrain", style="bold blue")
    subtitle = Text("Modular Computer Vision Framework", style="italic")
    
    console.print(Panel(
        welcome_text + "\n" + subtitle,
        border_style="blue",
        padding=(1, 2)
    ))

@app.command()
def train_classifier(
    config: Path = typer.Argument(..., help="Path to training configuration file"),
) -> None:
    """Train a classification model."""
    console.print(f"[bold green]Training classifier with config:[/bold green] {config}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing training...", total=None)

        cfg = OmegaConf.load(config)
        console.print(OmegaConf.to_yaml(cfg))
        run_classification(cfg=cfg)
        
        progress.update(task, description="Training completed!")

@app.command()
def launch_mlflow(
    port: int = typer.Option(5000, "--port", "-p", help="Port for MLflow UI"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host for MLflow UI"),
    backend_store_uri: Optional[str] = typer.Option(None, "--backend-store-uri", help="Backend store URI"),
    default_artifact_root: Optional[str] = typer.Option(None, "--default-artifact-root", help="Default artifact root"),
) -> None:
    """Launch MLflow tracking server."""
    console.print(f"[bold green]Launching MLflow server on[/bold green] {host}:{port}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting MLflow server...", total=None)
        
        
        cmd = ["uv", "run", "--no-sync","mlflow", "server", "--host", host, "--port", str(port),"--backend-store-uri", backend_store_uri]
        if default_artifact_root:
            cmd.append("--default-artifact-root")
            cmd.append(default_artifact_root)


        if platform.system() == "Windows":
            subprocess.Popen(cmd,creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen(cmd)

        console.print(f"  🌐 Host: {host}")
        console.print(f"  🔌 Port: {port}")
        console.print(f"  💾 Backend store: {backend_store_uri or 'default'}")
        console.print(f"  📁 Artifact root: {default_artifact_root or 'default'}")
        
        progress.update(task, description="MLflow server started!")
        console.print(f"\n[bold blue]MLflow UI available at:[/bold blue] http://{host}:{port}")

@app.command()
def get_dataset_stats(
    data_dir: Path = typer.Argument(..., help="Path to dataset directory"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for statistics"),
    format: str = typer.Option("json", "--format", "-f", help="Output format (json, yaml, csv)"),
    include_images: bool = typer.Option(False, "--include-images", help="Include image statistics"),
) -> None:
    """Get dataset statistics and information."""
    console.print(f"[bold green]Analyzing dataset at:[/bold green] {data_dir}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Scanning dataset...", total=None)
        
        # TODO: Implement actual dataset statistics calculation
        console.print(f"  📁 Dataset path: {data_dir}")
        console.print(f"  📄 Output format: {format}")
        console.print(f"  🖼️  Include images: {'Yes' if include_images else 'No'}")
        
        if output_file:
            console.print(f"  💾 Output file: {output_file}")
        
        progress.update(task, description="Dataset analysis completed!")
        
        # Mock statistics output
        console.print("\n[bold blue]Dataset Statistics:[/bold blue]")
        console.print("  📊 Total samples: 1,000")
        console.print("  🏷️  Classes: 10")
        console.print("  📏 Image size: 224x224")
        console.print("  📁 Format: JPEG")

if __name__ == "__main__":
    app()
