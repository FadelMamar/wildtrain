"""Registration-related CLI commands."""

import typer
from pathlib import Path
import traceback

from ..config_loader import ConfigLoader
from ...models.register import ModelRegistrar
from ...shared.config_types import ConfigType
from ...shared.validation import ConfigFileNotFoundError, ConfigParseError, ConfigValidationError
from .utils import console, setup_logging, log_file_path

register_app = typer.Typer(name="register", help="Model registration commands")


@register_app.command()
def detector(
    config: Path = typer.Option("", "--config", "-c", help="Path to detector registration configuration file"),
    weights_path: Path = typer.Option(None, "--weights-path", help="Path to the model weights file"),
    name: str = typer.Option("localizer", "--name", "-n", help="Model name for registration"),
    export_format: str = typer.Option("pt", "--format", "-f", help="Export format (torchscript, openvino, etc.)"),
    image_size: int = typer.Option(800, "--image-size", help="Input image size"),
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="Batch size for inference"),
    device: str = typer.Option("cpu", "--device", "-d", help="Device to use for export"),
    dynamic: bool = typer.Option(False, "--dynamic", help="Whether to use dynamic batching"),
    task: str = typer.Option("detect", "--task", help="YOLO task type (detect, classify, segment)"),
    mlflow_tracking_uri: str = typer.Option("http://localhost:5000", "--mlflow-uri", help="MLflow tracking server URI"),
    template: bool = typer.Option(False, "--template",help="Show default configuration template instead of registration")
) -> None:
    """Register a YOLO detection model to MLflow Model Registry."""
    if template:
        console.print(f"[bold green]Showing default detector registration configuration template...[/bold green]")
        try:
            template_yaml = ConfigLoader.generate_default_config(ConfigType.DETECTOR_REGISTRATION)
            console.print(f"\n[bold blue]Default detector registration configuration template:[/bold blue]")
            console.print(f"\n```yaml\n{template_yaml}```")
            return
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Failed to generate template: {str(e)}")
            raise typer.Exit(1)
    
    if config.exists():
        # Load configuration from file
        console.print(f"[bold green]Loading detector registration config from:[/bold green] {config}")
        try:
            validated_config = ConfigLoader.load_detector_registration_config(config)
            console.print(f"[bold green]✓[/bold green] Detector registration configuration validated successfully")
            
            # Use config values
            weights_path = Path(validated_config.weights_path)
            name = validated_config.name
            export_format = validated_config.export_format
            image_size = validated_config.image_size
            batch_size = validated_config.batch_size
            device = validated_config.device
            dynamic = validated_config.dynamic
            task = validated_config.task
            mlflow_tracking_uri = validated_config.mlflow_tracking_uri
            
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Configuration error: {str(e)}")
            raise typer.Exit(1)
    else:
        # Use command line arguments
        if weights_path is None:
            console.print(f"[bold red]✗[/bold red] Must provide either --config or --weights-path")
            raise typer.Exit(1)
    
    console.print(f"[bold green]Registering detector model:[/bold green] {weights_path}")
    
    log_file = log_file_path("register_detector")
    setup_logging(log_file=log_file)
    
    try:
        # Validate task parameter
        valid_tasks = ["detect", "classify", "segment"]
        if task not in valid_tasks:
            console.print(f"[bold red]✗[/bold red] Invalid task: {task}. Valid tasks are: {', '.join(valid_tasks)}")
            raise typer.Exit(1)
        
        # Create model registrar
        registrar = ModelRegistrar(mlflow_tracking_uri=mlflow_tracking_uri)
        
        # Register the detector
        registrar.register_detector(
            weights_path=weights_path,
            name=name,
            export_format=export_format,
            image_size=image_size,
            batch_size=batch_size,
            device=device,
            dynamic=dynamic,
            task=task
        )
        
        console.print(f"[bold green]✓[/bold green] Successfully registered detector model: {name}")
        console.print(f"[bold blue]Model registered to MLflow tracking URI:[/bold blue] {mlflow_tracking_uri}")
        
    except FileNotFoundError as e:
        console.print(f"[bold red]✗[/bold red] Model weights not found: {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Registration failed: {str(e)}")
        if log_file.exists():
            console.print(f"[dim]Check logs at: {log_file}[/dim]")
        raise typer.Exit(1)


@register_app.command()
def classifier(
    config: Path = typer.Option("", "--config", "-c", help="Path to classifier registration configuration file"),
    weights_path: Path = typer.Option(None, "--weights-path", help="Path to the model checkpoint file"),
    name: str = typer.Option("classifier", "--name", "-n", help="Model name for registration"),
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="Batch size for inference"),
    mlflow_tracking_uri: str = typer.Option("http://localhost:5000", "--mlflow-uri", help="MLflow tracking server URI"),
    template: bool = typer.Option(False, "--template", help="Show default configuration template instead of registration")
) -> None:
    """Register a classification model to MLflow Model Registry."""
    if template:
        console.print(f"[bold green]Showing default classifier registration configuration template...[/bold green]")
        try:
            template_yaml = ConfigLoader.generate_default_config(ConfigType.CLASSIFIER_REGISTRATION)
            console.print(f"\n[bold blue]Default classifier registration configuration template:[/bold blue]")
            console.print(f"\n```yaml\n{template_yaml}```")
            return
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Failed to generate template: {str(e)}")
            raise typer.Exit(1)
    
    if config.exists():
        # Load configuration from file
        console.print(f"[bold green]Loading classifier registration config from:[/bold green] {config}")
        try:
            validated_config = ConfigLoader.load_classifier_registration_config(config)
            console.print(f"[bold green]✓[/bold green] Classifier registration configuration validated successfully")
            
            # Use config values
            weights_path = Path(validated_config.weights_path)
            name = validated_config.name
            batch_size = validated_config.batch_size
            mlflow_tracking_uri = validated_config.mlflow_tracking_uri
            
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Configuration error: {str(e)}")
            raise typer.Exit(1)
    else:
        # Use command line arguments
        if weights_path is None:
            console.print(f"[bold red]✗[/bold red] Must provide either --config or --weights-path")
            raise typer.Exit(1)
    
    console.print(f"[bold green]Registering classifier model:[/bold green] {weights_path}")
    
    log_file = log_file_path("register_classifier")
    setup_logging(log_file=log_file)
    
    try:
        # Create model registrar
        registrar = ModelRegistrar(mlflow_tracking_uri=mlflow_tracking_uri)
        
        # Register the classifier
        registrar.register_classifier(
            weights_path=weights_path,
            name=name,
            batch_size=batch_size
        )
        
        console.print(f"[bold green]✓[/bold green] Successfully registered classifier model: {name}")
        console.print(f"[bold blue]Model registered to MLflow tracking URI:[/bold blue] {mlflow_tracking_uri}")
        
    except FileNotFoundError as e:
        console.print(f"[bold red]✗[/bold red] Model weights not found: {traceback.format_exc()}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Registration failed: {traceback.format_exc()}")
        if log_file.exists():
            console.print(f"[dim]Check logs at: {log_file}[/dim]")
        raise typer.Exit(1)
