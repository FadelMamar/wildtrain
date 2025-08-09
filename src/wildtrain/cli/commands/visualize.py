"""Visualization-related CLI commands."""

import typer
from pathlib import Path
from omegaconf import OmegaConf

from ..config_loader import ConfigLoader
from ...visualization import add_predictions_from_classifier, add_predictions_from_detector
from ...models.detector import Detector
from ...models.localizer import UltralyticsLocalizer
from ...models.classifier import GenericClassifier
from ...shared.config_types import ConfigType
from ...shared.validation import ConfigFileNotFoundError, ConfigParseError, ConfigValidationError
from .utils import console, setup_logging, log_file_path

visualize_app = typer.Typer(name="visualize", help="Visualization commands")


@visualize_app.command()
def classifier_predictions(
    config: Path = typer.Option("", "--config", "-c",help="Path to classification visualization configuration YAML file"),
    template: bool = typer.Option(False, "--template", "-t", help="Show default configuration template instead of visualization")
) -> None:
    """Upload classifier predictions to a FiftyOne dataset for visualization using YAML configuration."""
    if template:
        console.print(f"[bold green]Showing default classification visualization configuration template...[/bold green]")
        try:
            template_yaml = ConfigLoader.generate_default_config(ConfigType.CLASSIFICATION_VISUALIZATION)
            console.print(f"\n[bold blue]Default classification visualization configuration template:[/bold blue]")
            console.print(f"\n```yaml\n{template_yaml}```")
            return
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Failed to generate template: {str(e)}")
            raise typer.Exit(1)
    
    console.print(f"[bold green]Loading classification visualization config from:[/bold green] {config}")
    
    try:
        # Load and validate configuration using Pydantic
        validated_config = ConfigLoader.load_classification_visualization_config(config)
        console.print(f"[bold green]✓[/bold green] Classification visualization configuration validated successfully")
        
        # Convert validated config back to DictConfig for backward compatibility
        cfg = OmegaConf.create(validated_config.model_dump())
        console.print("cfg:",cfg)
        
        # Extract configuration values
        dataset_name = cfg.dataset_name
        checkpoint_path = cfg.checkpoint_path
        prediction_field = cfg.prediction_field
        batch_size = cfg.batch_size
        device = cfg.device
        debug = cfg.debug
        
        console.print(f"[bold green]Uploading classifier predictions to FiftyOne dataset:[/bold green] {dataset_name}")
        
        log_file = log_file_path("visualize_classifier_predictions")
        setup_logging(log_file=log_file)

        add_predictions_from_classifier(
            dataset_name=dataset_name,
            checkpoint_path=str(checkpoint_path),
            prediction_field=prediction_field,
            batch_size=batch_size,
            device=device,
            debug=debug,
        )
        console.print(f"[bold blue]Classifier predictions uploaded to FiftyOne dataset:[/bold blue] {dataset_name}")
        
    except (ConfigFileNotFoundError, ConfigParseError, ConfigValidationError) as e:
        console.print(f"[bold red]✗[/bold red] Configuration error: {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Visualization failed: {str(e)}")
        raise typer.Exit(1)


@visualize_app.command()
def detector_predictions(
    config: Path = typer.Option("", "--config", "-c",help="Path to visualization configuration YAML file"),
    template: bool = typer.Option(False, "--template", "-t", help="Show default configuration template instead of visualization")
) -> None:
    """Upload detector predictions to a FiftyOne dataset for visualization using YAML configuration."""
    if template:
        console.print(f"[bold green]Showing default detection visualization configuration template...[/bold green]")
        try:
            template_yaml = ConfigLoader.generate_default_config(ConfigType.DETECTION_VISUALIZATION)
            console.print(f"\n[bold blue]Default detection visualization configuration template:[/bold blue]")
            console.print(f"\n```yaml\n{template_yaml}```")
            return
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Failed to generate template: {str(e)}")
            raise typer.Exit(1)
    
    console.print(f"[bold green]Loading visualization config from:[/bold green] {config}")
    
    try:
        # Load and validate configuration using Pydantic
        validated_config = ConfigLoader.load_detection_visualization_config(config)
        console.print(f"[bold green]✓[/bold green] Detection visualization configuration validated successfully")
        
        # Convert validated config back to DictConfig for backward compatibility
        cfg = OmegaConf.create(validated_config.model_dump())
        console.print("cfg:",cfg)
        
        # Extract configuration values
        dataset_name = cfg.dataset_name
        prediction_field = cfg.prediction_field
        
        localizer_cfg = cfg.localizer
        classifier_cfg = cfg.classifier
        
        console.print(f"[bold green]Uploading detector predictions to FiftyOne dataset:[/bold green] {dataset_name}")
        
        # Create localizer with config
        localizer = UltralyticsLocalizer.from_config(localizer_cfg)
        
        # Create classifier if checkpoint provided
        classifier = None
        if classifier_cfg.checkpoint is not None:
            console.print(f"[bold blue]Loading classifier from:[/bold blue] {classifier_cfg.checkpoint}")
            classifier = GenericClassifier.load_from_checkpoint(str(classifier_cfg.checkpoint))
        
        # Create detector
        detector = Detector(localizer=localizer, classifier=classifier)
        
        log_file = log_file_path("visualize_detector_predictions")
        setup_logging(log_file=log_file)

        add_predictions_from_detector(
            dataset_name=dataset_name,
            detector=detector,
            imgsz=localizer_cfg.imgsz,
            prediction_field=prediction_field,
            batch_size=cfg.batch_size,
            debug=cfg.debug,
        )
        console.print(f"[bold blue]Detector predictions uploaded to FiftyOne dataset:[/bold blue] {dataset_name}")
        
    except (ConfigFileNotFoundError, ConfigParseError, ConfigValidationError) as e:
        console.print(f"[bold red]✗[/bold red] Configuration error: {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Visualization failed: {str(e)}")
        raise typer.Exit(1)
