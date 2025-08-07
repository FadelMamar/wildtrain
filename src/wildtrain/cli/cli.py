"""CLI module for WildTrain using Typer and Rich."""

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text
import json
import logging
from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf, DictConfig
from datetime import datetime


from ..trainers.classification_trainer import ClassifierTrainer
from ..trainers.detection_trainer import UltralyticsDetectionTrainer
from ..utils.logging import ROOT
from ..pipeline.detection_pipeline import DetectionPipeline
from ..pipeline.classification_pipeline import ClassificationPipeline
from ..visualization import add_predictions_from_classifier, add_predictions_from_detector
from ..evaluators.ultralytics import UltralyticsEvaluator
from ..evaluators.classification import ClassificationEvaluator
from ..models.detector import Detector
from ..models.localizer import UltralyticsLocalizer
from ..models.classifier import GenericClassifier
from .config_loader import ConfigLoader, ConfigValidationError, ConfigFileNotFoundError, ConfigParseError
from ..data.classification_datamodule import ClassificationDataModule, compute_dataset_stats

# Create Typer app
app = typer.Typer(
    name="wildtrain",
    help="Modular Computer Vision Framework for Detection and Classification",
    add_completion=False,
    rich_markup_mode="rich",
)

# Create Rich console
console = Console()


def setup_logging(verbose: bool = False, log_file: Optional[Path] = None) -> None:
    """Setup rich logging with appropriate level."""
    level = logging.DEBUG if verbose else logging.INFO

    handlers:list = [RichHandler(console=console, rich_tracebacks=True)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )


@app.callback()
def main(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    config_dir: Optional[Path] = typer.Option(
        None, "--config-dir", "-c", help="Configuration directory"
    ),
) -> None:
    """WildTrain - Modular Computer Vision Framework."""
    
    # Display welcome message
    welcome_text = Text("🚀 WildTrain", style="bold blue")
    subtitle = Text("Modular Computer Vision Framework", style="italic")

    console.print(
        Panel(welcome_text + "\n" + subtitle, border_style="blue", padding=(1, 2))
    )


@app.command()
def validate_config(
    config: Path = typer.Argument(..., help="Path to configuration file"),
    config_type: str = typer.Option(
        "classification", 
        "--type", 
        "-t", 
        help="Configuration type (classification, detection, classification_eval, detection_eval, classification_visualization, detection_visualization, pipeline)"
    ),
) -> None:
    """Validate a configuration file using Pydantic models."""
    console.print(f"[bold green]Validating {config_type} configuration:[/bold green] {config}")
    
    try:
        is_valid = ConfigLoader.validate_config_file(config, config_type)
        if is_valid:
            console.print(f"[bold green]✓[/bold green] Configuration is valid!")
        else:
            console.print(f"[bold red]✗[/bold red] Configuration validation failed!")
            raise typer.Exit(1)
    except (ConfigFileNotFoundError, ConfigParseError, ConfigValidationError) as e:
        console.print(f"[bold red]✗[/bold red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Unexpected error: {str(e)}")
        raise typer.Exit(1)


@app.command()
def train_classifier(
    config: Path = typer.Option("", help="Path to training configuration file"),
    template: bool = typer.Option(False, "--template", "-t", help="Show default configuration template instead of training")
) -> None:
    """Train a classification model."""
    if template:
        console.print(f"[bold green]Showing default classification configuration template...[/bold green]")
        try:
            template_yaml = ConfigLoader.generate_default_config("classification")
            console.print(f"\n[bold blue]Default classification configuration template:[/bold blue]")
            console.print(f"\n```yaml\n{template_yaml}```")
            return
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Failed to generate template: {str(e)}")
            raise typer.Exit(1)
    
    console.print(f"[bold green]Training classifier with config:[/bold green] {config}")
    log_file = ROOT / "logs" / "train_classifier" / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    setup_logging(log_file=log_file)

    try:
        # Load and validate configuration using Pydantic
        validated_config = ConfigLoader.load_classification_config(config)
        console.print(f"[bold green]✓[/bold green] Configuration validated successfully")
        
        # Convert validated config back to DictConfig for backward compatibility
        cfg = OmegaConf.create(validated_config.model_dump())
        console.print(OmegaConf.to_yaml(cfg))
        
        ClassifierTrainer(DictConfig(cfg)).run()
        
    except (ConfigFileNotFoundError, ConfigParseError, ConfigValidationError) as e:
        console.print(f"[bold red]✗[/bold red] Configuration error: {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Training failed: {str(e)}")
        raise typer.Exit(1)

@app.command()
def train_detector(
    config: Path = typer.Option(..., help="Path to training configuration file"),
    template: bool = typer.Option(False, "--template", "-t", help="Show default configuration template instead of training")
) -> None:
    """Train a detection model."""
    if template:
        console.print(f"[bold green]Showing default detection configuration template...[/bold green]")
        try:
            template_yaml = ConfigLoader.generate_default_config("detection")
            console.print(f"\n[bold blue]Default detection configuration template:[/bold blue]")
            console.print(f"\n```yaml\n{template_yaml}```")
            return
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Failed to generate template: {str(e)}")
            raise typer.Exit(1)
    
    console.print(f"[bold green]Training detector with config:[/bold green] {config}")
    log_file = ROOT / "logs" / "train_detector" / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    setup_logging(log_file=log_file)

    try:
        # Load and validate configuration using Pydantic
        validated_config = ConfigLoader.load_detection_config(config)
        console.print(f"[bold green]✓[/bold green] Configuration validated successfully")
        
        # Convert validated config back to DictConfig for backward compatibility
        cfg = OmegaConf.create(validated_config.model_dump())
        console.print(OmegaConf.to_yaml(cfg))
        
        UltralyticsDetectionTrainer(DictConfig(cfg)).run()
        
    except (ConfigFileNotFoundError, ConfigParseError, ConfigValidationError) as e:
        console.print(f"[bold red]✗[/bold red] Configuration error: {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Training failed: {str(e)}")
        raise typer.Exit(1)

@app.command()
def get_dataset_stats(
    data_dir: Path = typer.Argument(..., help="Path to dataset directory"),
    split: str = typer.Option("train", help="Split to compute statistics for"),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for statistics"
    ),
) -> None:
    """Get dataset statistics and information (mean, std)."""
    

    console.print(f"[bold green]Analyzing dataset at:[/bold green] {data_dir}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Computing dataset statistics...", total=None)

        # Create the data module and load data
        datamodule = ClassificationDataModule(
            root_data_directory=str(data_dir), batch_size=32, transforms=None, load_as_single_class=True
        )
        if split == "train":
            datamodule.setup(stage="fit")
            data = datamodule.train_dataset
        elif split == "val":
            datamodule.setup(stage="validate")
            data = datamodule.val_dataset
        elif split == "test":
            datamodule.setup(stage="test")
            data = datamodule.test_dataset
        else:
            raise ValueError(f"Invalid split: {split}")

        mean, std = compute_dataset_stats(
            data,
            batch_size=32,
            num_workers=0,
        )

        stats = {
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

        progress.update(task, description="Dataset analysis completed!")

        console.print("\n[bold blue]Dataset Statistics:[/bold blue]")
        console.print(f"  📊 Mean: {stats['mean']}")
        console.print(f"  📏 Std: {stats['std']}")

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            
            console.print(f"  💾 Statistics saved to: {output_file}")
        
        return stats


@app.command()
def run_detection_pipeline(
    config: Path = typer.Option("", help="Path to unified detection pipeline YAML config"),
    template: bool = typer.Option(False, "--template", "-t", help="Show default configuration template instead of running pipeline")
) -> None:
    """Run the full detection pipeline (train + eval) for object detection."""
    if template:
        console.print(f"[bold green]Showing default detection pipeline configuration template...[/bold green]")
        try:
            template_yaml = ConfigLoader.generate_default_config("detection_pipeline")
            console.print(f"\n[bold blue]Default detection pipeline configuration template:[/bold blue]")
            console.print(f"\n```yaml\n{template_yaml}```")
            return
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Failed to generate template: {str(e)}")
            raise typer.Exit(1)
    
    console.print(f"[bold green]Running detection pipeline with config:[/bold green] {config}")

    log_file = ROOT / "logs" / "run_detection_pipeline" / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    setup_logging(log_file=log_file)

    try:
        # Load and validate configuration using Pydantic
        validated_config = ConfigLoader.load_pipeline_config(config, pipeline_type="detection")
        console.print(f"[bold green]✓[/bold green] Detection pipeline configuration validated successfully")
        
        # Convert validated config back to DictConfig for backward compatibility
        cfg = OmegaConf.create(validated_config.model_dump())
        console.print(OmegaConf.to_yaml(cfg))
        
        pipeline = DetectionPipeline(str(config))
        results = pipeline.run()
        console.print("\n[bold blue]Detection pipeline completed. Evaluation results:[/bold blue]")
        console.print(results)
        
    except (ConfigFileNotFoundError, ConfigParseError, ConfigValidationError) as e:
        console.print(f"[bold red]✗[/bold red] Configuration error: {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Pipeline failed: {str(e)}")
        raise typer.Exit(1)


@app.command()
def run_classification_pipeline(
    config: Path = typer.Option("", help="Path to unified classification pipeline YAML config"),
    template: bool = typer.Option(False, "--template", "-t", help="Show default configuration template instead of running pipeline")
) -> None:
    """Run the full classification pipeline (train + eval) for image classification."""
    if template:
        console.print(f"[bold green]Showing default classification pipeline configuration template...[/bold green]")
        try:
            template_yaml = ConfigLoader.generate_default_config("classification_pipeline")
            console.print(f"\n[bold blue]Default classification pipeline configuration template:[/bold blue]")
            console.print(f"\n```yaml\n{template_yaml}```")
            return
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Failed to generate template: {str(e)}")
            raise typer.Exit(1)
    
    console.print(f"[bold green]Running classification pipeline with config:[/bold green] {config}")

    log_file = ROOT / "logs" / "run_classification_pipeline" / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    setup_logging(log_file=log_file)

    try:
        # Load and validate configuration using Pydantic
        validated_config = ConfigLoader.load_pipeline_config(config, pipeline_type="classification")
        console.print(f"[bold green]✓[/bold green] Classification pipeline configuration validated successfully")
        
        # Convert validated config back to DictConfig for backward compatibility
        cfg = OmegaConf.create(validated_config.model_dump())
        console.print(OmegaConf.to_yaml(cfg))
        
        pipeline = ClassificationPipeline(str(config))
        results = pipeline.run()
        console.print("\n[bold blue]Classification pipeline completed. Evaluation results:[/bold blue]")
        console.print(results)
        
    except (ConfigFileNotFoundError, ConfigParseError, ConfigValidationError) as e:
        console.print(f"[bold red]✗[/bold red] Configuration error: {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Pipeline failed: {str(e)}")
        raise typer.Exit(1)


@app.command()
def visualize_classifier_predictions(
    config: Path = typer.Option("", help="Path to classification visualization configuration YAML file"),
    template: bool = typer.Option(False, "--template", "-t", help="Show default configuration template instead of visualization")
) -> None:
    """Upload classifier predictions to a FiftyOne dataset for visualization using YAML configuration."""
    if template:
        console.print(f"[bold green]Showing default classification visualization configuration template...[/bold green]")
        try:
            template_yaml = ConfigLoader.generate_default_config("classification_visualization")
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
        console.print(OmegaConf.to_yaml(cfg))
        
        # Extract configuration values
        dataset_name = cfg.dataset_name
        checkpoint_path = cfg.checkpoint_path
        prediction_field = cfg.prediction_field
        batch_size = cfg.batch_size
        device = cfg.device
        debug = cfg.debug
        
        console.print(f"[bold green]Uploading classifier predictions to FiftyOne dataset:[/bold green] {dataset_name}")
        
        log_file = ROOT / "logs" / "visualize_classifier_predictions" / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
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


@app.command()
def visualize_detector_predictions(
    config: Path = typer.Option("", help="Path to visualization configuration YAML file"),
    template: bool = typer.Option(False, "--template", "-t", help="Show default configuration template instead of visualization")
) -> None:
    """Upload detector predictions to a FiftyOne dataset for visualization using YAML configuration."""
    if template:
        console.print(f"[bold green]Showing default detection visualization configuration template...[/bold green]")
        try:
            template_yaml = ConfigLoader.generate_default_config("detection_visualization")
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
        console.print(OmegaConf.to_yaml(cfg))
        
        # Extract configuration values
        dataset_name = cfg.fiftyone.dataset_name
        prediction_field = cfg.fiftyone.prediction_field
        
        localizer_cfg = cfg.model.localizer
        classifier_cfg = cfg.model.classifier
        processing_cfg = cfg.processing
        
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
        
        log_file = ROOT / "logs" / "visualize_detector_predictions" / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        setup_logging(log_file=log_file)

        add_predictions_from_detector(
            dataset_name=dataset_name,
            detector=detector,
            imgsz=localizer_cfg.imgsz,
            prediction_field=prediction_field,
            batch_size=processing_cfg.batch_size,
            debug=processing_cfg.debug,
        )
        console.print(f"[bold blue]Detector predictions uploaded to FiftyOne dataset:[/bold blue] {dataset_name}")
        
    except (ConfigFileNotFoundError, ConfigParseError, ConfigValidationError) as e:
        console.print(f"[bold red]✗[/bold red] Configuration error: {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Visualization failed: {str(e)}")
        raise typer.Exit(1)


@app.command()
def evaluate_detector(
    config: Path = typer.Option("", help="Path to YOLO evaluation YAML config file"),
    model_type: str = typer.Option("yolo", "--type", "-t", help="Type of detector to evaluate (yolo, yolo_v8, yolo_v11)"),
    template: bool = typer.Option(False, "--template", help="Show default configuration template instead of evaluation")
) -> None:
    """Evaluate a YOLO model using a YAML config file."""
    if template:
        console.print(f"[bold green]Showing default detection evaluation configuration template...[/bold green]")
        try:
            template_yaml = ConfigLoader.generate_default_config("detection_eval")
            console.print(f"\n[bold blue]Default detection evaluation configuration template:[/bold blue]")
            console.print(f"\n```yaml\n{template_yaml}```")
            return
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Failed to generate template: {str(e)}")
            raise typer.Exit(1)
    
    console.print(f"[bold green]Running {model_type} evaluation with config:[/bold green] {config}")

    log_file = ROOT / "logs" / "evaluate_detector" / f"evaluate_detector_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    setup_logging(log_file=log_file)

    try:
        # Load and validate configuration using Pydantic
        validated_config = ConfigLoader.load_detection_eval_config(config)
        console.print(f"[bold green]✓[/bold green] Detection evaluation configuration validated successfully")
        
        # Convert validated config back to DictConfig for backward compatibility
        cfg = OmegaConf.create(validated_config.model_dump())
        console.print(OmegaConf.to_yaml(cfg))
        
        if model_type == "yolo":
            evaluator = UltralyticsEvaluator(config=str(config))
        else:
            raise ValueError(f"Invalid detector type: {model_type}")
        results = evaluator.evaluate(debug=False)
        console.print(f"\n[bold blue]{model_type} Evaluation Results:[/bold blue]")
        console.print(results)
        
    except (ConfigFileNotFoundError, ConfigParseError, ConfigValidationError) as e:
        console.print(f"[bold red]✗[/bold red] Configuration error: {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Evaluation failed: {str(e)}")
        raise typer.Exit(1)


@app.command()
def evaluate_classifier(
    config: Path = typer.Option("", help="Path to classification evaluation YAML config file"),
    template: bool = typer.Option(False, "--template", "-t", help="Show default configuration template instead of evaluation")
) -> None:
    """Evaluate a classifier using a YAML config file."""
    if template:
        console.print(f"[bold green]Showing default classification evaluation configuration template...[/bold green]")
        try:
            template_yaml = ConfigLoader.generate_default_config("classification_eval")
            console.print(f"\n[bold blue]Default classification evaluation configuration template:[/bold blue]")
            console.print(f"\n```yaml\n{template_yaml}```")
            return
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Failed to generate template: {str(e)}")
            raise typer.Exit(1)
    
    console.print(f"[bold green]Running classifier evaluation with config:[/bold green] {config}")

    log_file = ROOT / "logs" / "evaluate_classifier" / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    setup_logging(log_file=log_file)

    try:
        # Load and validate configuration using Pydantic
        validated_config = ConfigLoader.load_classification_eval_config(config)
        console.print(f"[bold green]✓[/bold green] Classification evaluation configuration validated successfully")
        
        # Convert validated config back to DictConfig for backward compatibility
        cfg = OmegaConf.create(validated_config.model_dump())
        console.print(OmegaConf.to_yaml(cfg))
        
        evaluator = ClassificationEvaluator(str(config))
        results = evaluator.evaluate(debug=False)
        console.print("\n[bold blue]Classifier Evaluation Results:[/bold blue]")
        console.print(results)
        
    except (ConfigFileNotFoundError, ConfigParseError, ConfigValidationError) as e:
        console.print(f"[bold red]✗[/bold red] Configuration error: {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Evaluation failed: {str(e)}")
        raise typer.Exit(1)


@app.command()
def show_config_template(
    config_type: str = typer.Argument(..., help="Configuration type to show template for"),
    save_to: Optional[Path] = typer.Option(None, "--save", "-s", help="Save template to file")
) -> None:
    """Show a default YAML configuration template for the specified config type."""
    console.print(f"[bold green]Generating {config_type} configuration template...[/bold green]")
    
    try:
        template = ConfigLoader.generate_default_config(config_type)
        
        if save_to:
            save_to.parent.mkdir(parents=True, exist_ok=True)
            with open(save_to, 'w', encoding='utf-8') as f:
                f.write(template)
            console.print(f"[bold green]✓[/bold green] Template saved to: {save_to}")
        else:
            console.print(f"\n[bold blue]Default {config_type} configuration template:[/bold blue]")
            console.print(f"\n```yaml\n{template}```")
            
    except ValueError as e:
        console.print(f"[bold red]✗[/bold red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Failed to generate template: {str(e)}")
        raise typer.Exit(1)


