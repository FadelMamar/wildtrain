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
from omegaconf import OmegaConf, DictConfig
import subprocess
import platform
from datetime import datetime


from .trainers.classification_trainer import ClassifierTrainer
from .utils.logging import ROOT
from .pipeline.detection_pipeline import DetectionPipeline
from .pipeline.classification_pipeline import ClassificationPipeline
from .visualization import add_predictions_from_classifier, add_predictions_from_detector
from .evaluators.ultralytics import UltralyticsEvaluator
from .evaluators.classification import ClassificationEvaluator
from .models.detector import Detector
from .models.localizer import UltralyticsLocalizer
from .models.classifier import GenericClassifier

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
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    handlers = [RichHandler(console=console, rich_tracebacks=True)]
    if log_file:
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
def train_classifier(
    config: Path = typer.Argument(..., help="Path to training configuration file"),
) -> None:
    """Train a classification model."""
    console.print(f"[bold green]Training classifier with config:[/bold green] {config}")
    log_file = ROOT / "logs" / f"train_classifier_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    setup_logging(log_file=log_file)

    cfg = OmegaConf.load(config)
    console.print(OmegaConf.to_yaml(cfg))
    ClassifierTrainer(DictConfig(cfg)).run()


@app.command()
def get_dataset_stats(
    data_dir: Path = typer.Argument(..., help="Path to dataset directory"),
    split: str = typer.Option("train", help="Split to compute statistics for"),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for statistics"
    ),
) -> None:
    """Get dataset statistics and information (mean, std)."""
    from wildtrain.data.classification_datamodule import ClassificationDataModule, compute_dataset_stats
    import json

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
            with open(output_file, "w") as f:
                json.dump(stats, f, indent=2)
            
            console.print(f"  💾 Statistics saved to: {output_file}")


@app.command()
def run_detection_pipeline(
    config: Path = typer.Argument(..., help="Path to unified detection pipeline YAML config"),
) -> None:
    """Run the full detection pipeline (train + eval) for object detection."""
    console.print(f"[bold green]Running detection pipeline with config:[/bold green] {config}")

    log_file = ROOT / "logs" / f"run_detection_pipeline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    setup_logging(log_file=log_file)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing pipeline...", total=None)
        pipeline = DetectionPipeline(str(config))
        progress.update(task, description="Training...")
        pipeline.train()
        progress.update(task, description="Evaluating...")
        results = pipeline.evaluate()
        progress.update(task, description="Pipeline completed!")
        console.print("\n[bold blue]Detection pipeline completed. Evaluation results:[/bold blue]")
        console.print(results)


@app.command()
def run_classification_pipeline(
    config: Path = typer.Argument(..., help="Path to unified classification pipeline YAML config"),
) -> None:
    """Run the full classification pipeline (train + eval) for image classification."""
    console.print(f"[bold green]Running classification pipeline with config:[/bold green] {config}")

    log_file = ROOT / "logs" / f"run_classification_pipeline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    setup_logging(log_file=log_file)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing pipeline...", total=None)
        pipeline = ClassificationPipeline(str(config))
        progress.update(task, description="Training...")
        pipeline.train()
        progress.update(task, description="Evaluating...")
        results = pipeline.evaluate()
        progress.update(task, description="Pipeline completed!")
        console.print("\n[bold blue]Classification pipeline completed. Evaluation results:[/bold blue]")
        console.print(results)


@app.command()
def visualize_classifier_predictions(
    dataset_name: str = typer.Argument(..., help="Name of the FiftyOne dataset to use or create"),
    checkpoint_path: Path = typer.Option(...,"--weights", help="Path to the classifier checkpoint (.ckpt) file"),
    prediction_field: str = typer.Option("classification_predictions", "--prediction-field", help="Field name to store predictions in FiftyOne samples"),
    batch_size: int = typer.Option(32, help="Batch size for prediction inference"),
    device: str = typer.Option("cpu", "--device", help="Device to run inference on (e.g., 'cpu' or 'cuda')"),
    debug: bool = typer.Option(False, "--debug", help="If set, only process a small number of samples for debugging")
) -> None:
    """Upload classifier predictions to a FiftyOne dataset for visualization."""
    console.print(f"[bold green]Uploading predictions to FiftyOne dataset:[/bold green] {dataset_name}")
    
    log_file = ROOT / "logs" / f"visualize_classifier_predictions_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    setup_logging(log_file=log_file)

    add_predictions_from_classifier(
        dataset_name=dataset_name,
        checkpoint_path=str(checkpoint_path),
        prediction_field=prediction_field,
        batch_size=batch_size,
        device=device,
        debug=debug,
    )
    console.print(f"[bold blue]Predictions uploaded to FiftyOne dataset:[/bold blue] {dataset_name}")


@app.command()
def visualize_detector_predictions(
    config: Path = typer.Argument(..., help="Path to visualization configuration YAML file"),
) -> None:
    """Upload detector predictions to a FiftyOne dataset for visualization using YAML configuration."""
    
    console.print(f"[bold green]Loading visualization config from:[/bold green] {config}")
    
    # Load configuration
    cfg = OmegaConf.load(config)
        
    console.print(OmegaConf.to_yaml(cfg))
    
    # Extract configuration values
    dataset_name = cfg.fiftyone.dataset_name
    prediction_field = cfg.fiftyone.prediction_field
    
    localizer_cfg = cfg.model.localizer
    classifier_cfg = cfg.model.classifier
    processing_cfg = cfg.processing
    
    console.print(f"[bold green]Uploading detector predictions to FiftyOne dataset:[/bold green] {dataset_name}")
    
    # Create localizer with config
    localizer = UltralyticsLocalizer(
        weights=localizer_cfg.weights,
        imgsz=localizer_cfg.imgsz,
        device=localizer_cfg.device,
        conf_thres=localizer_cfg.conf_thres,
        iou_thres=localizer_cfg.iou_thres,
        max_det=localizer_cfg.max_det,
        overlap_metric=localizer_cfg.overlap_metric
    )
    
    # Create classifier if checkpoint provided
    classifier = None
    if classifier_cfg.checkpoint is not None:
        console.print(f"[bold blue]Loading classifier from:[/bold blue] {classifier_cfg.checkpoint}")
        classifier = GenericClassifier.load_from_checkpoint(str(classifier_cfg.checkpoint))
    
    # Create detector
    detector = Detector(localizer=localizer, classifier=classifier)
    
    log_file = ROOT / "logs" / f"visualize_detector_predictions_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
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


@app.command()
def evaluate_detector(
    config: Path = typer.Argument(..., help="Path to YOLO evaluation YAML config file"),
    model_type: str = typer.Option("yolo", "--type", "-t", help="Type of detector to evaluate (yolo, yolo_v8, yolo_v11)"),
) -> None:
    """Evaluate a YOLO model using a YAML config file."""
    console.print(f"[bold green]Running {model_type} evaluation with config:[/bold green] {config}")

    log_file = ROOT / "logs" / f"evaluate_detector_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    setup_logging(log_file=log_file)

    if model_type == "yolo":
        evaluator = UltralyticsEvaluator(config=str(config))
    else:
        raise ValueError(f"Invalid detector type: {model_type}")
    results = evaluator.evaluate(debug=False)
    console.print(f"\n[bold blue]{type} Evaluation Results:[/bold blue]")
    console.print(results)


@app.command()
def evaluate_classifier(
    config: Path = typer.Argument(..., help="Path to classification evaluation YAML config file"),
) -> None:
    """Evaluate a classifier using a YAML config file."""
    console.print(f"[bold green]Running classifier evaluation with config:[/bold green] {config}")

    log_file = ROOT / "logs" / f"evaluate_classifier_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    setup_logging(log_file=log_file)

    evaluator = ClassificationEvaluator(str(config))
    results = evaluator.evaluate(debug=False)
    console.print("\n[bold blue]Classifier Evaluation Results:[/bold blue]")
    console.print(results)


if __name__ == "__main__":
    app()
