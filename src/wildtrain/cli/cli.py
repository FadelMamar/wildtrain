"""CLI module for WildTrain using Typer and Rich."""

import typer
from pathlib import Path
from typing import Optional

from .commands.utils import display_welcome_message
from .commands import config, train, evaluate, register, pipeline, visualize, dataset

# Create Typer app
app = typer.Typer(
    name="wildtrain",
    help="Modular Computer Vision Framework for Detection and Classification",
    add_completion=False,
    rich_markup_mode="rich",
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
    display_welcome_message()


# Add command groups
app.add_typer(config.config_app, name="config", help="Configuration management")
app.add_typer(train.train_app, name="train", help="Training commands")
app.add_typer(evaluate.evaluate_app, name="evaluate", help="Evaluation commands")
app.add_typer(register.register_app, name="register", help="Model registration commands")
app.add_typer(pipeline.pipeline_app, name="pipeline", help="Pipeline commands")
app.add_typer(visualize.visualize_app, name="visualize", help="Visualization commands")
app.add_typer(dataset.dataset_app, name="dataset", help="Dataset commands")