from abc import ABC, abstractmethod
from omegaconf import DictConfig
from typing import Any, Optional


class ModelTrainer(ABC):
    """
    Abstract base class for model trainers.

    This class defines the interface that all trainers must implement.
    Subclasses should implement the run method to handle specific training logic.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the trainer with configuration.

        Args:
            config: Configuration object containing all training parameters
        """
        self.config = config
        self.best_model_path: Optional[str] = None

    @abstractmethod
    def run(self) -> None:
        """
        Run the training or evaluation process.

        This method must be implemented by subclasses to handle the specific
        training logic for different model types (classification, detection, etc.).
        """
        pass

    def validate_config(self) -> None:
        """
        Validate the configuration before running.

        This method can be overridden by subclasses to add specific validation logic.
        """
        if not hasattr(self.config, "mode"):
            raise ValueError("Config must contain 'mode' field")

        if self.config.mode not in ["train", "evaluate"]:
            raise ValueError(
                f"Mode must be 'train' or 'evaluate', got {self.config.mode}"
            )
