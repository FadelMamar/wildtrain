from typing import Any, Dict, Optional, List
from omegaconf import DictConfig, OmegaConf
import mlflow
from ultralytics import YOLO
from ultralytics import settings

from ..utils.logging import ROOT, get_logger
from .base import ModelTrainer

logger = get_logger(__name__)



class UltralyticsDetectionTrainer(ModelTrainer):
    """
    Trainer class for object detection models using Ultralytics YOLO.
    This class handles training using parameters from a DictConfig (e.g., from yolo.yaml).
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.model: Optional[YOLO] = None

    def validate_config(self) -> None:
        if (
            self.config.model.architecture_file is None
            and self.config.model.weights is None
        ):
            raise ValueError("Either architecture_file or weights must be provided")

        if self.config.dataset.data_cfg is None:
            raise ValueError("data_cfg must be provided")

    def run(self,debug:bool=False) -> None:
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        

        settings.update({"mlflow": True})

        self.validate_config()

        # Load model
        self.model = YOLO(
            self.config.model.architecture_file or self.config.model.weights,
            task="detect",
        )
        if self.config.model.weights is not None:
            self.model.load(self.config.model.weights)

        # Training parameters
        train_cfg = self.config.train

        # Run training
        self.model.train(
            single_cls=self.config.dataset.load_as_single_class,
            data=self.config.dataset.data_cfg,
            name=self.config.name,
            project=self.config.project,
            time=5/60 if debug else None,
            save=True,
            **train_cfg,
        )
