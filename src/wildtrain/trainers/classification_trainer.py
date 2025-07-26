import traceback
import os
from omegaconf import DictConfig
from omegaconf import OmegaConf
from typing import Any, Sequence
import mlflow
from lightning import Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import MLFlowLogger
from pathlib import Path
import torch
from wildata.pipeline import PathManager
from dotenv import load_dotenv
import lightning as L
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC
from typing import Any, Optional, Tuple

from ..models.classifier import GenericClassifier
from ..data import ClassificationDataModule
from ..utils.logging import ROOT, ENV_FILE
from ..utils.logging import get_logger
from ..utils.dvc_tracker import DVCTracker
from .base import ModelTrainer

logger = get_logger(__name__)


def create_transforms(transforms: dict[str, Any]) -> dict[str, Any]:
    """
    Create transforms for training and validation from a dictionary.

    Args:
        transforms: Dictionary with 'train' and 'val' keys, each containing a list of
                   transformation configurations. Each config should have:
                   - 'name': torchvision transformation class name (e.g., 'RandomResizedCrop')
                   - 'params': dictionary of parameters for the transformation

    Returns:
        Dictionary with 'train' and 'val' keys containing composed transforms
    """
    import torchvision.transforms.v2 as T
    import torchvision.transforms.functional as F

    def create_transform_list(transform_configs: list) -> T.Compose:
        """Create a list of transforms from configuration."""
        transform_list = []

        for config in transform_configs:
            if isinstance(config, str):
                # Simple case: just the transform name
                transform_name = config
                params = {}
            elif isinstance(config, (dict, DictConfig)):
                # Full configuration with parameters - convert DictConfig to dict if needed
                if isinstance(config, DictConfig):
                    config = dict(config)

                transform_name = config.get("name", config.get("type"))
                params = config.get("params", config.get("kwargs", {}))

                # Convert DictConfig params to regular dict if needed
                if isinstance(params, DictConfig):
                    params = dict(params)

                # Ensure params is a proper dictionary with string keys
                if not isinstance(params, dict):
                    params = {}
                else:
                    # Convert any non-string keys to strings
                    params = {str(k): v for k, v in params.items()}

                if transform_name is None:
                    raise ValueError(
                        f"Transform config missing 'name' or 'type' key: {config}"
                    )
            else:
                raise ValueError(f"Invalid transform config: {config}")

            # Get the transform class from torchvision
            if hasattr(T, transform_name):
                transform_class = getattr(T, transform_name)
            elif hasattr(F, transform_name):
                # For functional transforms, we need to handle them differently
                raise ValueError(
                    f"Functional transforms like {transform_name} are not supported yet"
                )
            else:
                raise ValueError(f"Unknown transform: {transform_name}")

            # Create the transform instance with parameters
            for key, value in params.items():
                if isinstance(value, Sequence) and not isinstance(value, str):
                    params[key] = list(value)
            try:
                transform_instance = transform_class(**params)
                transform_list.append(transform_instance)
            except Exception as e:
                raise ValueError(
                    f"Failed to create transform {transform_name} with params {params}: {traceback.format_exc()}"
                )

        return T.Compose(transform_list)

    result = {}

    # Create train transforms
    if "train" in transforms:
        result["train"] = create_transform_list(transforms["train"])
    else:
        # Default train transforms
        result["train"] = T.Compose([T.ToTensor()])

    # Create validation transforms
    if "val" in transforms:
        result["val"] = create_transform_list(transforms["val"])
    else:
        # Default validation transforms
        result["val"] = T.Compose([T.ToTensor()])

    return result


def track_dataset(cfg: DictConfig) -> None:
    """
    Track the dataset with DVC and log the dataset information to MLflow.
    """
    dvc_tracker = DVCTracker(ROOT / "dvc")
    path_manager = PathManager(cfg.dataset.root_data_directory)
    dataset_path = path_manager.get_framework_dir("roi").resolve()
    dataset_name = cfg.get("dataset_name", dataset_path.name)
    try:
        dvc_tracker.track_dataset_for_training(
            dataset_path, dataset_name, link_to_mlflow=True
        )
    except Exception:
        logger.warning(f"Failed to track dataset with DVC: {traceback.format_exc()}")


class ClassifierModule(L.LightningModule):
    def __init__(
        self,
        model: GenericClassifier,
        epochs: int = 20,
        label_smoothing: float = 0.0,
        lr: float = 1e-3,
        lrf: float = 1e-2,
        weight_decay: float = 5e-3,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.num_classes = model.num_classes.item()
        self.label_to_class_map = model.label_to_class_map
        
        self.mlflow_run_id = None
        self.mlflow_experiment_id = None

        # metrics
        cfg = dict(task="multiclass", num_classes=self.num_classes, average=None)
        self.accuracy = Accuracy(**cfg)
        self.precision = Precision(**cfg)
        self.recall = Recall(**cfg)
        self.f1score = F1Score(**cfg)
        self.ap = AUROC(**cfg)

        self.metrics = dict(
            accuracy=self.accuracy,
            precision=self.precision,
            recall=self.recall,
            f1score=self.f1score,
        )

        self.label_smoothing = label_smoothing            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch

        classes = y.cpu().flatten().tolist()
        weight = [
            len(classes) / (classes.count(i) + 1e-6) for i in range(self.num_classes)
        ]
        weight = torch.Tensor(weight).float().clamp(1.0, 10.).to(y.device)

        logits = self(x)
        loss = F.cross_entropy(
            logits,
            y.long().squeeze(1),
            label_smoothing=self.label_smoothing,
            weight=weight,
        )

        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y = y.long().squeeze(1)

        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)

        for name, metric in self.metrics.items():
            metric.update(logits.softmax(dim=1), y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        if self.mlflow_run_id is None:
            self.mlflow_run_id = getattr(self.logger, "run_id", None)
            self.mlflow_experiment_id = getattr(self.logger, "experiment_id", None)
            if self.mlflow_run_id is not None:
                logger.info(f"MLflow run_id: {self.mlflow_run_id}")
                logger.info(f"MLflow experiment_id: {self.mlflow_experiment_id}")
            else:
                logger.warning("No mlflow logger found")
                self.mlflow_run_id = "None"

        for name, metric in self.metrics.items():
            score = metric.compute().cpu()
            self.log(f"val_{name}", score.mean())
            for i, score in enumerate(score):
                cls_name = self.label_to_class_map.get(i, i)
                self.log(f"val_{name}_class_{cls_name}", score)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.hparams.epochs,
            T_mult=1,
            eta_min=self.hparams.lr * self.hparams.lrf,
        )
        return [optimizer], [lr_scheduler]


class ClassifierTrainer(ModelTrainer):
    """
    Trainer class for image classification models.

    This class handles the training and evaluation of classification models
    using PyTorch Lightning and MLflow for experiment tracking.
    """

    def get_callbacks(self) -> list[Any]:
        """
        Get the callbacks for the trainer.
        """
        assert ENV_FILE.exists(), "Environment file not found"
        load_dotenv(ENV_FILE, override=True)
        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
        if MLFLOW_TRACKING_URI is None:
            raise ValueError("MLFLOW_TRACKING_URI is not set")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor=self.config.checkpoint.monitor,
            save_top_k=self.config.checkpoint.save_top_k,
            save_last=self.config.checkpoint.save_last,
            mode=self.config.checkpoint.mode,
            dirpath=ROOT / self.config.checkpoint.dirpath,
            #filename=self.config.checkpoint.filename,
            save_weights_only=self.config.checkpoint.save_weights_only,
        )
        lr_callback = LearningRateMonitor(logging_interval="epoch")
        early_stopping = EarlyStopping(
            monitor=self.config.checkpoint.monitor,
            patience=self.config.checkpoint.patience,
            mode=self.config.checkpoint.mode,
            min_delta=self.config.checkpoint.min_delta,
        )
        mlflow_logger = MLFlowLogger(
            experiment_name=self.config.mlflow.experiment_name,
            run_name=self.config.mlflow.run_name,
            log_model=False,
            #checkpoint_path_prefix="classification",
            tracking_uri=MLFLOW_TRACKING_URI,
        )
        return [checkpoint_callback, early_stopping, lr_callback, mlflow_logger]

    def log_model(self, model: ClassifierModule) -> None:
        if model.mlflow_run_id:
            try:
                with mlflow.start_run(run_id=model.mlflow_run_id):
                    try:
                        ckpt = ROOT / self.config.checkpoint.dirpath / "best.ckpt"
                        best_model = GenericClassifier.load_from_lightning_ckpt(ckpt,map_location=model.device)
                        path = Path(ckpt).with_name("best_classifier.pt")
                        torch.save(best_model, path)
                        mlflow.log_artifact(str(path), "checkpoint")
                        self.best_model_path = str(path)
                    except Exception:
                        logger.warning(f"Error loading best model: {traceback.format_exc()}")

                    cfg_path = Path(ckpt).with_name("config.yaml")
                    OmegaConf.save(self.config, cfg_path)
                    mlflow.log_artifact(str(cfg_path), "config")

                    if self.config.get("track_dataset"):
                        track_dataset(self.config)

            except Exception:
                logger.error(f"Error logging model: {traceback.format_exc()}")

    def run(self, debug: bool = False) -> None:
        """
        Run image classification training or evaluation based on config.
        """
        # Set float32 matmul precision to medium
        if torch.cuda.is_available():   
            torch.set_float32_matmul_precision('medium')

        # DataModule
        data_cfg = self.config.dataset
        batch_size = self.config.train.batch_size

        datamodule = ClassificationDataModule(
            root_data_directory=data_cfg.root_data_directory,
            batch_size=batch_size,
            transforms=create_transforms(self.config.dataset.transforms),
            load_as_single_class=self.config.dataset.single_class.enable,
            background_class_name=self.config.dataset.single_class.background_class_name,
            single_class_name=self.config.dataset.single_class.single_class_name,
            keep_classes=self.config.dataset.single_class.keep_classes,
            discard_classes=self.config.dataset.single_class.discard_classes,
            rebalance=data_cfg.rebalance,
        )
        logger.info(f"Getting one batch of data to initialize lazy modules in classifier.")
        datamodule.setup(stage="fit")
        example_input, _ = next(iter(datamodule.train_dataloader()))

        # Model
        model_cfg = self.config.model
        cls_model = GenericClassifier(
            backbone=model_cfg.backbone,
            pretrained=model_cfg.pretrained,
            backbone_source=model_cfg.backbone_source,
            label_to_class_map=datamodule.class_mapping,
            dropout=model_cfg.dropout,
            freeze_backbone=model_cfg.freeze_backbone,
            input_size=model_cfg.input_size,
            mean=model_cfg.mean,
            std=model_cfg.std,
        )
        model = ClassifierModule(
            epochs=self.config.train.epochs,
            model=cls_model,
            lr=self.config.train.lr,
            label_smoothing=self.config.train.label_smoothing,
            weight_decay=self.config.train.weight_decay,
            lrf=self.config.train.lrf,
        )
        model.example_input_array = example_input
        callbacks = self.get_callbacks()
        mlflow_logger = callbacks.pop(-1)
        trainer = Trainer(
            max_epochs=self.config.train.epochs if not debug else 1,
            accelerator=self.config.train.accelerator,
            precision=self.config.train.precision,
            logger=mlflow_logger,
            limit_train_batches=3 if debug else None,
            limit_val_batches=3 if debug else None,
            callbacks=callbacks,
        )

        trainer.fit(model, datamodule=datamodule)
    
        #if self.config.mlflow.log_model:
        #    self.log_model(model)
