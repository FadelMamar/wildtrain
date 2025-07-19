import traceback
import os
from omegaconf import DictConfig
from omegaconf import OmegaConf
from typing import Any
import mlflow
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger
from pathlib import Path
import torch
from torchvision.transforms import v2
from wildata.pipeline import PathManager
from dotenv import load_dotenv

from ..models.classifier import Classifier,GenericClassifier
from ..data.classification_datamodule import ClassificationDataModule
from ..utils.logging import ROOT
from ..utils.logging import get_logger
from ..utils.dvc_tracker import DVCTracker

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
    from torchvision import transforms as T
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
                
                transform_name = config.get('name', config.get('type'))
                params = config.get('params', config.get('kwargs', {}))
                
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
                    raise ValueError(f"Transform config missing 'name' or 'type' key: {config}")
            else:
                raise ValueError(f"Invalid transform config: {config}")
            
            # Get the transform class from torchvision
            if hasattr(T, transform_name):
                transform_class = getattr(T, transform_name)
            elif hasattr(F, transform_name):
                # For functional transforms, we need to handle them differently
                raise ValueError(f"Functional transforms like {transform_name} are not supported yet")
            else:
                raise ValueError(f"Unknown transform: {transform_name}")
            
            # Create the transform instance with parameters
            try:
                transform_instance = transform_class(**params)
                transform_list.append(transform_instance)
            except Exception as e:
                raise ValueError(f"Failed to create transform {transform_name} with params {params}: {e}")
                
        return T.Compose(transform_list)
    
    result = {}
    
    # Create train transforms
    if 'train' in transforms:
        result['train'] = create_transform_list(transforms['train'])
    else:
        # Default train transforms
        result['train'] = T.Compose([T.ToTensor()])
    
    # Create validation transforms
    if 'val' in transforms:
        result['val'] = create_transform_list(transforms['val'])
    else:
        # Default validation transforms
        result['val'] = T.Compose([T.ToTensor()])
    
    return result

def track_dataset(cfg: DictConfig) -> None:
    """
    Track the dataset with DVC and log the dataset information to MLflow.
    """
    dvc_tracker = DVCTracker(ROOT/"dvc")
    path_manager = PathManager(cfg.dataset.root_data_directory)
    dataset_path = path_manager.get_framework_dir("roi").resolve()
    dataset_name = cfg.get('dataset_name', dataset_path.name)
    try:
        dvc_tracker.track_dataset_for_training(dataset_path, dataset_name,link_to_mlflow=True)
    except Exception:
        logger.warning(f"Failed to track dataset with DVC: {traceback.format_exc()}")

def run_classification(cfg: DictConfig) -> None:
    """
    Run image classification training or evaluation based on config.
    """
    load_dotenv(ROOT/".env",override=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    # DataModule
    data_cfg = cfg.dataset
    batch_size = cfg.train.batch_size
    
    datamodule = ClassificationDataModule(
        root_data_directory=data_cfg.root_data_directory,
        batch_size=batch_size,
        transforms=create_transforms(cfg.dataset.transforms)
    )
    datamodule.setup(stage='fit')
    num_classes = len(datamodule.class_mapping)
    example_input,_ = next(iter(datamodule.train_dataloader()))

    # Model
    model_cfg = cfg.model
    cls_model = GenericClassifier(num_classes=num_classes, 
    backbone=model_cfg.backbone, 
    pretrained= model_cfg.pretrained, 
    backbone_source=model_cfg.backbone_source,
    label_to_class_map=datamodule.class_mapping,
    dropout=model_cfg.dropout,
    freeze_backbone=model_cfg.freeze_backbone
    )
    model = Classifier(epochs=cfg.train.epochs, 
                        model=cls_model, lr=cfg.train.lr,
                        threshold=cfg.train.threshold, 
                        label_smoothing=cfg.train.label_smoothing, 
                        weight_decay=cfg.train.weight_decay,
                        lrf=cfg.train.lrf
                                )
    model.example_input_array = example_input

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.checkpoint.monitor,
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=cfg.checkpoint.save_last,
        mode=cfg.checkpoint.mode,
        dirpath=ROOT / cfg.checkpoint.dirpath,
        filename=cfg.checkpoint.filename,
        save_weights_only=cfg.checkpoint.save_weights_only
    )
    lr_callback = LearningRateMonitor(logging_interval="epoch")
    early_stopping = EarlyStopping(
        monitor=cfg.checkpoint.monitor,
        patience=cfg.checkpoint.patience,
        mode=cfg.checkpoint.mode,
        min_delta=cfg.checkpoint.min_delta,
    )
    mlf_logger = MLFlowLogger(
            experiment_name=cfg.mlflow.experiment_name,
            run_name=cfg.mlflow.run_name,
            log_model=False,
            checkpoint_path_prefix="classification",
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        )

    trainer = Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=cfg.train.accelerator,
        precision=cfg.train.precision,
        logger=mlf_logger,
        limit_train_batches=10,
        limit_val_batches=10,
        callbacks=[checkpoint_callback, early_stopping, lr_callback],
    )

    if cfg.mode == 'train':
        trainer.fit(model, datamodule=datamodule)
    elif cfg.mode == 'evaluate':
        trainer.validate(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}") 
    
    if model.mlflow_run_id and cfg.mlflow.log_model:
        try:
            with mlflow.start_run(run_id=model.mlflow_run_id):
                ckpt = ROOT / cfg.checkpoint.dirpath / "best.ckpt"
                best_model = Classifier.load_from_checkpoint(ckpt,model=model.model).model.to("cpu")
                path = Path(ckpt).with_suffix(".pt")
                torch.save(best_model, path)
                mlflow.log_artifact(str(path), "checkpoint")

                cfg_path = Path(ckpt).with_name("config.yaml")
                OmegaConf.save(cfg, cfg_path)
                mlflow.log_artifact(str(cfg_path), "config")
                

                #path = Path(ckpt).with_suffix(".torchscript")
                #best_model.to_torchscript(file_path=path, example_inputs=example_input)
                #mlflow.log_artifact(str(path), "checkpoint")

                if cfg.get('track_dataset'):
                    track_dataset(cfg)

        except Exception as e:
            logger.error(f"Error logging model: {e}")







