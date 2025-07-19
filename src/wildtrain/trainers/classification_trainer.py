from omegaconf import DictConfig
from typing import Any
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger
from ..models.classifier import Classifier,GenericClassifier
from ..data.classification_datamodule import ClassificationDataModule



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
    from omegaconf import DictConfig
    
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
        
        # Always add ToTensor at the end if not already present
        if not any(isinstance(t, T.ToTensor) for t in transform_list):
            transform_list.append(T.ToTensor())
        
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

def run_classification(cfg: DictConfig) -> None:
    """
    Run image classification training or evaluation based on config.
    """
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

    # Model
    model_cfg = cfg.model
    cls_model = GenericClassifier(num_classes=num_classes, 
    backbone=model_cfg.backbone, 
    pretrained= model_cfg.pretrained, 
    label_to_class_map=datamodule.class_mapping,
    dropout=model_cfg.dropout,
    no_grad_backbone=model_cfg.no_grad_backbone
    )
    model = Classifier(epochs=cfg.train.epochs, 
                                    model=cls_model, lr=cfg.train.lr,
                                    threshold=cfg.train.threshold, 
                                    label_smoothing=cfg.train.label_smoothing, 
                                    weight_decay=cfg.train.weight_decay,
                                    lrf=cfg.train.lrf
                                )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.checkpoint.monitor,
        save_top_k=1,
        save_last=True,
        mode=cfg.checkpoint.mode,
        dirpath=cfg.checkpoint.dirpath,
        filename='best',
        save_weights_only=True
    )
    lr_callback = LearningRateMonitor(logging_interval="epoch")
    early_stopping = EarlyStopping(
        monitor=cfg.checkpoint.monitor,
        patience=cfg.checkpoint.patience,
        mode=cfg.checkpoint.mode,
        min_delta=1e-4,
    )
    mlf_logger = MLFlowLogger(
            experiment_name=cfg.mlflow.experiment_name,
            run_name=cfg.mlflow.run_name,
            tracking_uri=cfg.mlflow.tracking_uri,
            log_model=cfg.mlflow.log_model,
            checkpoint_path_prefix="checkpoints",
        )

    trainer = Trainer(
        max_epochs=cfg.train.epochs,
        accelerator='auto',
        precision="bf16-mixed",
        logger=mlf_logger,
        callbacks=[checkpoint_callback, early_stopping, lr_callback],
    )

    if cfg.mode == 'train':
        trainer.fit(model, datamodule=datamodule)
    elif cfg.mode == 'evaluate':
        trainer.validate(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}") 