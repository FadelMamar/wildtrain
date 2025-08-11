"""MLflow model registration utilities for WildTrain models."""

import os
import platform
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from enum import Enum

import numpy as np
import torch
import mlflow
from ultralytics import YOLO
from pydantic import BaseModel, Field

from wildtrain.models.classifier import GenericClassifier
from wildtrain.models.detector import Detector
from wildtrain.utils.logging import get_logger
from wildtrain.cli.config_loader import ConfigLoader


# Disable torch XNNPACK for compatibility
os.environ["TORCH_XNNPACK_DISABLE"] = "1"

logger = get_logger(__name__)


class ModelTask(Enum):
    """Supported model tasks."""
    DETECT = "detect"
    CLASSIFY = "classify"
    SEGMENT = "segment"


class ModelType(Enum):
    """Supported model types."""
    DETECTOR = "detector"
    CLASSIFIER = "classifier"
    LOCALIZER = "localizer"


class ModelMetadata(BaseModel):
    """Metadata for model registration.
    
    This class provides structured metadata for MLflow model registration,
    including validation and default values.
    """
    batch: int = Field(default=8, gt=0, description="Batch size for inference")
    imgsz: int = Field(default=800, gt=0, description="Input image size")
    task: ModelTask = Field(default=ModelTask.DETECT, description="Model task type")
    num_classes: Optional[int] = Field(default=None, ge=2, description="Number of classes")
    model_type: ModelType = Field(description="Type of model (detector or classifier)")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MLflow metadata."""
        return {
            "batch": self.batch,
            "imgsz": self.imgsz,
            "task": self.task.value,
            "num_classes": self.num_classes,
            "model_type": self.model_type.value
        }


def normalize_path(model_path: Path) -> Path:
    """Normalize path for cross-platform compatibility."""
    resolved_path = model_path.resolve()
    if platform.system().lower() != "windows":
        return Path(resolved_path.as_posix().replace("\\", "/"))
    return resolved_path


class UltralyticsWrapper(mlflow.pyfunc.PythonModel):
    """MLflow wrapper for YOLO detection models."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[YOLO] = None
        self.artifacts: Optional[Dict[str, Any]] = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the YOLO model from the artifacts."""
        model_path = Path(context.artifacts["localizer_ckpt"]).resolve()
        
        # Handle Windows path issues
        if platform.system().lower() != "windows":
            model_path = Path(normalize_path(model_path))
        
        self.model = YOLO(str(model_path), task="detect")
        self.artifacts = context.artifacts
    

class ClassifierWrapper(mlflow.pyfunc.PythonModel):
    """MLflow wrapper for classification models."""
    
    def __init__(self):
        super().__init__()

    def __getstate__(self):
        """Handle serialization - exclude model objects."""
        state = self.__dict__.copy()
        # Remove any model references that can't be serialized
        state.pop('model', None)
        state.pop('artifacts', None)
        return state

    def __setstate__(self, state):
        """Handle deserialization."""
        self.__dict__.update(state)
        # Model will be loaded in load_context

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the TorchScript classification model from the artifacts."""
        model_path = Path(context.artifacts["classifier_ckpt"]).resolve()
        
        # Handle Windows path issues
        if platform.system().lower() != "windows":
            model_path = Path(normalize_path(model_path))
        
        self.model = torch.jit.load(str(model_path), map_location="cpu")
        self.model.eval()
        self.artifacts = context.artifacts


class DetectorWrapper(mlflow.pyfunc.PythonModel):
    """MLflow wrapper for Detector system."""
    
    def __init__(self):
        super().__init__()

    def __getstate__(self):
        """Handle serialization - exclude model objects."""
        state = self.__dict__.copy()
        # Remove any model references that can't be serialized
        #state.pop('model', None)
        #state.pop('artifacts', None)
        return state

    def __setstate__(self, state):
        """Handle deserialization."""
        self.__dict__.update(state)
        # Model will be loaded in load_context

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the TorchScript classification model from the artifacts."""
        localizer_ckpt = Path(context.artifacts["localizer_ckpt"])
        classifier_ckpt = Path(context.artifacts["classifier_ckpt"])
        config = Path(context.artifacts["config"])

        for a in [localizer_ckpt,classifier_ckpt,config]:
            a = normalize_path(a)
        
        config = ConfigLoader.load_detector_registration_config(config)

        assert (config.localizer.yolo is None) + (config.localizer.mmdet is None) == 1, f"Exactly one should be given"
        localizer_cfg = config.localizer.yolo or config.localizer.mmdet
        localizer_cfg.weights = str(localizer_ckpt)

        self.model = Detector.from_config(localizer_config=localizer_cfg,
                                        classifier_ckpt=classifier_ckpt)
        self.artifacts = context.artifacts
    
    #def predict(self,context: mlflow.pyfunc.PythonModelContext,model_input:List[List[float]]):
    #    torch_input = torch.tensor(model_input)
    #    return self.model.predict(torch_input,return_as_dict=True)


def get_experiment_id(name: str) -> str:
    """Get or create an MLflow experiment ID.
    
    Args:
        name: MLflow experiment name
        
    Returns:
        Experiment ID
    """
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        logger.info(f"Created new experiment: {name} (ID: {exp_id})")
        return exp_id
    return exp.experiment_id


def get_conda_env() -> Dict[str, Any]:
    """Get the conda environment configuration for MLflow models."""
    return {
        "channels": ["defaults"],
        "dependencies": [
            "python>=3.11",
            "pip",
            {
                "pip": [
                    "pillow",
                    "mlflow",
                    "ultralytics",
                    "torch",
                    "pydantic"
                ],
            },
        ],
        "name": "wildtrain_env",
    }


class ModelRegistrar:
    """Handles model registration to MLflow Model Registry."""
    
    def __init__(self, mlflow_tracking_uri: str = "http://localhost:5000"):
        """Initialize the ModelRegistrar.
        
        Args:
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri
    
    def register_detector(
        self,config_path:Path
    ) -> None:
        """Register a YOLO detection model to MLflow Model Registry.
        """

        config = ConfigLoader.load_detector_registration_config(config_path)

        assert (config.localizer.yolo is None) + (config.localizer.mmdet is None) == 1, f"Exactly one should be given"
        localizer_cfg = config.localizer.yolo or config.localizer.mmdet
        localizer_processing = config.localizer.processing

        self.tracking_uri = config.processing.mlflow_tracking_uri

        localizer_ckpt = self._export_localizer(
            model_path=Path(localizer_cfg.weights),
            export_format=localizer_processing.export_format,
            image_size=localizer_cfg.imgsz,
            batch_size=localizer_processing.batch_size,
            device=localizer_cfg.device,
            dynamic=localizer_processing.dynamic,
            task=localizer_cfg.task,
        )
        
        self.model = Detector.from_config(localizer_config=localizer_cfg,
                                        classifier_ckpt=str(config.classifier.weights_path))
          
        artifacts = {"localizer_ckpt": str(localizer_ckpt),
                    "classifier_ckpt":str(config.classifier.weights_path),
                    "config":str(config_path)
        }

        if getattr(localizer_cfg,"config",None) is not None:
            artifacts["mmdet_config"] = getattr(localizer_cfg,"config",None)
        
        # Create metadata using the new ModelMetadata class
        task = getattr(localizer_cfg,"task","detect")
        metadata = ModelMetadata(
            batch=localizer_processing.batch_size,
            imgsz=localizer_cfg.imgsz,
            task=ModelTask(task),
            model_type=ModelType.DETECTOR,
        )
        
        self._register_model(
            artifacts=artifacts,
            metadata=metadata.to_dict(),
            name=config.processing.name,
            python_model=DetectorWrapper(),
        )
        
        logger.info(f"Successfully registered detector model: {config.processing.name}")
    
    
    def register_localizer(
        self,
        weights_path: Union[str, Path],
        name: str = "detector",
        export_format: str = "torchscript",
        image_size: int = 800,
        batch_size: int = 8,
        device: str = "cpu",
        dynamic: bool = False,
        task: str = "detect",
        model_type: str = "yolo"
    ) -> None:
        """Register a YOLO detection model to MLflow Model Registry.
        
        Args:
            weights_path: Path to the model weights
            name: Model name for registration
            export_format: Export format (torchscript, openvino, etc.)
            image_size: Input image size
            batch_size: Batch size for inference
            device: Device to use for export
            dynamic: Whether to use dynamic batching
            task: YOLO task type
            model_type: yolo or MMDet
        """

        if model_type != "yolo":
            raise NotImplementedError(f"Supports only 'yolo' not '{model_type}'")
        model_path = Path(weights_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        
        export_path = self._export_localizer(
            model_path, export_format, image_size, batch_size, device, dynamic, task
        )
        
        artifacts = {"localizer_ckpt": str(export_path)}
        
        # Create metadata using the new ModelMetadata class
        metadata = ModelMetadata(
            batch=batch_size,
            imgsz=image_size,
            task=ModelTask(task) if isinstance(task, str) else task,
            model_type=ModelType.LOCALIZER,
        )
        
        self._register_model(
            artifacts=artifacts,
            metadata=metadata.to_dict(),
            name=name,
            python_model=UltralyticsWrapper(),
        )
        
        logger.info(f"Successfully registered detector model: {name}")
    
    def register_classifier(
        self,
        weights_path: Union[str, Path],
        name: str = "classifier",
        batch_size: int = 8,
    ) -> None:
        """Register a classification model to MLflow Model Registry.
        
        Args:
            weights_path: Path to the model checkpoint
            name: Model name for registration
            batch_size: Batch size for inference
        """
        model_path = Path(weights_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        
        # Load the model
        model = GenericClassifier.load_from_checkpoint(str(model_path))
        
        classifier_ckpt = self._export_classifier(
            model_path=model_path,
            batch_size=batch_size,
        )
        
        artifacts = {"classifier_ckpt": str(classifier_ckpt)}
        
        metadata = ModelMetadata(
            batch=batch_size,
            imgsz=model.input_size.item(),
            task=ModelTask.CLASSIFY,
            num_classes=model.num_classes.item(),
            model_type=ModelType.CLASSIFIER
        )
        
        self._register_model(
            artifacts=artifacts,
            metadata=metadata.to_dict(),
            name=name,
            python_model=ClassifierWrapper(),
        )
        
        logger.info(f"Successfully registered classifier model: {name}")
    
    def _export_localizer(
        self,
        model_path: Path,
        export_format: str,
        image_size: int,
        batch_size: int,
        device: str,
        dynamic: bool,
        task: str,
    ) -> Path:
        """Export the detector model in the specified format."""
        if export_format == "pt":
            return normalize_path(model_path)
        
        try:
            model = YOLO(model_path, task=task)
            model.export(
                format=export_format,
                imgsz=image_size,
                optimize=device == "cpu",
                nms=True,
                dynamic=dynamic,
                batch=batch_size,
                device=device,
            )
            
            if export_format == "openvino":
                export_path = model_path.with_name(f"{model_path.stem}_openvino_model")
            else:
                export_path = model_path.with_suffix(f".{export_format}")
            
            return normalize_path(export_path)
            
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            raise
    
    def _export_classifier(
        self,
        model_path: Path,
        batch_size: int,
    ) -> Path:
        """Export the classifier model in the specified format."""
        
        try:
            model = GenericClassifier.load_from_checkpoint(str(model_path))
            model.eval()
            dummy_input = torch.randn(batch_size, 3, model.input_size.item(), model.input_size.item())
            try:
                scripted_model = torch.jit.script(model)
            except Exception as e:
                logger.info(f"Scripting failed. Falling back to tracing.")
                scripted_model = torch.jit.trace(model, dummy_input)
            scripted_path = model_path.with_suffix(".torchscript")
            scripted_model.save(str(scripted_path))
            return normalize_path(scripted_path)
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            raise
    
    def _register_model(
        self,
        artifacts: Dict[str, str],
        metadata: Dict[str, Any],
        name: str,
        python_model: mlflow.pyfunc.PythonModel,
    ) -> None:
        """Register a model to MLflow Model Registry."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        exp_id = get_experiment_id(name)
        
        with mlflow.start_run(experiment_id=exp_id):
            mlflow.pyfunc.log_model(
                "model",
                python_model=python_model,
                conda_env=get_conda_env(),
                artifacts=artifacts,
                registered_model_name=name,
                metadata=metadata,
            )

