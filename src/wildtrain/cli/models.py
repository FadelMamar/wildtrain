"""Pydantic models for CLI configuration validation."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator
from pydantic.types import DirectoryPath, FilePath
import yaml


class BaseConfig(BaseModel):
    """Base configuration class with common fields."""
    
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "BaseConfig":
        """Create config from YAML file."""
        if not yaml_path.exists():
            raise ValueError(f"YAML file does not exist: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            data = {}
        
        return cls(**data)
    
    def to_yaml(self, yaml_path: Path) -> None:
        """Save config to YAML file."""
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, indent=2)


class LoggingConfig(BaseConfig):
    """Logging configuration."""
    mlflow: bool = True
    log_dir: Path = Field(default=Path("./logs"), description="Logging directory")
    experiment_name: str = Field(default="wildtrain", description="MLflow experiment name")
    run_name: str = Field(default="default", description="MLflow run name")


class DatasetStatsConfig(BaseConfig):
    """Dataset statistics configuration."""
    mean: List[float] = Field(description="Mean values for normalization")
    std: List[float] = Field(description="Standard deviation values for normalization")
    
    @field_validator ('mean', 'std')
    @classmethod
    def validate_stats_length(cls, v):
        if len(v) != 3:
            raise ValueError("Mean and std must have exactly 3 values (RGB)")
        return v


class TransformConfig(BaseConfig):
    """Individual transform configuration."""
    name: str = Field(description="Transform name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Transform parameters")


class TransformsConfig(BaseConfig):
    """Transforms configuration for train and validation."""
    train: List[TransformConfig] = Field(default_factory=list, description="Training transforms")
    val: List[TransformConfig] = Field(default_factory=list, description="Validation transforms")


class CurriculumConfig(BaseConfig):
    """Curriculum learning configuration."""
    enabled: bool = Field(default=False, description="Enable curriculum learning")
    type: Literal["difficulty"] = Field(default="difficulty", description="Curriculum type")
    difficulty_strategy: Literal["linear"] = Field(default="linear", description="Difficulty strategy")
    start_difficulty: float = Field(default=0.0, ge=0.0, le=1.0, description="Starting difficulty")
    end_difficulty: float = Field(default=1.0, ge=0.0, le=1.0, description="Ending difficulty")
    warmup_epochs: int = Field(default=0, ge=0, description="Warmup epochs")
    log_frequency: int = Field(default=10, ge=1, description="Log frequency")
    
    @field_validator('end_difficulty')
    @classmethod
    def validate_end_difficulty(cls, v, values):
        if 'start_difficulty' in values and v <= values['start_difficulty']:
            raise ValueError("end_difficulty must be greater than start_difficulty")
        return v


class SingleClassConfig(BaseConfig):
    """Single class configuration for ROI datasets."""
    enable: bool = Field(default=False, description="Enable single class mode")
    background_class_name: str = Field(default="background", description="Background class name")
    single_class_name: str = Field(default="wildlife", description="Single class name")
    keep_classes: Optional[List[str]] = Field(default=None, description="Classes to keep")
    discard_classes: Optional[List[str]] = Field(default=None, description="Classes to discard")


class DatasetConfig(BaseConfig):
    """Dataset configuration."""
    root_data_directory: Path = Field(description="Root data directory path")
    dataset_type: Literal["roi", "crop"] = Field(description="Dataset type")
    input_size: int = Field(gt=0, description="Input image size")
    batch_size: int = Field(gt=0, description="Batch size")
    num_workers: int = Field(default=0, ge=0, description="Number of workers")
    stats: DatasetStatsConfig = Field(description="Dataset statistics")
    transforms: TransformsConfig = Field(description="Data transforms")
    curriculum_config: Optional[CurriculumConfig] = Field(default=None, description="Curriculum configuration")
    single_class: Optional[SingleClassConfig] = Field(default=None, description="Single class configuration")
    rebalance: bool = Field(default=False, description="Enable dataset rebalancing")
    
    # Crop dataset specific fields
    crop_size: Optional[int] = Field(default=None, gt=0, description="Crop size for crop datasets")
    max_tn_crops: Optional[int] = Field(default=None, gt=0, description="Maximum true negative crops")
    p_draw_annotations: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Probability of drawing annotations")
    compute_difficulties: Optional[bool] = Field(default=None, description="Compute crop difficulties")
    preserve_aspect_ratio: Optional[bool] = Field(default=None, description="Preserve aspect ratio")
    
    @field_validator('root_data_directory')
    @classmethod
    def validate_data_directory(cls, v):
        if not v.exists():
            raise ValueError(f"Data directory does not exist: {v}")
        return v
    
    @field_validator('crop_size', 'max_tn_crops', 'p_draw_annotations', 'compute_difficulties', 'preserve_aspect_ratio')
    @classmethod
    def validate_crop_fields(cls, v, values):
        if 'dataset_type' in values and values['dataset_type'] == 'crop':
            if v is None:
                raise ValueError(f"Field is required for crop dataset type")
        return v


class ModelConfig(BaseConfig):
    """Model configuration."""
    backbone: str = Field(description="Model backbone")
    pretrained: bool = Field(default=True, description="Use pretrained weights")
    backbone_source: str = Field(default="timm", description="Backbone source")
    dropout: float = Field(default=0.2, ge=0.0, le=1.0, description="Dropout rate")
    freeze_backbone: bool = Field(default=False, description="Freeze backbone")
    input_size: int = Field(gt=0, description="Model input size")
    mean: List[float] = Field(description="Normalization mean values")
    std: List[float] = Field(description="Normalization std values")
    
    @field_validator('mean', 'std')
    @classmethod
    def validate_normalization(cls, v):
        if len(v) != 3:
            raise ValueError("Normalization values must have exactly 3 values (RGB)")
        return v


class TrainingConfig(BaseConfig):
    """Training configuration."""
    batch_size: int = Field(gt=0, description="Training batch size")
    epochs: int = Field(gt=0, description="Number of training epochs")
    lr: float = Field(gt=0.0, description="Learning rate")
    label_smoothing: float = Field(default=0.0, ge=0.0, le=1.0, description="Label smoothing")
    weight_decay: float = Field(default=0.0001, ge=0.0, description="Weight decay")
    lrf: float = Field(default=0.1, gt=0.0, description="Learning rate factor")
    precision: str = Field(default="32", description="Training precision")
    accelerator: str = Field(default="auto", description="Training accelerator")


class CheckpointConfig(BaseConfig):
    """Checkpoint configuration."""
    monitor: str = Field(description="Metric to monitor")
    save_top_k: int = Field(default=1, ge=0, description="Number of best models to save")
    mode: Literal["min", "max"] = Field(description="Monitor mode")
    save_last: bool = Field(default=True, description="Save last checkpoint")
    dirpath: Path = Field(description="Checkpoint directory")
    patience: int = Field(default=10, gt=0, description="Early stopping patience")
    save_weights_only: bool = Field(default=True, description="Save only weights")
    filename: str = Field(description="Checkpoint filename pattern")
    min_delta: float = Field(default=0.001, ge=0.0, description="Minimum improvement delta")


class MLflowConfig(BaseConfig):
    """MLflow configuration."""
    experiment_name: str = Field(description="MLflow experiment name")


class ClassificationConfig(BaseConfig):
    """Complete classification configuration."""
    dataset: DatasetConfig = Field(description="Dataset configuration")
    model: ModelConfig = Field(description="Model configuration")
    train: TrainingConfig = Field(description="Training configuration")
    checkpoint: CheckpointConfig = Field(description="Checkpoint configuration")
    mlflow: MLflowConfig = Field(description="MLflow configuration")
    
    @field_validator('model')
    @classmethod
    def validate_model_dataset_compatibility(cls, v, values):
        if 'dataset' in values:
            dataset = values['dataset']
            if v.input_size != dataset.input_size:
                raise ValueError(f"Model input_size ({v.input_size}) must match dataset input_size ({dataset.input_size})")
        return v


class YoloConfig(BaseConfig):
    """YOLO model configuration."""
    weights: Optional[Path] = Field(default=None, description="Model weights path")
    imgsz: int = Field(gt=0, description="Input image size")
    device: str = Field(default="cpu", description="Device to use")
    conf_thres: float = Field(default=0.2, ge=0.0, le=1.0, description="Confidence threshold")
    iou_thres: float = Field(default=0.3, ge=0.0, le=1.0, description="IoU threshold")
    max_det: int = Field(default=300, gt=0, description="Maximum detections")
    overlap_metric: str = Field(default="IOU", description="Overlap metric")
    
    @field_validator('device')
    @classmethod
    def validate_device(cls, v):
        valid_devices = ['cpu', 'cuda', 'auto']
        if v not in valid_devices:
            raise ValueError(f"Device must be one of {valid_devices}")
        return v


class MMDetConfig(BaseConfig):
    """MMDetection model configuration."""
    config_file: Path = Field(description="MMDetection config file path")
    checkpoint: Optional[Path] = Field(default=None, description="Model checkpoint path")
    device: str = Field(default="cpu", description="Device to use")
    
    @field_validator('config_file')
    @classmethod
    def validate_config_file(cls, v):
        if not v.exists():
            raise ValueError(f"Config file does not exist: {v}")
        return v


class DetectionConfig(BaseConfig):
    """Complete detection configuration."""
    model_type: Literal["yolo", "mmdet"] = Field(description="Detection model type")
    detection_model_config: Union[YoloConfig, MMDetConfig] = Field(description="Model configuration")
    dataset: DatasetConfig = Field(description="Dataset configuration")
    train: TrainingConfig = Field(description="Training configuration")
    checkpoint: CheckpointConfig = Field(description="Checkpoint configuration")
    mlflow: MLflowConfig = Field(description="MLflow configuration")


class FiftyOneConfig(BaseConfig):
    """FiftyOne configuration."""
    dataset_name: str = Field(description="FiftyOne dataset name")
    prediction_field: str = Field(description="Prediction field name")


class ClassifierConfig(BaseConfig):
    """Classifier configuration for visualization."""
    checkpoint: Optional[Path] = Field(default=None, description="Classifier checkpoint path")


class VisualizationModelConfig(BaseConfig):
    """Model configuration for visualization."""
    localizer: YoloConfig = Field(description="Localizer configuration")
    classifier: Optional[ClassifierConfig] = Field(default=None, description="Classifier configuration")


class ProcessingConfig(BaseConfig):
    """Processing configuration."""
    batch_size: int = Field(gt=0, description="Processing batch size")
    debug: bool = Field(default=False, description="Debug mode")
    num_workers: int = Field(default=0, ge=0, description="Number of workers")


class VisualizationConfig(BaseConfig):
    """Visualization configuration."""
    fiftyone: FiftyOneConfig = Field(description="FiftyOne configuration")
    model: VisualizationModelConfig = Field(description="Model configuration")
    processing: ProcessingConfig = Field(description="Processing configuration")


class TrainPipelineConfig(BaseConfig):
    """Training pipeline configuration."""
    config: Path = Field(description="Training config file path")
    debug: bool = Field(default=False, description="Debug mode")
    
    @field_validator('config')
    @classmethod
    def validate_config_file(cls, v):
        if not v.exists():
            raise ValueError(f"Training config file does not exist: {v}")
        return v


class EvalPipelineConfig(BaseConfig):
    """Evaluation pipeline configuration."""
    config: Path = Field(description="Evaluation config file path")
    debug: bool = Field(default=False, description="Debug mode")
    
    @field_validator('config')
    @classmethod
    def validate_config_file(cls, v):
        if not v.exists():
            raise ValueError(f"Evaluation config file does not exist: {v}")
        return v


class PipelineConfig(BaseConfig):
    """Base pipeline configuration."""
    disable_train: bool = Field(default=False, description="Disable training pipeline")
    train: TrainPipelineConfig = Field(description="Training configuration")
    disable_eval: bool = Field(default=False, description="Disable evaluation pipeline")
    eval: EvalPipelineConfig = Field(description="Evaluation configuration")
    results_dir: Path = Field(description="Results directory for pipeline outputs")
    
    @field_validator('results_dir')
    @classmethod
    def validate_results_dir(cls, v):
        # Ensure results directory exists or can be created
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator('train', 'eval')
    @classmethod
    def validate_pipeline_configs(cls, v, values):
        # Validate that at least one pipeline is enabled
        if 'disable_train' in values and 'disable_eval' in values:
            if values['disable_train'] and values['disable_eval']:
                raise ValueError("At least one pipeline (train or eval) must be enabled")
        return v
    
    def is_train_enabled(self) -> bool:
        """Check if training pipeline is enabled."""
        return not self.disable_train
    
    def is_eval_enabled(self) -> bool:
        """Check if evaluation pipeline is enabled."""
        return not self.disable_eval
    
    def get_enabled_pipelines(self) -> List[str]:
        """Get list of enabled pipeline names."""
        pipelines = []
        if self.is_train_enabled():
            pipelines.append("train")
        if self.is_eval_enabled():
            pipelines.append("eval")
        return pipelines
    
    def validate_pipeline_files_exist(self) -> None:
        """Validate that all referenced config files exist."""
        if self.is_train_enabled() and not self.train.config.exists():
            raise ValueError(f"Training config file does not exist: {self.train.config}")
        if self.is_eval_enabled() and not self.eval.config.exists():
            raise ValueError(f"Evaluation config file does not exist: {self.eval.config}")


class ClassificationPipelineConfig(PipelineConfig):
    """Classification pipeline configuration.
    
    This configuration manages the classification training and evaluation pipeline.
    It supports both training and evaluation phases with separate configurations.
    
    Example:
        config = ClassificationPipelineConfig(
            disable_train=False,
            train=TrainPipelineConfig(
                config=Path("configs/classification/classification_train.yaml"),
                debug=False
            ),
            disable_eval=True,
            eval=EvalPipelineConfig(
                config=Path("configs/classification/classification_eval.yaml"),
                debug=False
            ),
            results_dir=Path("results/classification")
        )
    """
    
    @field_validator('results_dir')
    @classmethod
    def validate_classification_results_dir(cls, v):
        # Ensure classification-specific results directory
        v = v / "classification" if v.name != "classification" else v
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    def get_classification_results_path(self) -> Path:
        """Get the classification-specific results path."""
        return self.results_dir / "classification" if self.results_dir.name != "classification" else self.results_dir
    
    @classmethod
    def create_default(cls, results_dir: Path = Path("results/classification")) -> "ClassificationPipelineConfig":
        """Create a default classification pipeline configuration."""
        return cls(
            disable_train=False,
            train=TrainPipelineConfig(
                config=Path("configs/classification/classification_train.yaml"),
                debug=False
            ),
            disable_eval=True,
            eval=EvalPipelineConfig(
                config=Path("configs/classification/classification_eval.yaml"),
                debug=False
            ),
            results_dir=results_dir
        )


class DetectionPipelineConfig(PipelineConfig):
    """Detection pipeline configuration.
    
    This configuration manages the detection training and evaluation pipeline.
    It supports both training and evaluation phases with separate configurations.
    
    Example:
        config = DetectionPipelineConfig(
            disable_train=False,
            train=TrainPipelineConfig(
                config=Path("configs/detection/yolo_configs/yolo.yaml"),
                debug=False
            ),
            disable_eval=True,
            eval=EvalPipelineConfig(
                config=Path("configs/detection/yolo_configs/yolo_eval.yaml"),
                debug=False
            ),
            results_dir=Path("results/yolo")
        )
    """
    
    @field_validator('results_dir')
    @classmethod
    def validate_detection_results_dir(cls, v):
        # Ensure detection-specific results directory
        v = v / "detection" if v.name != "detection" else v
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    def get_detection_results_path(self) -> Path:
        """Get the detection-specific results path."""
        return self.results_dir / "detection" if self.results_dir.name != "detection" else self.results_dir
    
    @classmethod
    def create_default(cls, results_dir: Path = Path("results/yolo")) -> "DetectionPipelineConfig":
        """Create a default detection pipeline configuration."""
        return cls(
            disable_train=False,
            train=TrainPipelineConfig(
                config=Path("configs/detection/yolo_configs/yolo.yaml"),
                debug=False
            ),
            disable_eval=True,
            eval=EvalPipelineConfig(
                config=Path("configs/detection/yolo_configs/yolo_eval.yaml"),
                debug=False
            ),
            results_dir=results_dir
        )
