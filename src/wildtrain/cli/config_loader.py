"""Configuration loader for CLI with Pydantic validation."""

import json
from pathlib import Path
from typing import Dict, Any, Type, TypeVar, Union
from omegaconf import OmegaConf
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .models import (
    ClassificationConfig,
    DetectionConfig,
    VisualizationConfig,
    ClassificationPipelineConfig,
    DetectionPipelineConfig,
)

console = Console()

T = TypeVar('T', bound=BaseModel)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigFileNotFoundError(Exception):
    """Raised when configuration file is not found."""
    pass


class ConfigParseError(Exception):
    """Raised when configuration file cannot be parsed."""
    pass


class ConfigLoader:
    """Configuration loader with Pydantic validation."""
    
    @staticmethod
    def load_and_validate(
        config_path: Path,
        config_class: Type[T],
        config_name: str = "configuration"
    ) -> T:
        """
        Load YAML configuration file and validate against Pydantic model.
        
        Args:
            config_path: Path to the configuration file
            config_class: Pydantic model class to validate against
            config_name: Name of the configuration for error messages
            
        Returns:
            Validated configuration object
            
        Raises:
            ConfigFileNotFoundError: If config file doesn't exist
            ConfigParseError: If config file cannot be parsed
            ConfigValidationError: If config fails validation
        """
        # Check if file exists
        if not config_path.exists():
            raise ConfigFileNotFoundError(f"{config_name} file not found: {config_path}")
        
        try:
            # Load YAML using OmegaConf
            cfg = OmegaConf.load(config_path)
            
            # Convert to dict
            config_dict = OmegaConf.to_container(cfg, resolve=True)
            
            # Validate against Pydantic model
            validated_config = config_class(**config_dict)
            
            console.print(f"[bold green]✓[/bold green] {config_name} loaded and validated successfully")
            
            return validated_config
            
        except Exception as e:
            if isinstance(e, (ConfigFileNotFoundError, ConfigParseError, ConfigValidationError)):
                raise
            
            if isinstance(e, ValidationError):
                # Format Pydantic validation errors
                error_messages = []
                for error in e.errors():
                    field_path = " -> ".join(str(loc) for loc in error["loc"])
                    error_messages.append(f"  • {field_path}: {error['msg']}")  # type: ignore
                
                error_text = f"Configuration validation failed:\n" + "\n".join(error_messages)
                console.print(Panel(error_text, title="❌ Validation Error", border_style="red"))
                raise ConfigValidationError(f"{config_name} validation failed: {str(e)}")
            
            raise ConfigParseError(f"Failed to parse {config_name} file: {str(e)}")
    
    @staticmethod
    def load_classification_config(config_path: Path) -> ClassificationConfig:
        """Load and validate classification configuration."""
        return ConfigLoader.load_and_validate(
            config_path, 
            ClassificationConfig, 
            "Classification configuration"
        )
    
    @staticmethod
    def load_detection_config(config_path: Path) -> DetectionConfig:
        """Load and validate detection configuration."""
        return ConfigLoader.load_and_validate(
            config_path, 
            DetectionConfig, 
            "Detection configuration"
        )
    
    @staticmethod
    def load_visualization_config(config_path: Path) -> VisualizationConfig:
        """Load and validate visualization configuration."""
        return ConfigLoader.load_and_validate(
            config_path, 
            VisualizationConfig, 
            "Visualization configuration"
        )
    
    @staticmethod
    def load_pipeline_config(
        config_path: Path, 
        pipeline_type: str = "classification"
    ) -> Union[ClassificationPipelineConfig, DetectionPipelineConfig]:
        """Load and validate pipeline configuration."""
        if pipeline_type == "classification":
            return ConfigLoader.load_and_validate(
                config_path, 
                ClassificationPipelineConfig, 
                "Classification pipeline configuration"
            )
        elif pipeline_type == "detection":
            return ConfigLoader.load_and_validate(
                config_path, 
                DetectionPipelineConfig, 
                "Detection pipeline configuration"
            )
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
    
    @staticmethod
    def validate_config_file(config_path: Path, config_type: str) -> bool:
        """
        Validate a configuration file without loading it into memory.
        
        Args:
            config_path: Path to the configuration file
            config_type: Type of configuration (classification, detection, visualization, pipeline)
            
        Returns:
            True if valid, raises exception if invalid
        """
        config_map = {
            "classification": ClassificationConfig,
            "detection": DetectionConfig,
            "visualization": VisualizationConfig,
            "pipeline": ClassificationPipelineConfig,  # Default to classification pipeline
        }
        
        if config_type not in config_map:
            raise ValueError(f"Unknown config type: {config_type}. Valid types: {list(config_map.keys())}")
        
        config_class = config_map[config_type]
        
        try:
            # Use type: ignore to bypass the type checker for this dynamic call
            ConfigLoader.load_and_validate(config_path, config_class, f"{config_type.title()} configuration")  # type: ignore
            return True
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Configuration validation failed: {str(e)}")
            return False
    
    @staticmethod
    def generate_schema(config_class: Type[T]) -> Dict[str, Any]:
        """Generate JSON schema for a configuration class."""
        return config_class.model_json_schema()
    
    @staticmethod
    def save_schema(config_class: Type[T], output_path: Path) -> None:
        """Save JSON schema to file."""
        schema = ConfigLoader.generate_schema(config_class)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=2)
        console.print(f"[bold green]✓[/bold green] Schema saved to: {output_path}")
