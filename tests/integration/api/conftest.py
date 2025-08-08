"""
Shared fixtures and utilities for API integration tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from wildtrain.api.main import fastapi_app as app
from wildtrain.cli.config_loader import ConfigLoader, ROOT


def safe_load_config(config_path: Path, loader_method, fallback_config: Dict = None):
    """Safely load config with fallback to template or empty config."""
    try:
        validated_config = loader_method(config_path)
        # Convert WindowsPath objects to strings for JSON serialization
        config_dict = validated_config.model_dump()
        
        # Recursively convert any WindowsPath objects to strings
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif hasattr(obj, '__class__') and 'WindowsPath' in str(type(obj)):
                return str(obj)
            else:
                return obj
        
        config_dict = convert_paths(config_dict)
        return config_dict
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        return fallback_config or {}


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


@pytest.fixture
def classification_config():
    """Load real classification config for testing."""
    config_path = ROOT / "configs" / "classification" / "classification_train.yaml"
    return safe_load_config(
        config_path,
        ConfigLoader.load_classification_config,
        
    )


@pytest.fixture
def detection_config():
    """Load real detection config for testing."""
    config_path = ROOT / "configs" / "detection" / "yolo_configs" / "yolo.yaml"
    return safe_load_config(
        config_path,
        ConfigLoader.load_detection_config,
        
    )


@pytest.fixture
def classification_eval_config():
    """Load real classification evaluation config for testing."""
    config_path = ROOT / "configs" / "classification" / "classification_eval.yaml"
    return safe_load_config(
        config_path,
        ConfigLoader.load_classification_eval_config,
        
    )


@pytest.fixture
def detection_eval_config():
    """Load real detection evaluation config for testing."""
    config_path = ROOT / "configs" / "detection" / "yolo_configs" / "yolo_eval.yaml"
    return safe_load_config(
        config_path,
        ConfigLoader.load_detection_eval_config,
        
    )


@pytest.fixture
def classification_visualization_config():
    """Load real classification visualization config for testing."""
    config_path = ROOT / "configs" / "classification" / "classification_visualization.yaml"
    return safe_load_config(
        config_path,
        ConfigLoader.load_classification_visualization_config,
        
    )


@pytest.fixture
def detection_visualization_config():
    """Load real detection visualization config for testing."""
    config_path = ROOT / "configs" / "detection" / "visualization.yaml"
    return safe_load_config(
        config_path,
        ConfigLoader.load_detection_visualization_config,
        
    )


@pytest.fixture
def classification_pipeline_config():
    """Load real classification pipeline config for testing."""
    config_path = ROOT / "configs" / "classification" / "classification_pipeline_config.yaml"
    return safe_load_config(
        config_path,
        lambda p: ConfigLoader.load_pipeline_config(p, "classification"),
        
    )


@pytest.fixture
def detection_pipeline_config():
    """Load real detection pipeline config for testing."""
    config_path = ROOT / "configs" / "detection" / "yolo_configs" / "yolo_pipeline_config.yaml"
    return safe_load_config(
        config_path,
        lambda p: ConfigLoader.load_pipeline_config(p, "detection"),
        
    )
