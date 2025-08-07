"""Evaluation service for integrating CLI functionality with the API."""

import tempfile
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from omegaconf import OmegaConf
from wildtrain.evaluators.classification import ClassificationEvaluator
from wildtrain.evaluators.ultralytics import UltralyticsEvaluator
from wildtrain.cli.config_loader import ConfigLoader
from wildtrain.cli.models import ClassificationEvalConfig, DetectionEvalConfig

logger = logging.getLogger(__name__)


class EvaluationService:
    """Service for handling evaluation operations."""

    @staticmethod
    def evaluate_classifier(config: ClassificationEvalConfig, debug: bool = False) -> Dict[str, Any]:
        """Evaluate a classification model using the CLI evaluator."""
        try:
            # Create a temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                config_dict = config.model_dump()
                yaml_content = OmegaConf.to_yaml(OmegaConf.create(config_dict))
                f.write(yaml_content)
                temp_config_path = f.name

            logger.info(f"Created temporary config file: {temp_config_path}")

            # Use the CLI evaluator
            evaluator = ClassificationEvaluator(temp_config_path)
            results = evaluator.evaluate(debug=debug)

            # Clean up temporary file
            Path(temp_config_path).unlink(missing_ok=True)

            logger.info("Classification evaluation completed successfully")
            return results

        except Exception as e:
            logger.error(f"Classification evaluation failed: {e}")
            raise

    @staticmethod
    def evaluate_detector(config: DetectionEvalConfig, debug: bool = False) -> Dict[str, Any]:
        """Evaluate a detection model using the CLI evaluator."""
        try:
            # Create a temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                config_dict = config.model_dump()
                yaml_content = OmegaConf.to_yaml(OmegaConf.create(config_dict))
                f.write(yaml_content)
                temp_config_path = f.name

            logger.info(f"Created temporary config file: {temp_config_path}")

            # Use the CLI evaluator
            evaluator = UltralyticsEvaluator(temp_config_path)
            results = evaluator.evaluate(debug=debug)

            # Clean up temporary file
            Path(temp_config_path).unlink(missing_ok=True)

            logger.info("Detection evaluation completed successfully")
            return results

        except Exception as e:
            logger.error(f"Detection evaluation failed: {e}")
            raise

    @staticmethod
    def generate_classification_eval_template() -> str:
        """Generate a classification evaluation template."""
        try:
            template = ConfigLoader.generate_default_config("classification_eval")
            return template
        except Exception as e:
            logger.error(f"Failed to generate classification evaluation template: {e}")
            raise

    @staticmethod
    def generate_detection_eval_template() -> str:
        """Generate a detection evaluation template."""
        try:
            template = ConfigLoader.generate_default_config("detection_eval")
            return template
        except Exception as e:
            logger.error(f"Failed to generate detection evaluation template: {e}")
            raise
