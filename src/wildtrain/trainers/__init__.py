"""
Trainers module for WildTrain.

This module provides trainer classes for different types of computer vision tasks.
"""

from .base import ModelTrainer
from .classification_trainer import ClassifierTrainer, run_classification
from .detection_trainer import DetectorTrainer, run_detection

__all__ = [
    "ModelTrainer",
    "ClassifierTrainer", 
    "DetectorTrainer",
    "run_classification",
    "run_detection",
]
