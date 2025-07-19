#!/usr/bin/env python3
"""
Example script for training a classifier using WildTrain framework.

This script demonstrates how to:
1. Set up the training configuration
2. Run classification training
3. Use different backbones and configurations

Usage:
    python examples/train_classifier.py
"""

import os
import sys
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

# Add the src directory to the path so we can import wildtrain modules

from wildtrain.trainers.classification_trainer import run_classification


def create_basic_config():
    """
    Create a basic configuration for classification training.
    """
    config = {
        "task": "classification",
        "mode": "train",
        "dataset": {
            "root_data_directory": r"D:\workspace\data\demo-dataset",            
            "transforms": {
                "train": [
                    {"name": "RandomResizedCrop", "params": {"size": 128, "scale": (0.8, 1.0)}},
                    {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
                    {"name": "ColorJitter", "params": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1}},
                    {"name": "RandomRotation", "params": {"degrees": 10}},
                    {"name": "Normalize", "params": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
                ],
                "val": [
                    {"name": "Resize", "params": {"size": 128}},
                    {"name": "CenterCrop", "params": {"size": 128}},
                    {"name": "Normalize", "params": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
                ]
            }
        },
        "model": {
            "backbone": "resnet18",
            "pretrained": True,
            "dropout": 0.2,
            "no_grad_backbone": False
        },
        "train": {
            "batch_size": 16,
            "epochs": 30,
            "lr": 0.001,
            "threshold": 0.5,
            "label_smoothing": 0.0,
            "weight_decay": 1e-4,
            "lrf": 0.01
        },
        "checkpoint": {
            "monitor": "val_acc",
            "save_top_k": 1,
            "mode": "max",
            "dirpath": "./checkpoints/classification",
            "patience": 10
        },
        "mlflow": {
            "experiment_name": "wildtrain_classification",
            "run_name": "resnet18_basic",
            "tracking_uri": "file:./mlruns",
            "log_model": True
        }
    }
    return OmegaConf.create(config)


def create_advanced_config():
    """
    Create an advanced configuration with EfficientNet and more sophisticated transforms.
    """
    config = {
        "task": "classification",
        "mode": "train",
        "dataset": {
            "root_data_directory": r"D:\workspace\data\demo-dataset",
            "transforms": {
                "train": [
                    {"name": "RandomResizedCrop", "params": {"size": 224, "scale": (0.7, 1.0)}},
                    {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
                    {"name": "RandomVerticalFlip", "params": {"p": 0.3}},
                    {"name": "ColorJitter", "params": {"brightness": 0.3, "contrast": 0.3, "saturation": 0.3, "hue": 0.1}},
                    {"name": "RandomRotation", "params": {"degrees": 15}},
                    {"name": "RandomAffine", "params": {"degrees": 0, "translate": (0.1, 0.1), "scale": (0.9, 1.1)}},
                    {"name": "GaussianBlur", "params": {"kernel_size": 3, "sigma": (0.1, 2.0)}},
                    {"name": "Normalize", "params": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
                ],
                "val": [
                    {"name": "Resize", "params": {"size": 256}},
                    {"name": "CenterCrop", "params": {"size": 224}},
                    {"name": "Normalize", "params": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
                ]
            }
        },
        "model": {
            "backbone": "efficientnet_b0",
            "pretrained": True,
            "dropout": 0.3,
            "no_grad_backbone": False
        },
        "train": {
            "batch_size": 16,  # Smaller batch size for EfficientNet
            "epochs": 50,
            "lr": 0.0005,
            "threshold": 0.5,
            "label_smoothing": 0.1,
            "weight_decay": 1e-4,
            "lrf": 0.01
        },
        "checkpoint": {
            "monitor": "val_acc",
            "save_top_k": 3,
            "mode": "max",
            "dirpath": "./checkpoints/classification",
            "patience": 15
        },
        "mlflow": {
            "experiment_name": "wildtrain_classification",
            "run_name": "efficientnet_advanced",
            "tracking_uri": "file:./mlruns",
            "log_model": True
        }
    }
    return OmegaConf.create(config)


def main():
    """
    Main function to run classification training with different configurations.
    """
    print("🚀 Starting WildTrain Classification Training Example")
    print("=" * 60)
    
    # Create output directories
    os.makedirs("./checkpoints/classification", exist_ok=True)
    os.makedirs("./data/classification", exist_ok=True)
    
    # Example 1: Basic ResNet18 training
    print("\n📋 Example 1: Basic ResNet18 Training")
    print("-" * 40)
    
    basic_config = create_basic_config()
    print("Configuration:")
    print(OmegaConf.to_yaml(basic_config))
    
    try:
        run_classification(basic_config)
        print("✅ Basic training completed successfully!")
    except Exception as e:
        print(f"❌ Basic training failed: {e}")
    
    # Example 2: Advanced EfficientNet training
    print("\n📋 Example 2: Advanced EfficientNet Training")
    print("-" * 40)
    
    advanced_config = create_advanced_config()
    print("Configuration:")
    print(OmegaConf.to_yaml(advanced_config))
    
    try:
        run_classification(advanced_config)
        print("✅ Advanced training completed successfully!")
    except Exception as e:
        print(f"❌ Advanced training failed: {e}")
    
    print("\n🎉 Training examples completed!")
    print("\n📁 Check the following directories for outputs:")
    print("   - ./checkpoints/classification/ - Model checkpoints")
    print("   - ./mlruns/ - MLflow experiment tracking")
    print("   - ./logs/ - Training logs")


if __name__ == "__main__":
    main() 