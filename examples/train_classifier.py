#!/usr/bin/env python3
"""
Example script for training a classifier using WildTrain framework.

This script demonstrates how to:
1. Load training configurations from YAML files
2. Run classification training with different configurations
3. Use different backbones and configurations

Usage:
    python examples/train_classifier.py
"""

import os
import sys
from pathlib import Path
from omegaconf import OmegaConf
from wildtrain.trainers.classification_trainer import run_classification
import traceback

def load_config(config_path: str):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        OmegaConf configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    print(f"✅ Loaded configuration from: {config_path}")
    return config


def main():
    """
    Main function to run classification training with different configurations.
    """
    print("🚀 Starting WildTrain Classification Training Example")
    print("=" * 60)
    
    # Create output directories
    print("-" * 40)
    
    basic_config = load_config(r"D:\workspace\repos\wildtrain\configs\classification\example_config.yaml")
    print("Configuration:")
    print(OmegaConf.to_yaml(basic_config))
    
    try:
        run_classification(basic_config)
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {traceback.format_exc()}")
        

if __name__ == "__main__":
    main() 