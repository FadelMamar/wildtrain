#!/usr/bin/env python3
"""
Example script demonstrating curriculum learning integration with classification.

This script shows how to:
1. Configure curriculum learning for classification
2. Use the integrated curriculum callback
3. Monitor curriculum progress through MLflow logging
"""

import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wildtrain.trainers.classification_trainer import ClassifierTrainer
from wildtrain.data import ClassificationDataModule
from wildtrain.data.curriculum import CurriculumConfig, CurriculumCallback


def create_curriculum_config() -> DictConfig:
    """Create a curriculum configuration for testing."""
    config_dict = {
        "dataset": {
            "root_data_directory": "D:/workspace/data/demo-dataset",
            "batch_size": 8,
            "dataset_type": "crop",  # Use crop dataset for curriculum
            "crop_size": 224,
            "max_tn_crops": 1,
            "p_draw_annotations": 0.0,
            "compute_difficulties": True,
            "preserve_aspect_ratio": True,
            
            # Curriculum configuration
            "curriculum_config": {
                "enabled": True,
                "type": "difficulty",
                "difficulty_strategy": "linear",
                "start_difficulty": 0.0,
                "end_difficulty": 1.0,
                "warmup_epochs": 2,
                "log_frequency": 1
            }
        },
        "model": {
            "backbone": "resnet18",
            "backbone_source": "timm",
            "pretrained": True,
            "dropout": 0.2,
            "freeze_backbone": True,
            "input_size": 224,
            "mean": [0.554, 0.469, 0.348],
            "std": [0.203, 0.173, 0.144]
        },
        "train": {
            "epochs": 10,
            "lr": 1e-3,
            "lrf": 1e-2,
            "weight_decay": 5e-3,
            "label_smoothing": 0.0,
            "accelerator": "auto",
            "precision": "16-mixed"
        },
        "checkpoint": {
            "monitor": "val_loss",
            "save_top_k": 1,
            "save_last": True,
            "mode": "min",
            "dirpath": "checkpoints",
            "filename": "best_classifier",
            "save_weights_only": False,
            "patience": 10,
            "min_delta": 0.001
        },
        "mlflow": {
            "experiment_name": "curriculum_example",
            "run_name": "curriculum_classification_demo",
            "log_model": False
        },
        "logging": {
            "mlflow": True,
            "log_dir": "./logs"
        }
    }
    
    return OmegaConf.create(config_dict)


def demonstrate_curriculum_integration():
    """Demonstrate the curriculum learning integration."""
    print("=" * 80)
    print("Curriculum Learning Integration Demo")
    print("=" * 80)
    
    # Create configuration
    config = create_curriculum_config()
    print("✓ Configuration created with curriculum settings")
    
    # Create trainer
    trainer = ClassifierTrainer(config)
    print("✓ ClassifierTrainer created")
    
    # Test datamodule creation
    datamodule = ClassificationDataModule.from_dict_config(config)
    print("✓ ClassificationDataModule created")
    
    # Check curriculum status
    if datamodule.is_curriculum_enabled():
        print("✓ Curriculum learning is enabled")
        print(f"  - Type: {datamodule.curriculum_config.type}")
        print(f"  - Strategy: {datamodule.curriculum_config.difficulty_strategy}")
        print(f"  - Start difficulty: {datamodule.curriculum_config.start_difficulty}")
        print(f"  - End difficulty: {datamodule.curriculum_config.end_difficulty}")
        print(f"  - Warmup epochs: {datamodule.curriculum_config.warmup_epochs}")
    else:
        print("✗ Curriculum learning is disabled")
    
    # Test curriculum callback creation
    if datamodule.is_curriculum_enabled():
        callback = CurriculumCallback(datamodule)
        print("✓ CurriculumCallback created successfully")
        
        # Test curriculum state management
        datamodule.setup_curriculum(10)  # 10 epochs
        initial_state = datamodule.get_curriculum_state()
        print(f"✓ Initial curriculum state: {initial_state}")
        
        # Simulate epoch updates
        for epoch in range(5):
            state = datamodule.update_curriculum_epoch(epoch)
            print(f"  Epoch {epoch}: difficulty = {state.get('difficulty', 0.0):.3f}")
    
    print("\n" + "=" * 80)
    print("Curriculum integration demo completed successfully!")
    print("=" * 80)


def run_training_example():
    """Run a short training example with curriculum learning."""
    print("\n" + "=" * 80)
    print("Running Training Example with Curriculum Learning")
    print("=" * 80)
    
    try:
        # Create configuration for short training
        config = create_curriculum_config()
        config.train.epochs = 3  # Short training for demo
        
        # Create trainer
        trainer = ClassifierTrainer(config)
        
        # Run training with curriculum
        print("Starting training with curriculum learning...")
        trainer.run(debug=True)  # Use debug mode for short training
        
        print("✓ Training completed successfully!")
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run the curriculum learning example."""
    print("Curriculum Learning Classification Example")
    print("This example demonstrates how to integrate curriculum learning")
    print("with the classification trainer in WildTrain.\n")
    
    # Demonstrate the integration
    demonstrate_curriculum_integration()
    
    # Ask user if they want to run training
    print("\nWould you like to run a short training example? (y/n): ", end="")
    try:
        response = input().lower().strip()
        if response in ['y', 'yes']:
            run_training_example()
        else:
            print("Skipping training example.")
    except KeyboardInterrupt:
        print("\nSkipping training example.")


if __name__ == "__main__":
    main()