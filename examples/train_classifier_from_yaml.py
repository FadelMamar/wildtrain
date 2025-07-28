#!/usr/bin/env python3
"""
Example script showing how to use ClassificationDataModule.from_yaml() 
for clean instantiation from YAML configuration files.
"""

import sys
from pathlib import Path
from tqdm import tqdm


from wildtrain.data import ClassificationDataModule
from wildtrain.trainers.classification_trainer import ClassifierTrainer
from omegaconf import OmegaConf


def main():
    """Example of using ClassificationDataModule.from_yaml()."""
    
    # Method 1: Load directly from YAML file
    config_path = "configs/classification/unified_classification_config.yaml"
    
    print("=== Loading ClassificationDataModule from YAML file ===")
    datamodule = ClassificationDataModule.from_yaml(config_path)
    
    # Setup the datamodule
    datamodule.setup(stage="fit")
    
    print(f"Dataset type: {datamodule.dataset_type}")
    print(f"Class mapping: {datamodule.class_mapping}")
    print(f"Train dataset size: {len(datamodule.train_dataset)}")
    print(f"Val dataset size: {len(datamodule.val_dataset)}")
    
    # Get a sample batch
    train_loader = datamodule.train_dataloader()
    sample_batch = next(iter(train_loader))
    print(f"Sample batch shape: {sample_batch[0].shape}")
    print(f"Sample labels shape: {sample_batch[1].shape}")

    for _ in tqdm(train_loader):
        pass
    
    print("\n" + "="*50 + "\n")
    
        
    print("=== Using with ClassificationTrainer ===")
    config = OmegaConf.load(config_path)
    
    # Create trainer (this would normally be done with proper config)
    trainer = ClassifierTrainer(config)
    trainer.run(debug=True)

    
    
    print("\n" + "="*50 + "\n")
    
    


if __name__ == "__main__":
    main() 