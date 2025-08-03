"""
Example: Using the Refactored Curriculum Detection Dataset

This script demonstrates how to use the new curriculum detection dataset
that inherits from supervision.DetectionDataset with convenience loading methods.
"""


from torchvision import transforms
from tqdm import tqdm
import traceback
from wildtrain.data.curriculum import (
    CurriculumConfig, 
    CurriculumDetectionDataset,
    CropDataset
)
from wildtrain.trainers.classification_trainer import ClassifierTrainer
from wildtrain.data import ClassificationDataModule
from wildtrain.data.curriculum import CurriculumConfig, CurriculumCallback
from omegaconf import DictConfig, OmegaConf


def Crop_loading():
    """Example demonstrating CropDataset with ClassificationRebalanceFilter."""
    
    print("\n=== CropDataset with Rebalancing Example ===")
    
    # 1. Create curriculum configuration
    curriculum_config = CurriculumConfig(
        enabled=True,
        type="difficulty",
        difficulty_strategy="linear",
        start_difficulty=0.0,
        end_difficulty=1.0,
        warmup_epochs=0,
    )

    # 2. Create transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    try:
        # 3. Load base dataset
        dataset = CurriculumDetectionDataset.from_data_directory(
            root_data_directory=r"D:\workspace\data\demo-dataset",
            split="train",
            curriculum_config=curriculum_config,
            transform=transform
        )
        
        classes = set()
        for detection in dataset.annotations.values():
            classes = classes.union(set(detection.class_id))
        
        for class_id in classes:
            class_name = dataset.classes[class_id]
            print(f"   {class_name}: {class_id}")


        #print(f"✅ Classes: {classes}");exit(1)


        print(f"✅ Base dataset loaded successfully!")
        print(f"✅ Number of samples: {len(dataset)}")
        print(f"✅ Classes: {dataset.classes}")

        # 4. Create CropDataset
        crop_dataset = CropDataset(
            dataset=dataset,
            crop_size=224,
            max_tn_crops=1,
            p_draw_annotations=0.0  # No annotations for cleaner crops
        )
        
        classes = dict()
        for i in range(len(crop_dataset)):
            crop_info = crop_dataset.get_crop_info(i)
            class_id = crop_info['label']
            if class_id != -1:
                classes[class_id] = classes.get(class_id,0) + 1
            else:
                classes[-1] = classes.get(-1,0) + 1
        
        for class_name, count in classes.items():
            print(f"   {class_name}: {count} crops")
        
        print(f"✅ CropDataset created successfully!")
        print(f"✅ Number of crops: {len(crop_dataset)}")

        annotations_before = crop_dataset.get_annotations_for_filter()
        
        # Count crops per class
        class_counts_before = {}
        for ann in annotations_before:
            class_name = ann['class_name']
            class_counts_before[class_name] = class_counts_before.get(class_name, 0) + 1

        # 5. Analyze class distribution before rebalancing
        print("\n📊 Class distribution")
        for class_name, count in class_counts_before.items():
            print(f"   {class_name}: {count} crops")
        
        for _ in tqdm(crop_dataset,desc="Iterating CropDataset",unit="samples"):
            continue

    except FileNotFoundError:
        print("⚠️  Dataset files not found. This is expected if you don't have the data files.")
        print("   Replace the paths with your actual dataset paths.")
    except Exception as e:
        print(f"❌ Error: {e}")
        
        print(traceback.format_exc())


def run_training_example():
    """Run a short training example with curriculum learning."""
    print("\n" + "=" * 80)
    print("Running Training Example with Curriculum Learning")
    print("=" * 80)
    
    try:
        # Create configuration for short training
        config = config = OmegaConf.load(r"D:\workspace\repos\wildtrain\configs\classification\classification_train.yaml")
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

if __name__ == "__main__":
    print("Curriculum Detection Dataset Examples")
    print("=" * 60)

    Crop_loading()
    
    
    

    
