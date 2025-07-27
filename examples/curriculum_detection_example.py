"""
Example: Using the Refactored Curriculum Detection Dataset

This script demonstrates how to use the new curriculum detection dataset
that inherits from supervision.DetectionDataset with convenience loading methods.
"""


from torchvision import transforms
from tqdm import tqdm
import traceback
# Import the new curriculum learning components
from wildtrain.data.curriculum import (
    CurriculumConfig, 
    CurriculumDetectionDataset,
    CurriculumCallback,
    CropDataset
)


def example_coco_loading():
    """Example: Loading dataset from COCO format using convenience method."""
    print("=== COCO Loading Example ===")
    
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
    
    # 3. Load dataset using convenience method
    try:
        dataset = CurriculumDetectionDataset.from_coco_with_curriculum(
            images_directory_path=r"D:\workspace\data\demo-dataset\datasets",  # Your images directory
            annotations_path=r"D:\workspace\data\demo-dataset\datasets\savmap\annotations\train.json",  # Your COCO annotations
            curriculum_config=curriculum_config,
            transform=transform
        )

        for _ in tqdm(dataset,desc="Loading CurriculumDetectionDataset"):
            continue
        
        print(f"✅ Dataset loaded successfully!")
        print(f"✅ Number of samples: {len(dataset)}")
        print(f"✅ Classes: {dataset.classes}")
        print(f"✅ Available samples: {len(dataset.available_indices)}")
        
        # Test curriculum progression
        print("\nTesting curriculum progression:")
        for epoch in [0, 10, 20, 30, 40]:
            dataset.update_curriculum_state(
                difficulty=min(1.0, epoch / 40.0)
            )
            print(f"Epoch {epoch:2d}: difficulty={dataset.current_difficulty:.3f}, "
                  f"available_samples={len(dataset.available_indices)}")
        
    except FileNotFoundError:
        print("⚠️  COCO files not found. This is expected if you don't have the data files.")
        print("   Replace the paths with your actual COCO dataset paths.")


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
        dataset = CurriculumDetectionDataset.from_coco_with_curriculum(
            images_directory_path=r"D:\workspace\data\savmap_dataset_v2\annotated_py_paul\coco-format\images",  # Your images directory
            annotations_path=r"D:\workspace\data\savmap_dataset_v2\annotated_py_paul\coco-format\annotations.json",  # Your COCO annotations
            curriculum_config=curriculum_config,
            transform=transform
        )

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

    except FileNotFoundError:
        print("⚠️  Dataset files not found. This is expected if you don't have the data files.")
        print("   Replace the paths with your actual dataset paths.")
    except Exception as e:
        print(f"❌ Error: {e}")
        
        print(traceback.format_exc())


if __name__ == "__main__":
    print("Curriculum Detection Dataset Examples")
    print("=" * 60)

    Crop_loading()
    
    # Run examples
    #try:
        #example_coco_loading()
    #except Exception as e:
        #print(f"COCO loading example failed: {traceback.format_exc()}")
    
    

    
