"""
Example demonstrating CropDataset with ClusteringFilter using adapter pattern.

This example shows how to:
1. Create a CropDataset from detection annotations
2. Use CropClusteringAdapter to apply ClusteringFilter to crop annotations
3. Use the clustered dataset with PyTorch DataLoader
4. Analyze clustering results and crop diversity
"""

import torch
from torchvision import transforms
from tqdm import tqdm
import traceback
from pathlib import Path
import numpy as np

# Import our modules
from wildtrain.data.curriculum.dataset import CurriculumDetectionDataset, CropDataset
from wildtrain.data.curriculum.manager import CurriculumConfig
from wildtrain.data.filters import ClusteringFilter, CropClusteringAdapter


def main():
    """Main example function."""
    
    print("CropDataset with ClusteringFilter (Adapter Pattern) Example")
    print("=" * 70)
    
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
        print("📁 Loading base dataset...")
        dataset = CurriculumDetectionDataset.from_data_directory(
            root_data_directory=r"D:\workspace\data\demo-dataset",  # Your images directory
            split="train",
            curriculum_config=curriculum_config,
            transform=transform
        )

        print(f"✅ Base dataset loaded successfully!")
        print(f"✅ Number of samples: {len(dataset)}")
        print(f"✅ Classes: {dataset.classes}")

        # 4. Create CropDataset
        print("\n🌾 Creating CropDataset...")
        crop_dataset = CropDataset(
            dataset=dataset,
            crop_size=224,
            max_tn_crops=1,
            p_draw_annotations=0.0  # No annotations for cleaner crops
        )

        print(f"✅ CropDataset created successfully!")
        print(f"✅ Number of crops: {len(crop_dataset)}")

        # 5. Analyze class distribution before clustering
        print("\n📊 Class distribution before clustering:")
        annotations_before = crop_dataset.get_annotations_for_filter()
        
        # Count crops per class
        class_counts_before = {}
        for ann in annotations_before:
            class_name = ann['class_name']
            class_counts_before[class_name] = class_counts_before.get(class_name, 0) + 1
        
        for class_name, count in class_counts_before.items():
            print(f"   {class_name}: {count} crops")

        # 6. Create ClusteringFilter and adapter
        print("\n🔄 Creating ClusteringFilter with adapter...")
        
        # Create the base clustering filter
        clustering_filter = ClusteringFilter(
            batch_size=32,
            reduction_factor=0.5  # Keep 50% of crops
        )
        
        # Create the adapter to make it work with crop annotations
        clustering_adapter = CropClusteringAdapter(clustering_filter)

        # Apply filter to create clustered dataset
        clustered_crop_dataset = crop_dataset.apply_clustering_filter(clustering_adapter)

        print(f"✅ Clustered CropDataset created successfully!")
        print(f"✅ Number of crops after clustering: {len(clustered_crop_dataset)}")

        # 7. Analyze class distribution after clustering
        print("\n📊 Class distribution after clustering:")
        annotations_after = clustered_crop_dataset.get_annotations_for_filter()
        
        class_counts_after = {}
        for ann in annotations_after:
            class_name = ann['class_name']
            class_counts_after[class_name] = class_counts_after.get(class_name, 0) + 1
        
        for class_name, count in class_counts_after.items():
            print(f"   {class_name}: {count} crops")

        # 8. Test DataLoader compatibility
        print("\n🧪 Testing DataLoader compatibility...")
        
        from torch.utils.data import DataLoader
        
        # Create DataLoader with clustered dataset
        dataloader = DataLoader(
            clustered_crop_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0  # Set to 0 for this example
        )

        # Test a few batches
        print("Testing batch loading:")
        for batch_idx, (crops, labels) in enumerate(dataloader):
            print(f"   Batch {batch_idx + 1}: crops shape={crops.shape}, labels={labels.tolist()}")
            if batch_idx >= 2:  # Only test first 3 batches
                break

        # 9. Demonstrate utility methods
        print("\n🔧 Testing utility methods:")
        
        # Get crops by class (assuming class_id 24 is wildlife)
        if len(dataset.classes) > 24:
            wildlife_crops = clustered_crop_dataset.get_crops_by_class(24)
            print(f"   Wildlife crops: {len(wildlife_crops)} indices")
        
        # Get crops by type
        detection_crops = clustered_crop_dataset.get_crops_by_type('detection')
        random_crops = clustered_crop_dataset.get_crops_by_type('random')
        print(f"   Detection crops: {len(detection_crops)} indices")
        print(f"   Random crops: {len(random_crops)} indices")
        
        # Get detailed info for first crop
        if len(clustered_crop_dataset) > 0:
            crop_info = clustered_crop_dataset.get_crop_info(0)
            print(f"   First crop info keys: {list(crop_info.keys())}")

        # 10. Demonstrate different clustering configurations
        print("\n🔄 Testing different clustering configurations:")
        
        # More aggressive clustering (keep only 30%)
        aggressive_filter = ClusteringFilter(
            batch_size=32,
            reduction_factor=0.3
        )
        
        aggressive_adapter = CropClusteringAdapter(aggressive_filter)
        aggressive_clustered = crop_dataset.apply_clustering_filter(aggressive_adapter)
        print(f"   Aggressive clustering: {len(aggressive_clustered)} crops")
        
        # Less aggressive clustering (keep 70%)
        conservative_filter = ClusteringFilter(
            batch_size=32,
            reduction_factor=0.7
        )
        
        conservative_adapter = CropClusteringAdapter(conservative_filter)
        conservative_clustered = crop_dataset.apply_clustering_filter(conservative_adapter)
        print(f"   Conservative clustering: {len(conservative_clustered)} crops")

        # 11. Show adapter information
        print("\n📝 Adapter information:")
        adapter_info = clustering_adapter.get_filter_info()
        for key, value in adapter_info.items():
            print(f"   {key}: {value}")

        # 12. Show annotation format
        print("\n📝 Example annotation format:")
        if len(annotations_after) > 0:
            example_ann = annotations_after[0]
            print("   Example annotation:")
            for key, value in example_ann.items():
                print(f"     {key}: {value}")

        print("\n✅ All tests completed successfully!")

    except FileNotFoundError:
        print("⚠️  Dataset files not found. This is expected if you don't have the data files.")
        print("   Replace the paths with your actual dataset paths.")
        print("\n   Expected paths:")
        print("   - Images: D:\\workspace\\data\\demo-dataset\\datasets")
        print("   - Annotations: D:\\workspace\\data\\demo-dataset\\datasets\\savmap\\annotations\\train.json")
    except Exception as e:
        print(f"❌ Error: {e}")
        print(traceback.format_exc())


def demonstrate_usage():
    """Demonstrate typical usage patterns."""
    
    print("\n" + "="*70)
    print("USAGE PATTERNS")
    print("="*70)
    
    print("""
1. Basic Usage with Adapter:
   clustering_filter = ClusteringFilter(reduction_factor=0.5)
   clustering_adapter = CropClusteringAdapter(clustering_filter)
   clustered_dataset = crop_dataset.apply_clustering_filter(clustering_adapter)

2. With DataLoader:
   dataloader = DataLoader(clustered_dataset, batch_size=32, shuffle=True)

3. Different Clustering Configurations:
   - reduction_factor=0.3: Keep 30% of crops (aggressive)
   - reduction_factor=0.5: Keep 50% of crops (balanced)
   - reduction_factor=0.7: Keep 70% of crops (conservative)

4. Utility Methods:
   - get_crops_by_class(class_id): Filter by class
   - get_crops_by_type('detection'): Filter by crop type
   - get_crop_info(idx): Get detailed crop metadata

5. Adapter Information:
   - get_filter_info(): Get clustering configuration and results
   - last_silhouette_scores: Access clustering quality metrics
   - last_samples_per_cluster: Access cluster allocation results
    """)


def demonstrate_adapter_pattern():
    """Demonstrate the adapter pattern benefits."""
    
    print("\n" + "="*70)
    print("ADAPTER PATTERN BENEFITS")
    print("="*70)
    
    print("""
1. Interface Compatibility:
   - ClusteringFilter expects: List[Dict] with 'id' and 'file_name'
   - CropDataset provides: List[Dict] with 'roi_id' and crop-specific fields
   - Adapter converts between these formats seamlessly

2. Reuse Without Modification:
   - ClusteringFilter remains unchanged
   - No code duplication
   - Single responsibility principle maintained

3. Flexibility:
   - Can use any ClusteringFilter configuration
   - Easy to extend for other filter types
   - Clean separation of concerns

4. Maintainability:
   - Changes to ClusteringFilter automatically benefit crop filtering
   - Adapter logic is isolated and testable
   - Clear interface boundaries
    """)


if __name__ == "__main__":
    main()
    demonstrate_usage()
    demonstrate_adapter_pattern() 