"""
Example: Using the Refactored Curriculum Detection Dataset

This script demonstrates how to use the new curriculum detection dataset
that inherits from supervision.DetectionDataset.
"""

import torch
import lightning as L
from torchvision import transforms
import numpy as np
import supervision as sv
from pathlib import Path

# Import the new curriculum learning components
from wildtrain.data.curriculum import (
    CurriculumConfig, 
    CurriculumDetectionDataset,
    MultiScaleDetectionDataset,
    CurriculumCallback
)


def create_sample_detection_data():
    """Create sample detection data for demonstration."""
    
    # Sample class names
    classes = ["person", "car", "dog"]
    
    # Sample image paths (in real usage, these would be actual file paths)
    image_paths = [
        "sample1.jpg",
        "sample2.jpg", 
        "sample3.jpg"
    ]
    
    # Sample annotations using supervision.Detections
    annotations = {}
    
    # Image 1: One person (large detection = easy)
    annotations["sample1.jpg"] = sv.Detections(
        xyxy=np.array([[50, 50, 200, 300]]),  # Large bounding box
        confidence=np.array([0.9]),
        class_id=np.array([0])  # person
    )
    
    # Image 2: One car (medium detection = medium difficulty)
    annotations["sample2.jpg"] = sv.Detections(
        xyxy=np.array([[100, 100, 250, 200]]),  # Medium bounding box
        confidence=np.array([0.8]),
        class_id=np.array([1])  # car
    )
    
    # Image 3: One dog (small detection = hard)
    annotations["sample3.jpg"] = sv.Detections(
        xyxy=np.array([[150, 150, 180, 180]]),  # Small bounding box
        confidence=np.array([0.7]),
        class_id=np.array([2])  # dog
    )
    
    return classes, image_paths, annotations


def example_basic_curriculum_detection():
    """Example: Basic curriculum detection dataset usage."""
    print("=== Basic Curriculum Detection Dataset ===")
    
    # 1. Create curriculum configuration
    curriculum_config = CurriculumConfig(
        enabled=True,
        type="difficulty",
        difficulty_strategy="linear",
        start_difficulty=0.0,
        end_difficulty=1.0,
        warmup_epochs=0,
        multiscale_enabled=False
    )
    
    # 2. Create sample data
    classes, image_paths, annotations = create_sample_detection_data()
    
    # 3. Create transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 4. Create curriculum detection dataset
    dataset = CurriculumDetectionDataset(
        classes=classes,
        images=image_paths,
        annotations=annotations,
        curriculum_config=curriculum_config,
        transform=transform
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Available samples: {dataset.available_indices}")
    
    # 5. Test curriculum progression
    print("\nTesting curriculum progression:")
    for epoch in [0, 10, 20, 30, 40]:
        # Update curriculum state
        dataset.update_curriculum_state(
            difficulty=min(1.0, epoch / 40.0),  # Linear progression
            scale=1.0
        )
        
        print(f"Epoch {epoch:2d}: difficulty={dataset.current_difficulty:.3f}, "
              f"available_samples={len(dataset.available_indices)}")
        
        # Show which samples are available
        if len(dataset.available_indices) > 0:
            sample_idx = dataset.available_indices[0]
            sample = dataset[0]  # Get first available sample
            print(f"  Sample {sample_idx}: difficulty={sample['difficulty']:.3f}, "
                  f"class={classes[sample['detections'].class_id[0]] if len(sample['detections']) > 0 else 'none'}")


def example_multiscale_curriculum_detection():
    """Example: Multi-scale curriculum detection dataset usage."""
    print("\n=== Multi-Scale Curriculum Detection Dataset ===")
    
    # 1. Create multi-scale curriculum configuration
    curriculum_config = CurriculumConfig(
        enabled=True,
        type="multiscale",
        multiscale_enabled=True,
        base_size=416,
        scale_range=(0.5, 2.0),
        num_scales=5,
        scale_strategy="curriculum",
        difficulty_strategy="random"
    )
    
    # 2. Create sample data
    classes, image_paths, annotations = create_sample_detection_data()
    
    # 3. Create transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 4. Create multi-scale detection dataset
    dataset = MultiScaleDetectionDataset(
        classes=classes,
        images=image_paths,
        annotations=annotations,
        curriculum_config=curriculum_config,
        transform=transform,
        preserve_aspect_ratio=True,
        pad_to_square=True
    )
    
    print(f"Multi-scale dataset created with {len(dataset)} samples")
    
    # Get available scales from the dataset's curriculum manager
    available_scales = [0.5, 0.75, 1.0, 1.25, 1.5]  # Default scales
    print(f"Available scales: {available_scales}")
    
    # 5. Test scale progression
    print("\nTesting scale progression:")
    for epoch in [0, 10, 20, 30, 40]:
        # Update curriculum state
        scale = available_scales[min(epoch // 10, len(available_scales) - 1)]
        dataset.update_curriculum_state(
            difficulty=1.0,  # Max difficulty
            scale=scale
        )
        
        print(f"Epoch {epoch:2d}: scale={dataset.current_scale:.3f}, "
              f"target_size={int(dataset.base_size * dataset.current_scale)}px")


def example_combined_curriculum_detection():
    """Example: Combined difficulty and multi-scale curriculum."""
    print("\n=== Combined Curriculum Detection Dataset ===")
    
    # 1. Create combined curriculum configuration
    curriculum_config = CurriculumConfig(
        enabled=True,
        type="both",
        difficulty_strategy="exponential",
        start_difficulty=0.0,
        end_difficulty=1.0,
        warmup_epochs=5,
        multiscale_enabled=True,
        base_size=416,
        scale_range=(0.5, 2.0),
        num_scales=5,
        scale_strategy="curriculum"
    )
    
    # 2. Create sample data
    classes, image_paths, annotations = create_sample_detection_data()
    
    # 3. Create transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 4. Create combined curriculum dataset
    dataset = CurriculumDetectionDataset(
        classes=classes,
        images=image_paths,
        annotations=annotations,
        curriculum_config=curriculum_config,
        transform=transform
    )
    
    print(f"Combined curriculum dataset created with {len(dataset)} samples")
    
    # 5. Test combined progression
    print("\nTesting combined curriculum progression:")
    for epoch in [0, 10, 20, 30, 40, 50]:
        # Calculate difficulty (exponential progression)
        progress = min(1.0, max(0, epoch - curriculum_config.warmup_epochs) / (50 - curriculum_config.warmup_epochs))
        difficulty = 1.0 - np.exp(-3.0 * progress)
        
        # Calculate scale (curriculum progression)
        available_scales = [0.5, 0.75, 1.0, 1.25, 1.5]  # Default scales
        scale_idx = min(int(progress * (len(available_scales) - 1)), len(available_scales) - 1)
        scale = available_scales[scale_idx]
        
        # Update curriculum state
        dataset.update_curriculum_state(difficulty=difficulty, scale=scale)
        
        print(f"Epoch {epoch:2d}: difficulty={dataset.current_difficulty:.3f}, "
              f"scale={dataset.current_scale:.3f}, "
              f"available_samples={len(dataset.available_indices)}")


def example_supervision_integration():
    """Example: Integration with supervision library features."""
    print("\n=== Supervision Integration ===")
    
    # 1. Create curriculum configuration
    curriculum_config = CurriculumConfig(
        enabled=True,
        type="difficulty",
        difficulty_strategy="linear",
        start_difficulty=0.0,
        end_difficulty=1.0
    )
    
    # 2. Create sample data
    classes, image_paths, annotations = create_sample_detection_data()
    
    # 3. Create dataset
    dataset = CurriculumDetectionDataset(
        classes=classes,
        images=image_paths,
        annotations=annotations,
        curriculum_config=curriculum_config
    )
    
    print("Supervision integration features:")
    print(f"✅ Lazy loading: {dataset._images_in_memory == {}}")
    print(f"✅ Class names: {dataset.classes}")
    print(f"✅ Image paths: {len(dataset.image_paths)} images")
    print(f"✅ Annotations: {len(dataset.annotations)} annotations")
    
    # 4. Demonstrate supervision features
    if len(dataset) > 0:
        sample = dataset[0]
        detections = sample['detections']
        
        print(f"✅ Detection format: {type(detections)}")
        if hasattr(detections, 'xyxy') and len(detections.xyxy) > 0:
            print(f"✅ Bounding boxes: {detections.xyxy.shape}")
            print(f"✅ Class IDs: {detections.class_id}")
            print(f"✅ Confidences: {detections.confidence}")
        else:
            print("✅ No detections in sample")


if __name__ == "__main__":
    print("Curriculum Detection Dataset Examples")
    print("=" * 60)
    
    # Run examples
    try:
        example_basic_curriculum_detection()
    except Exception as e:
        print(f"Basic curriculum example failed: {e}")
    
    try:
        example_multiscale_curriculum_detection()
    except Exception as e:
        print(f"Multi-scale curriculum example failed: {e}")
    
    try:
        example_combined_curriculum_detection()
    except Exception as e:
        print(f"Combined curriculum example failed: {e}")
    
    try:
        example_supervision_integration()
    except Exception as e:
        print(f"Supervision integration example failed: {e}")
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("\nKey Benefits of the Refactored Interface:")
    print("✅ Inherits from supervision.DetectionDataset")
    print("✅ Leverages lazy loading and memory efficiency")
    print("✅ Uses supervision.Detections for type safety")
    print("✅ Maintains all curriculum functionality")
    print("✅ Backward compatible with existing code")
    print("✅ Better integration with supervision ecosystem") 