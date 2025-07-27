"""
Simplified Curriculum-Aware Dataset.

This module provides a clean dataset implementation that supports
both difficulty-based and multi-scale curriculum learning.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
from PIL import Image
import supervision as sv
import albumentations as A
import cv2
from .manager import CurriculumConfig
from tqdm import tqdm
from pathlib import Path
import json
import os

def group_coco_annotations_by_image_id(
        coco_annotations: list[dict],
    ) -> dict[int, list[dict]]:
        annotations = {}
        for annotation in coco_annotations:
            image_id = annotation["image_id"]
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        return annotations

class CurriculumDetectionDataset(sv.DetectionDataset):
    """
    Curriculum-aware dataset that inherits from supervision.DetectionDataset.
    
    This dataset can:
    1. Filter samples based on difficulty (for difficulty-based curriculum)
    2. Resize images based on current scale (for multi-scale curriculum)
    3. Work seamlessly with the CurriculumManager
    4. Leverage supervision's lazy loading and memory efficiency
    5. Use supervision's convenience loading methods (from_coco, from_yolo, etc.)
    """
    
    def __init__(self, 
                 classes: List[str],
                 images: Union[List[str], Dict[str, np.ndarray]],
                 annotations: Dict[str, sv.Detections],
                 curriculum_config: Optional[CurriculumConfig] = None,
                 transform=None,
                 compute_difficulties: bool = True,
                 preserve_aspect_ratio: bool = True,
                 pad_to_square: bool = True):
        """
        Initialize curriculum detection dataset.
        
        Args:
            classes: List of class names
            images: List of image paths or dict of loaded images
            annotations: Dictionary mapping image paths to Detections
            curriculum_config: Curriculum configuration
            transform: Image transformations
            compute_difficulties: Whether to compute sample difficulties
            preserve_aspect_ratio: Whether to preserve aspect ratio when resizing
            pad_to_square: Whether to pad images to square
        """
        super().__init__(classes, images, annotations)
        self.curriculum_config = curriculum_config
        self.transform = transform
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.pad_to_square = pad_to_square
        
        # Compute sample difficulties if needed
        if compute_difficulties:
            self.sample_difficulties = self._compute_sample_difficulties()
        else:
            self.sample_difficulties = [0.5] * len(self.image_paths)  # Default medium difficulty
        
        # Initialize available indices (all samples initially)
        self.available_indices = list(range(len(self.image_paths)))
        
        # Current curriculum state
        self.current_difficulty = 1.0

        #print("Number of classes:",len(self.classes))
        #print("Number of images:",len(self.image_paths))
        #print("Number of annotations:",len(self.annotations))
    
    @classmethod
    def from_coco_with_curriculum(cls,
                                 images_directory_path: str,
                                 annotations_path: str,
                                 curriculum_config: Optional[CurriculumConfig] = None,
                                 transform=None,
                                 compute_difficulties: bool = True,
                                 preserve_aspect_ratio: bool = True,
                                 pad_to_square: bool = True):
        """
        Create curriculum detection dataset from COCO format.
        
        Args:
            images_directory_path: Path to directory containing images
            annotations_path: Path to COCO annotations JSON file
            curriculum_config: Curriculum configuration
            transform: Image transformations
            compute_difficulties: Whether to compute sample difficulties
            preserve_aspect_ratio: Whether to preserve aspect ratio when resizing
            pad_to_square: Whether to pad images to square
            
        Returns:
            CurriculumDetectionDataset instance
        """
        # Use supervision's from_coco method
        detection_dataset = sv.DetectionDataset.from_coco(
            images_directory_path=images_directory_path,
            annotations_path=annotations_path
        )

        with open(annotations_path, "r",encoding="utf-8") as f:
            coco_annotations = json.load(f)

        grouped_annotations = group_coco_annotations_by_image_id(coco_annotations["annotations"])

        #print("Number of images:",len(coco_annotations["images"]))
        #print("Number of annotations:",len(coco_annotations["annotations"]))
        #print("Number of grouped annotations:",len(grouped_annotations))
        
        background_images = []
        empty_annotations = {}
        for coco_image in coco_annotations["images"]:
            if len(grouped_annotations.get(coco_image["id"], [])) < 1:
                image_path = os.path.join(images_directory_path, coco_image["file_name"])
                background_images.append(image_path)
                empty_annotations[image_path] = sv.Detections.empty()

        # Add empty annotations to the dataset
        detection_dataset.annotations.update(empty_annotations)

        #print(empty_annotations)
        #print("background_images:",len(background_images))

        # Create curriculum dataset using the loaded data
        return cls(
            classes=detection_dataset.classes,
            images=detection_dataset.image_paths, # + background_images,
            annotations=detection_dataset.annotations,
            curriculum_config=curriculum_config,
            transform=transform,
            compute_difficulties=compute_difficulties,
            preserve_aspect_ratio=preserve_aspect_ratio,
            pad_to_square=pad_to_square
        )
    
       
    def _compute_sample_difficulties(self) -> List[float]:
        """Compute difficulty scores for each sample using multiple strategies."""
        difficulties = []
        
        for image_path in self.image_paths:
            detections = self.annotations[image_path]
            
            if len(detections) > 0:
                # Use multiple difficulty metrics and combine them
                area_difficulty = self._difficulty_by_area(detections)
                count_difficulty = self._difficulty_by_count(detections)
                ratio_difficulty = self._difficulty_by_ratio(detections)
                
                # Combine difficulties (weighted average)
                combined_difficulty = (
                    0.4 * area_difficulty + 
                    0.3 * count_difficulty + 
                    0.3 * ratio_difficulty
                )
                difficulties.append(combined_difficulty)
            else:
                difficulties.append(0.5)  # Default medium difficulty for empty images
        
        return difficulties
    
    def _difficulty_by_area(self, detections: sv.Detections) -> float:
        """
        Compute difficulty based on detection area (smaller = harder).
        
        Args:
            detections: Detection annotations
            
        Returns:
            Difficulty score (0.0 = easiest, 1.0 = hardest)
        """
        if len(detections) == 0:
            return 0.5
        
        # Calculate average area of all detections
        areas = []
        for bbox in detections.xyxy:
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            areas.append(area)
        
        avg_area = sum(areas) / len(areas)
        
        # Normalize area to difficulty (smaller area = higher difficulty)
        # Use a reference area of 200x200 pixels
        reference_area = 200 * 200
        normalized_difficulty = 1.0 - min(1.0, avg_area / reference_area)
        
        return normalized_difficulty
    
    def _difficulty_by_count(self, detections: sv.Detections) -> float:
        """
        Compute difficulty based on number of detections (more = easier).
        
        Args:
            detections: Detection annotations
            
        Returns:
            Difficulty score (0.0 = easiest, 1.0 = hardest)
        """
        if detections.xyxy is None:
            return 0.5
        
        # More detections = lower difficulty
        # Normalize to a reasonable range (0-10 detections)
        max_expected_detections = 10
        count_difficulty = 1 - min(1.0, len(detections.xyxy) / max_expected_detections)
        
        return count_difficulty
    
    def _difficulty_by_ratio(self, detections: sv.Detections,) -> float:
        """
        Compute difficulty based on ratio between bbox and image dimensions.
        
        Args:
            detections: Detection annotations
            image_path: Path to the image
            
        Returns:
            Difficulty score (0.0 = easiest, 1.0 = hardest)
        """
        if len(detections) == 0:
            return 0.5
        
        
        # Reference dimensions
        img_height, img_width = 800,800
        img_area = img_height * img_width
        
        # Calculate total bbox area
        total_bbox_area = 0
        for bbox in detections.xyxy:
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            total_bbox_area += bbox_area
        
        # Calculate coverage ratio
        coverage_ratio = total_bbox_area / img_area
        
        # Normalize ratio to difficulty
        # Very small coverage (hard to detect) = high difficulty
        # Very large coverage (easy to detect) = low difficulty
        # Medium coverage = medium difficulty
        
        if coverage_ratio < 0.01:  # Very small objects
            ratio_difficulty = 1.0
        elif coverage_ratio > 0.5:  # Very large objects
            ratio_difficulty = 0.0
        else:
            # Linear interpolation for medium ratios
            ratio_difficulty = 1.0 - (coverage_ratio - 0.01) / (0.5 - 0.01)
            ratio_difficulty = max(0.0, min(1.0, ratio_difficulty))
        
        return ratio_difficulty
            
    def update_curriculum_state(self, difficulty: float):
        """
        Update dataset based on current curriculum state.
        
        Args:
            difficulty: Current difficulty level (0.0 = easiest, 1.0 = hardest)
        """
        self.current_difficulty = difficulty
        
        # Update available samples based on difficulty
        if self.curriculum_config and self.curriculum_config.enabled:
            self._update_available_samples()
    
    def _update_available_samples(self):
        """Update which samples are available based on current difficulty."""
        if not self.curriculum_config or self.curriculum_config.type != "difficulty":
            self.available_indices = list(range(len(self.image_paths)))
            return
        
        self.available_indices = []
        
        for i, difficulty in enumerate(self.sample_difficulties):
            # Include samples that are not harder than current difficulty level
            if difficulty <= self.current_difficulty + 0.1:  # Small buffer
                self.available_indices.append(i)
        
        # Ensure we always have some samples
        if len(self.available_indices) == 0:
            self.available_indices = list(range(len(self.image_paths)))
    
    def __len__(self):
        return len(self.available_indices)
    
    def __getitem__(self, idx):
        # Map to actual sample index
        actual_idx = self.available_indices[idx]
        
        # Get image path
        image_path = self.image_paths[actual_idx]
        
        # Load image using supervision's lazy loading
        image = self._get_image(image_path)
        
        # Get annotations
        detections = self.annotations[image_path]
        
        # Convert to PIL and apply transforms
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform to tensor
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Return the expected tuple format for supervision compatibility
        return image_path, image, detections

class CropDataset(torch.utils.data.Dataset):
    """
    Custom dataloader for loading cropped regions of interest (ROIs) from images.
    
    This dataloader extracts crops around detection bounding boxes and returns
    the cropped image along with the corresponding label. Crops are generated
    lazily for memory efficiency.
    """
    
    def __init__(self, 
                 dataset: CurriculumDetectionDataset,
                 crop_size: int = 224,
                 max_tn_crops: int = 1,
                 p_draw_annotations: float = 0.,
                 ):
        """
        Initialize crop dataloader.
        
        Args:
            dataset: CurriculumDetectionDataset to load crops from
            crop_size: Size of the square crop (width = height)
            expand_ratio: Ratio to expand bounding box (0.1 = 10% expansion)
            max_crops_per_image: Maximum number of crops to extract per image
            min_crop_size: Minimum size of crop to consider valid
            random_crops: Whether to also generate random crops
            random_crop_prob: Probability of generating a random crop
        """
        super().__init__()
        self.dataset = dataset
        self.crop_size = crop_size
        self.max_tn_crops = max_tn_crops
        self.p_draw_annotations = p_draw_annotations
        
        # Pre-compute crop indices for lazy generation
        self.crop_indices = self._compute_crop_indices()

        self.pad_roi = A.Compose(
            [
                A.PadIfNeeded(
                    min_height=self.crop_size,
                    min_width=self.crop_size,
                    border_mode=0,  # cv2.BORDER_CONSTANT
                    fill=0,
                ),
                A.CenterCrop(
                    height=self.crop_size,
                    width=self.crop_size,
                    p=1,
                ),
            ]
        )
    
    def _compute_crop_indices(self) -> List[Dict[str, Any]]:
        """Pre-compute crop indices for lazy generation."""
        crop_indices = []
        
        for dataset_idx,image_path in tqdm(enumerate(self.dataset.image_paths),desc="Computing crop indices"):
            
            with Image.open(image_path) as img:
                w,h = img.size
            
            detections = self.dataset.annotations[image_path] 

            # Add random crop indices if enabled
            if detections.is_empty():
                random_indices = self._compute_random_indices(
                    image_path, dataset_idx, h, w
                )
                crop_indices.extend(random_indices)
            
            else:
                # Add detection crop indices
                detection_indices = self._compute_detection_indices(
                    detections, image_path, dataset_idx, h, w
                )
                crop_indices.extend(detection_indices)
        
        return crop_indices
    
    def _compute_detection_indices(self, 
                                  detections: sv.Detections, 
                                  image_path: str, 
                                  dataset_idx: int,
                                  img_height: int,
                                  img_width: int) -> List[Dict[str, Any]]:
        """Compute indices for detection-based crops."""
        indices = []
        
        if len(detections) == 0:
            return indices
        
        # Sort detections by area (largest first for better crops)
        areas = []
        for bbox in detections.xyxy:
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            areas.append(area)
        
        # Get indices sorted by area (largest first)
        
        for i, bbox in enumerate(detections.xyxy):
            class_id = detections.class_id[i] if detections.class_id is not None else 0
            
            # Expand bounding box
            x1, y1, x2, y2 = bbox.tolist()          
                        
            # Expand bbox
            x1 = max(0, x1 - self.crop_size)
            y1 = max(0, y1 - self.crop_size)
            x2 = min(img_width, x2 + self.crop_size)
            y2 = min(img_height, y2 + self.crop_size)
                       
            indices.append({
                'dataset_idx': dataset_idx,
                'image_path': image_path,
                'crop_bbox': [x1, y1, x2, y2],
                'original_bbox': bbox,
                'label': class_id,
                'crop_type': 'detection'
            })
        
        return indices
    
    def _compute_random_indices(self, 
                               image_path: str, 
                               dataset_idx: int,
                               img_height: int,
                               img_width: int) -> List[Dict[str, Any]]:
        """Compute indices for random crops."""
        indices = []
        
        # Generate 1-3 random crops
        num_random_crops = np.random.randint(1, 4)
        
        for _ in range(num_random_crops):
           
            
            # Random position
            x1 = np.random.randint(0, max(1, img_width - self.crop_size))
            y1 = np.random.randint(0, max(1, img_height - self.crop_size))
            x2 = x1 + self.crop_size
            y2 = y1 + self.crop_size
            
            indices.append({
                'dataset_idx': dataset_idx,
                'image_path': image_path,
                'crop_bbox': [x1, y1, x2, y2],
                'original_bbox': None,
                'label': -1,  # -1 indicates random crop (no specific label)
                'crop_type': 'random'
            })
        
        return indices
    
    def _extract_crop(self, crop_info: Dict[str, Any]) -> np.ndarray:
        """Extract crop from image using pre-computed indices."""
        dataset_idx = crop_info['dataset_idx']
        x1, y1, x2, y2 = crop_info['crop_bbox']
        
        # Get original image
        _, image, detections = self.dataset[dataset_idx]
        
        # Convert to numpy if needed
        if isinstance(image, torch.Tensor):
            img_np = image.permute(1, 2, 0).numpy()
        else:
            img_np = np.array(image)
        
        if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
        
        # Extract crop
        crop = img_np[int(y1):int(y2), int(x1):int(x2)]
        
        # Pad crop to square
        crop = self.pad_roi(image=crop)["image"]
        
        return crop
    
    def get_annotations_for_filter(self) -> List[Dict[str, Any]]:
        """
        Generate annotations in the format expected by ClassificationRebalanceFilter.
        
        Returns:
            List of annotation dictionaries with roi_id, class_name, class_id, file_name
        """
        annotations = []
        roi_id = 0
        
        for idx, crop_info in enumerate(self.crop_indices):
            # Generate filename
            dataset_idx = crop_info['dataset_idx']
            image_path = crop_info['image_path']
            class_id = crop_info['label']
            
            # Get class name from dataset
            class_name = self.dataset.classes[class_id] if class_id >= 0 else "background"
            
            # Generate filename
            base_name = Path(image_path).stem
            file_name = f"{base_name}_roi_{roi_id:06d}.jpg"
            
            annotation = {
                "roi_id": roi_id,
                "class_name": class_name,
                "class_id": class_id,
                "file_name": file_name,
                "dataset_idx": dataset_idx,
                "crop_bbox": crop_info['crop_bbox'],
                "crop_type": crop_info['crop_type']
            }
            
            # Add original annotation info if available
            if crop_info.get('original_bbox') is not None:
                annotation["original_bbox"] = crop_info['original_bbox']
                annotation["original_annotation_id"] = idx
            
            annotations.append(annotation)
            roi_id += 1
        
        return annotations
    
    def apply_rebalance_filter(self, filter_instance) -> 'CropDataset':
        """
        Apply ClassificationRebalanceFilter to create a balanced dataset.
        
        Args:
            filter_instance: Instance of ClassificationRebalanceFilter
            
        Returns:
            New CropDataset with balanced crop indices
        """
        # Get annotations in the expected format
        annotations = self.get_annotations_for_filter()
        
        # Apply filter
        filtered_annotations = filter_instance(annotations)
        
        # Create new dataset with filtered indices
        filtered_indices = []
        for filtered_ann in filtered_annotations:
            roi_id = filtered_ann['roi_id']
            # Find the corresponding crop index
            for idx, crop_info in enumerate(self.crop_indices):
                if idx == roi_id:
                    filtered_indices.append(crop_info)
                    break
        
        # Create new dataset instance
        new_dataset = CropDataset(
            dataset=self.dataset,
            crop_size=self.crop_size,
            max_tn_crops=self.max_tn_crops,
            p_draw_annotations=self.p_draw_annotations
        )
        
        # Replace crop indices with filtered ones
        new_dataset.crop_indices = filtered_indices
        
        return new_dataset
    
    def __len__(self) -> int:
        """Return number of crops."""
        return len(self.crop_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a crop and its label (lazy generation).
        
        Args:
            idx: Index of the crop
            
        Returns:
            Tuple of (crop_image, label)
        """
        crop_info = self.crop_indices[idx]
        
        # Extract crop lazily
        crop = self._extract_crop(crop_info)
        
        # Convert to tensor
        crop_tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        
        return crop_tensor, crop_info['label']
    
    def get_crop_info(self, idx: int) -> Dict[str, Any]:
        """
        Get detailed information about a crop.
        
        Args:
            idx: Index of the crop
            
        Returns:
            Dictionary with crop information
        """
        return self.crop_indices[idx].copy()
    
    def get_crops_by_class(self, class_id: int) -> List[int]:
        """
        Get indices of crops belonging to a specific class.
        
        Args:
            class_id: Class ID to filter by
            
        Returns:
            List of crop indices
        """
        return [i for i, crop_info in enumerate(self.crop_indices) if crop_info['label'] == class_id]
    
    def get_crops_by_type(self, crop_type: str) -> List[int]:
        """
        Get indices of crops by type (detection or random).
        
        Args:
            crop_type: Type of crop ('detection' or 'random')
            
        Returns:
            List of crop indices
        """
        return [i for i, crop_info in enumerate(self.crop_indices) if crop_info['crop_type'] == crop_type]
