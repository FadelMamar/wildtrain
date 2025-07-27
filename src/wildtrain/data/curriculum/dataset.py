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
import cv2
import supervision as sv
from .manager import CurriculumConfig


class CurriculumDetectionDataset(sv.DetectionDataset):
    """
    Curriculum-aware dataset that inherits from supervision.DetectionDataset.
    
    This dataset can:
    1. Filter samples based on difficulty (for difficulty-based curriculum)
    2. Resize images based on current scale (for multi-scale curriculum)
    3. Work seamlessly with the CurriculumManager
    4. Leverage supervision's lazy loading and memory efficiency
    """
    
    def __init__(self, 
                 classes: List[str],
                 images: Union[List[str], Dict[str, np.ndarray]],
                 annotations: Dict[str, sv.Detections],
                 curriculum_config: Optional[CurriculumConfig] = None,
                 transform=None,
                 compute_difficulties: bool = True):
        """
        Initialize curriculum detection dataset.
        
        Args:
            classes: List of class names
            images: List of image paths or dict of loaded images
            annotations: Dictionary mapping image paths to Detections
            curriculum_config: Curriculum configuration
            transform: Image transformations
            compute_difficulties: Whether to compute sample difficulties
        """
        super().__init__(classes, images, annotations)
        self.curriculum_config = curriculum_config
        self.transform = transform
        
        # Compute sample difficulties if needed
        if compute_difficulties:
            self.sample_difficulties = self._compute_sample_difficulties()
        else:
            self.sample_difficulties = [0.5] * len(self.image_paths)  # Default medium difficulty
        
        # Initialize available indices (all samples initially)
        self.available_indices = list(range(len(self.image_paths)))
        
        # Current curriculum state
        self.current_difficulty = 1.0
        self.current_scale = 1.0
        self.base_size = curriculum_config.base_size if curriculum_config else 416
    
    def _compute_sample_difficulties(self) -> List[float]:
        """Compute difficulty scores for each sample based on detection sizes."""
        difficulties = []
        
        for image_path in self.image_paths:
            detections = self.annotations[image_path]
            
            if len(detections) > 0:
                # Use detection size as difficulty metric (smaller = harder)
                # Get the first detection's bounding box
                bbox = detections.xyxy[0] if len(detections.xyxy) > 0 else [0, 0, 100, 100]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                
                # Normalize area to difficulty (smaller area = higher difficulty)
                normalized_difficulty = 1.0 - min(1.0, area / (200 * 200))
                difficulties.append(normalized_difficulty)
            else:
                difficulties.append(0.5)  # Default medium difficulty
        
        return difficulties
    
    def update_curriculum_state(self, difficulty: float, scale: float):
        """
        Update dataset based on current curriculum state.
        
        Args:
            difficulty: Current difficulty level (0.0 = easiest, 1.0 = hardest)
            scale: Current scale multiplier
        """
        self.current_difficulty = difficulty
        self.current_scale = scale
        
        # Update available samples based on difficulty
        if self.curriculum_config and self.curriculum_config.enabled:
            self._update_available_samples()
    
    def _update_available_samples(self):
        """Update which samples are available based on current difficulty."""
        if not self.curriculum_config or self.curriculum_config.type not in ["difficulty", "both"]:
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
        
        # Apply current scale if multi-scale is enabled
        if (self.curriculum_config and 
            self.curriculum_config.multiscale_enabled and 
            self.curriculum_config.type in ["multiscale", "both"]):
            
            target_size = int(self.base_size * self.current_scale)
            image, detections = self._apply_multiscale(image, detections, target_size)
        
        # Convert to PIL and apply transforms
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform to tensor
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': image,
            'detections': detections,
            'difficulty': self.sample_difficulties[actual_idx],
            'scale': self.current_scale,
            'sample_idx': actual_idx,
            'image_path': image_path
        }
    
    def _apply_multiscale(self, image: np.ndarray, detections: sv.Detections, target_size: int) -> Tuple[np.ndarray, sv.Detections]:
        """Apply multi-scale transformation to image and detections."""
        h, w = image.shape[:2]
        
        # Calculate scale to fit the longer side
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create square canvas if needed
        if new_h != target_size or new_w != target_size:
            canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            y_offset = (target_size - new_h) // 2
            x_offset = (target_size - new_w) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
            resized_image = canvas
        else:
            y_offset, x_offset = 0, 0
        
        # Adjust detections
        if len(detections) > 0:
            # Scale bounding boxes
            scaled_xyxy = detections.xyxy * scale
            scaled_xyxy[:, [0, 2]] += x_offset  # Add x offset
            scaled_xyxy[:, [1, 3]] += y_offset  # Add y offset
            
            # Create new detections object
            adjusted_detections = sv.Detections(
                xyxy=scaled_xyxy,
                confidence=detections.confidence,
                class_id=detections.class_id
            )
        else:
            adjusted_detections = detections
        
        return resized_image, adjusted_detections


class MultiScaleDetectionDataset(CurriculumDetectionDataset):
    """
    Dataset for multi-scale object detection with curriculum learning.
    
    This dataset extends CurriculumDetectionDataset to handle bounding box annotations
    and multi-scale resizing for object detection tasks.
    """
    
    def __init__(self, 
                 classes: List[str],
                 images: Union[List[str], Dict[str, np.ndarray]],
                 annotations: Dict[str, sv.Detections],
                 curriculum_config: Optional[CurriculumConfig] = None,
                 transform=None,
                 preserve_aspect_ratio: bool = True,
                 pad_to_square: bool = True):
        """
        Initialize multi-scale detection dataset.
        
        Args:
            classes: List of class names
            images: List of image paths or dict of loaded images
            annotations: Dictionary mapping image paths to Detections
            curriculum_config: Curriculum configuration
            transform: Image transformations
            preserve_aspect_ratio: Whether to preserve aspect ratio when resizing
            pad_to_square: Whether to pad images to square
        """
        super().__init__(classes, images, annotations, curriculum_config, transform, compute_difficulties=True)
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.pad_to_square = pad_to_square
    
    def _apply_multiscale(self, image: np.ndarray, detections: sv.Detections, target_size: int) -> Tuple[np.ndarray, sv.Detections]:
        """Apply multi-scale transformation with aspect ratio preservation."""
        h, w = image.shape[:2]
        
        if self.preserve_aspect_ratio:
            # Calculate scale to fit the longer side
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
        else:
            # Direct resize
            scale_h, scale_w = target_size / h, target_size / w
            new_h, new_w = target_size, target_size
        
        # Resize image
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to square if needed
        if self.pad_to_square and (new_h != target_size or new_w != target_size):
            # Create square canvas
            canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            
            # Center the image
            y_offset = (target_size - new_h) // 2
            x_offset = (target_size - new_w) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
            
            resized_image = canvas
        else:
            y_offset, x_offset = 0, 0
        
        # Adjust detections
        if len(detections) > 0:
            if self.preserve_aspect_ratio:
                # Scale coordinates uniformly
                scaled_xyxy = detections.xyxy * scale
            else:
                # Different scaling for width and height
                scaled_xyxy = detections.xyxy.copy()
                scaled_xyxy[:, [0, 2]] *= scale_w  # Scale x coordinates
                scaled_xyxy[:, [1, 3]] *= scale_h  # Scale y coordinates
            
            # Add offsets
            scaled_xyxy[:, [0, 2]] += x_offset
            scaled_xyxy[:, [1, 3]] += y_offset
            
            # Create new detections object
            adjusted_detections = sv.Detections(
                xyxy=scaled_xyxy,
                confidence=detections.confidence,
                class_id=detections.class_id
            )
        else:
            adjusted_detections = detections
        
        return resized_image, adjusted_detections
