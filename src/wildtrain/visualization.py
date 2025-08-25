"""
FiftyOne integration for WildDetect.

This module handles dataset creation, visualization, and annotation collection
using FiftyOne for wildlife detection datasets.
"""

import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any, Dict, ItemsView, List, Optional, Union

import fiftyone as fo
from dotenv import load_dotenv
from tqdm import tqdm
from omegaconf import OmegaConf
import supervision as sv
import numpy as np
from wildtrain.models.detector import Detector
from wildtrain.models.classifier import GenericClassifier
import torch
from PIL import Image
import torchvision.transforms.v2 as T

logger = logging.getLogger(__name__)


def add_predictions_from_classifier(dataset_name: str, 
                                    model: GenericClassifier, 
                                    prediction_field: str = "classification_predictions", 
                                    batch_size: int = 32,
                                    debug: bool = False
                                    ):

    """Add predictions from a GenericClassifier to all samples in the FiftyOne dataset using batching."""
    
    
    if dataset_name not in fo.list_datasets():
        raise ValueError(f"Dataset {dataset_name} not found")

    dataset = fo.load_dataset(dataset_name)
    
    samples = list(dataset)
    num_samples = len(samples)
    if debug:
        num_samples = 25
    for i in tqdm(range(0, num_samples, batch_size),desc="Adding predictions",total=num_samples//batch_size):
        batch_samples = samples[i:i+batch_size]
        batch_images = []
        for sample in batch_samples:
            image = Image.open(sample.filepath).convert("RGB")
            image_tensor = T.PILToTensor()(image)
            batch_images.append(image_tensor)
        if not batch_images:
            continue
        batch_tensor = torch.stack(batch_images,)
        preds = model.predict(batch_tensor)
        for sample, pred in zip(batch_samples, preds):
            if pred is not None:
                sample[prediction_field] = fo.Classification(label=pred["class"], confidence=pred["score"])
                sample.save()
    
    dataset.save()

def add_predictions_from_detector(dataset_name: str, 
                                    detector: Detector, 
                                    imgsz: int,
                                    prediction_field: str = "detection_predictions", 
                                    batch_size: int = 32,
                                    debug: bool = False,
                                    ):
    """Add predictions from a Detector to all samples in the FiftyOne dataset using batching."""

    if dataset_name not in fo.list_datasets():
        raise ValueError(f"Dataset {dataset_name} not found")

    dataset = fo.load_dataset(dataset_name)

    samples = list(dataset)
    num_samples = len(samples)
    if debug:
        num_samples = min(25, num_samples)
        samples = samples[:num_samples]
    
    for i in tqdm(range(0, num_samples, batch_size), desc="Adding detection predictions", total=(num_samples + batch_size - 1) // batch_size):
        batch_samples = samples[i:i+batch_size]
        batch_images = []
        
        for sample in batch_samples:
            image = Image.open(sample.filepath).convert("RGB")
            # Convert to tensor and normalize to [0, 1] range
            image_tensor = T.PILToTensor()(image).float() / 255.0
            image_tensor = T.Resize((imgsz,imgsz),interpolation=T.InterpolationMode.NEAREST)(image_tensor)
            batch_images.append(image_tensor)
        
        if not batch_images:
            continue
            
        batch_tensor = torch.stack(batch_images)

        B,C,H,W = batch_tensor.shape
        
        # Get predictions from detector
        detections_list = detector.predict(batch_tensor)
        
        # Process each sample and its corresponding detections
        for sample, detections in zip(batch_samples, detections_list):
            if detections is None or len(detections) == 0:
                # No detections found
                pass
            else:
                # Convert supervision detections to FiftyOne detections
                fo_detections = []
                
                for j in range(len(detections)):
                    # Get bounding box (xyxy format)
                    bbox = detections.xyxy[j].copy()
                    bbox[[0,2]] = bbox[[0,2]]/W
                    bbox[[1,3]] = bbox[[1,3]]/H
                    x1, y1, x2, y2 = bbox.tolist()
                    
                    # Get confidence score
                    confidence = float(detections.confidence[j])
                    
                    # Get class information
                    class_id = int(detections.class_id[j])
                    class_name = detections.metadata["class_mapping"][class_id]
                    
                    # Create FiftyOne detection
                    fo_detection = fo.Detection(
                        label=class_name,
                        bounding_box=[x1, y1, x2 - x1, y2 - y1],  # Convert to [x, y, width, height]
                        confidence=confidence
                    )
                    fo_detections.append(fo_detection)
                
                # Add detections to sample
                sample[prediction_field] = fo.Detections(detections=fo_detections)
            
            sample.save()
    
    dataset.save()
