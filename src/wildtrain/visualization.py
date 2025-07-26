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
logger = logging.getLogger(__name__)


def add_predictions_from_classifier(dataset_name: str, 
                                    checkpoint_path: str, 
                                    prediction_field: str = "predictions", 
                                    batch_size: int = 32,
                                    device: str = "cpu",
                                    debug: bool = False
                                    ):

    """Add predictions from a GenericClassifier to all samples in the FiftyOne dataset using batching."""
    from wildtrain.models.classifier import GenericClassifier
    import torch
    from PIL import Image
    import torchvision.transforms as T

    dataset = fo.load_dataset(dataset_name,create_if_necessary=False)
    try:
        model = GenericClassifier.load_from_checkpoint(checkpoint_path,map_location=device)
    except Exception:
        model = GenericClassifier.load_from_checkpoint(checkpoint_path,map_location="cpu")
    model.to(device)

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

def add_predictions_from_detector(dataset_name: str, 
                                    checkpoint_path: str, 
                                    prediction_field: str = "predictions", 
                                    batch_size: int = 32,
                                    device: str = "cpu",
                                    debug: bool = False
                                    ):
    pass
