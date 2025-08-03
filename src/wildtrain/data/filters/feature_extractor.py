"""
Feature extraction for image filtering algorithms.

This module provides feature extraction capabilities for clustering
and filtering algorithms in object detection training data selection.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
import timm
from torchvision.transforms.v2 import Compose, Resize, Normalize, ToDtype,ToImage, InterpolationMode
import torch.nn.functional as F

from wildtrain.utils.logging import get_logger


class FeatureExtractor:
    """
    Feature extractor from timm (timm/vit_small_patch16_224.dino).
    """

    def __init__(
        self,
        model_name: str = "timm/vit_small_patch16_224.dino",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        pretrained: bool = True,
        mean: list[float] = [0.554, 0.469, 0.348],
        std: list[float] = [0.203, 0.173, 0.144],
        input_size: int = 224,
        weights: Optional[str] = None,
    ):
        """
        Initialize the feature extractor.
        Args:
            model_name: timm model name (default: 'timm/vit_small_patch16_224.dino')
            device: Device to run inference on ('cpu', 'cuda',)
        """
        self.transform = Compose([
            ToImage(),
            ToDtype(torch.float32),
            Resize(input_size,interpolation=InterpolationMode.NEAREST),
            Normalize(mean=mean, std=std),
        ])
        self.model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0
            )
        self.model.set_input_size(img_size=(input_size,input_size))

        self.model.eval()
        self.model.to(device)
        self.device = device
        if weights:
            self.model.load_state_dict(torch.load(weights))

    @property
    def feature_dim(self) -> int:
        """
        Return the dimension of the extracted feature vector.
        """
        return self.model.embed_dim

    @torch.no_grad()
    def __call__(self, image_paths: List[Union[str, Path]]) -> np.ndarray:
        """
        Extract features from a list of images.
        Args:
            image_paths: List of image file paths
        Returns:
            Features as a numpy array
        """
        images = [Image.open(image_path) for image_path in image_paths]
        images = torch.stack([self.transform(image).to(self.device) for image in images])
        outputs = self.model(images)
        features = outputs.cpu().reshape(len(image_paths), -1).numpy()
        return features
