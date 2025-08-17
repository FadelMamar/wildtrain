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
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """
    Feature extractor from timm (timm/vit_small_patch16_224.dino).
    """

    def __init__(
        self,
        backbone: str = "timm/vit_small_patch16_224.dino",
        backbone_source: str = "timm",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        pretrained: bool = True,
    ):
        """
        Initialize the feature extractor.
        Args:
            model_name: timm model name (default: 'timm/vit_small_patch16_224.dino')
            device: Device to run inference on ('cpu', 'cuda',)
        """
        super().__init__()
        if backbone_source != "timm":
            raise ValueError(f"Backbone source must be 'timm', got {backbone_source}")

        self.model = timm.create_model(
                backbone, pretrained=pretrained, num_classes=0
            )
        data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.transform = nn.Sequential(*timm.data.create_transform(**data_cfg))
        self.model.eval()
        self.model.to(device)
        self.device = device

    @property
    def feature_dim(self) -> int:
        """
        Return the dimension of the extracted feature vector.
        """
        return self.model.num_features

    @torch.no_grad()
    def forward(self, image_paths: List[Union[str, Path]]) -> np.ndarray:
        """
        Extract features from a list of images.
        Args:
            image_paths: List of image file paths
        Returns:
            Features as a numpy array
        """
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        images = torch.stack([self.transform(image).to(self.device) for image in images],dim=0)
        outputs = self.model(images)
        features = outputs.cpu().reshape(len(image_paths), -1).numpy()
        return features
