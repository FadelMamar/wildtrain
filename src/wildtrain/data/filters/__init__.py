"""
Filtering algorithms for object detection training data.

This module provides algorithms to select the most unique and valuable
data points for object detection training, focusing on diversity,
quality, and representativeness.
"""

from .algorithms import ClusteringFilter, SizeFilter
from .feature_extractor import FeatureExtractor
from .filter_config import FilterConfig
from .filter_pipeline import FilterPipeline
from .roi_filter_pipeline import ROIFilterPipeline
from .roi_to_coco_converter import ROIToCOCOConverter

__all__ = [
    "FeatureExtractor",
    "FilterConfig",
    "FilterPipeline",
    "ROIFilterPipeline",
    "ROIToCOCOConverter",
    "SizeFilter",
    "ClusteringFilter",
]
