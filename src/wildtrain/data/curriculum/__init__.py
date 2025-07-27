"""
Simplified Curriculum Learning Interface for WildTrain.

This module provides a unified interface for curriculum learning strategies:
- Difficulty-based curriculum learning
- Multi-scale curriculum learning
- Combined strategies

Usage:
    from wildtrain.data.curriculum import CurriculumConfig, CurriculumDataModuleMixin
    
    # Configure curriculum
    config = CurriculumConfig(
        enabled=True,
        type="both",
        difficulty_strategy="linear",
        multiscale_enabled=True
    )
    
    # Add to data module
    class MyDataModule(CurriculumDataModuleMixin, L.LightningDataModule):
        def __init__(self, curriculum_config=None, **kwargs):
            CurriculumDataModuleMixin.__init__(self, curriculum_config)
            L.LightningDataModule.__init__(self)
            # ... rest of initialization
"""

from .manager import CurriculumConfig, CurriculumManager
from .mixins import CurriculumDataModuleMixin
from .callback import CurriculumCallback
from .dataset import (
    CurriculumDetectionDataset, 
    MultiScaleDetectionDataset,
)

__all__ = [
    "CurriculumConfig",
    "CurriculumManager", 
    "CurriculumDataModuleMixin",
    "CurriculumCallback",
    "CurriculumDetectionDataset",
    "MultiScaleDetectionDataset",
]
