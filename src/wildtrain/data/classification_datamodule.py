import os
from typing import Any, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, ConcatDataset
import lightning as L
from torchvision import transforms as T
from wildata.datasets.roi import load_all_splits_concatenated, ROIDataset
from pathlib import Path
import numpy as np
from tqdm import tqdm
from ..utils.logging import get_logger

logger = get_logger(__name__)


def compute_dataset_stats(dataset: Union[ROIDataset, ConcatDataset], 
                         batch_size: int = 32, 
                         num_workers: int = 0,) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and standard deviation of images in a dataset.
    
    Args:
        dataset: ROIDataset or ConcatDataset containing images
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading
        max_samples: Maximum number of samples to use (None for all samples)
    
    Returns:
        Tuple of (mean, std) tensors with shape (C,) where C is number of channels
    """
    # Create a dataloader with the existing dataset
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        drop_last=False
    )
    
    # Initialize accumulators
    mean_acc = torch.zeros(3)  # Assuming RGB images
    std_acc = torch.zeros(3)
    total_pixels = 0
    num_samples = 0
    
    logger.info("Computing dataset statistics...")
    for batch in tqdm(dataloader, desc="Computing mean/std"):
        if isinstance(batch, (list, tuple)):
            images = batch[0]  # Assume first element is images
        else:
            images = batch
            
        if not isinstance(images, torch.Tensor):
            continue
            
        # Ensure images are in the right format (B, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
            
        batch_size_actual = images.size(0)
                    
        # Compute mean for this batch
        batch_mean = images.mean(dim=[0, 2, 3])  # Mean across batch, height, width
        batch_pixels = images.size(0) * images.size(2) * images.size(3)
        
        # Accumulate mean
        mean_acc += batch_mean * batch_pixels
        total_pixels += batch_pixels
        num_samples += batch_size_actual
    
    # Compute final mean
    mean = mean_acc / total_pixels
    
    # Reset for std computation
    logger.info("Computing standard deviation...")
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        drop_last=False
    )
    
    for batch in tqdm(dataloader, desc="Computing std"):
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
            
        if not isinstance(images, torch.Tensor):
            continue
            
        if images.dim() == 3:
            images = images.unsqueeze(0)
            
        batch_size_actual = images.size(0)
        
        # Compute variance for this batch
        batch_var = ((images - mean.view(1, 3, 1, 1)) ** 2).mean(dim=[0, 2, 3])
        batch_pixels = images.size(0) * images.size(2) * images.size(3)
        
        # Accumulate variance
        std_acc += batch_var * batch_pixels
        num_samples += batch_size_actual
    
    # Compute final std
    std = torch.sqrt(std_acc / total_pixels)
    
    logger.info(f"Dataset statistics computed from {num_samples} samples:")

    
    return mean, std


class ClassificationDataModule(L.LightningDataModule):
    """
    LightningDataModule for multiclass image classification using torchvision datasets.
    """
    def __init__(self, root_data_directory:str, batch_size: int,transforms: Optional[dict[str,Any]]=None):
        super().__init__()
        self.batch_size = batch_size
        self.root_data_directory = Path(root_data_directory).resolve()
        
        self.train_dataset: ROIDataset|ConcatDataset
        self.val_dataset: ROIDataset|ConcatDataset
        self.test_dataset: ROIDataset|ConcatDataset
        self.class_mapping: dict[int, str] = dict()

        self.transforms = transforms
    
    def _get_class_mapping(self, dataset: ROIDataset|ConcatDataset):
        if isinstance(dataset, ConcatDataset):
            if isinstance(dataset.datasets[0], ROIDataset):
                return dataset.datasets[0].class_mapping
        elif isinstance(dataset, ROIDataset):
            return dataset.class_mapping
        else:
            raise ValueError(f"Dataset {type(dataset)} not supported")

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            data = load_all_splits_concatenated(self.root_data_directory, splits=['train', 'val'], transform=self.transforms)
            self.train_dataset = data.pop('train')
            self.val_dataset = data.pop('val')
            self.class_mapping = self._get_class_mapping(self.train_dataset)
            assert self.class_mapping == self._get_class_mapping(self.val_dataset), "Class mapping mismatch between train and val datasets"
        elif stage == "validate":
            self.val_dataset = load_all_splits_concatenated(self.root_data_directory, splits=['val'], transform=self.transforms)['val']
            self.class_mapping = self._get_class_mapping(self.val_dataset)
        elif stage == "test":
            self.test_dataset = load_all_splits_concatenated(self.root_data_directory, splits=['test'], transform=self.transforms)['test']
            self.class_mapping = self._get_class_mapping(self.test_dataset)
        else:
            raise ValueError(f"Stage {stage} not supported")

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Train dataset not found")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("Validation dataset not found")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("Test dataset not found")
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0) 