import os
from typing import Any, Optional
import torch
from torch.utils.data import DataLoader, ConcatDataset
import lightning as L
from torchvision import transforms as T
from wildata.datasets.roi import load_all_splits_concatenated, ROIDataset
from pathlib import Path

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

        if transforms is not None:
            self.transforms = transforms

        else:
            self.transforms = {
                'train': T.Compose([
                    T.ToTensor(),
                ]),
                'val': T.Compose([
                    T.ToTensor(),])
            }
    
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