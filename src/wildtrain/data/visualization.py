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
from wildata.datasets.roi import ROIDataset, ConcatDataset
from .utils import load_yolo_dataset
from .classification_datamodule import ClassificationDataModule
import supervision as sv
import numpy as np
logger = logging.getLogger(__name__)


class FiftyOneManager:
    """Manages FiftyOne datasets for wildlife detection."""

    def __init__(
        self,
        dataset_name: str,
        config: Optional[Dict[str, Any]] = None,
        persistent: bool = True,
    ):
        """Initialize FiftyOne manager.

        Args:
            dataset_name: Name of the dataset to use
            config: Optional configuration override
        """
        self.config = config
        self.dataset_name = dataset_name
        self.dataset: fo.Dataset = None
        self.persistent = persistent

        self.prediction_field = "detections"

    def _init_dataset(self):
        """Initialize or load the FiftyOne dataset."""
        try:
            self.dataset = fo.load_dataset(self.dataset_name)
            logger.info(f"Loaded existing dataset: {self.dataset_name}")
        except ValueError:
            self.dataset = fo.Dataset(self.dataset_name, persistent=self.persistent)
            logger.info(f"Created new dataset: {self.dataset_name}")

    def _ensure_dataset_initialized(self):
        """Ensure dataset is initialized before operations."""
        if self.dataset is None:
            self._init_dataset()
    
    
    def load_classification_datamodule(self, root_data_directory: Optional[str] = None, batch_size: Optional[int] = None, transforms: Optional[dict] = None, load_as_single_class: Optional[bool] = None, background_class_name: Optional[str] = None, single_class_name: Optional[str] = None, keep_classes: Optional[list[str]] = None, discard_classes: Optional[list[str]] = None, stage: str = "fit", config_path: Optional[str] = None) -> ClassificationDataModule:
        """Load a ClassificationDataModule and set up datasets. If config_path is provided, load parameters from config file."""
        config: dict[str, Any] = {}
        if config_path is not None:
            loaded_cfg = OmegaConf.load(config_path)
            config = dict(loaded_cfg)
        else:
            # Override config values with directly passed arguments if not None
            config["root_data_directory"] = root_data_directory
            config["batch_size"] = batch_size
            config["transforms"] = transforms
            config["load_as_single_class"] = load_as_single_class
            config["background_class_name"] = background_class_name
            config["single_class_name"] = single_class_name
            config["keep_classes"] = keep_classes
            config["discard_classes"] = discard_classes

        datamodule = ClassificationDataModule(
            root_data_directory=config.get("root_data_directory",""),
            batch_size=config.get("batch_size", 32),
            transforms=config.get("transforms", None),
            load_as_single_class=config.get("load_as_single_class", False),
            background_class_name=config.get("background_class_name", "background"),
            single_class_name=config.get("single_class_name", "wildlife"),
            keep_classes=config.get("keep_classes", None),
            discard_classes=config.get("discard_classes", None),
        )
        datamodule.setup(stage)
        return datamodule

    def _create_classification_sample(self, image_path: str, label: int, class_mapping: dict[int, str]):
        """Create a FiftyOne sample for classification."""
        class_name = class_mapping[label] if label in class_mapping else str(label)
        sample = fo.Sample(
            filepath=str(image_path),
            ground_truth=fo.Classification(label=class_name)
        )
        return sample
    
    def import_classification_dataset(self, datamodule: ClassificationDataModule, split: str = "val"):
        """Import a classification dataset from a ClassificationDataModule into FiftyOne."""
        self._ensure_dataset_initialized()
        # Select dataset split
        if split == "train":
            dataset = datamodule.train_dataset
        elif split == "val":
            dataset = datamodule.val_dataset
        elif split == "test":
            dataset = datamodule.test_dataset
        else:
            raise ValueError(f"Unknown split: {split}")

        print(f"len(dataset): {len(dataset)}")

        class_mapping = datamodule.class_mapping
        samples = []

        data: List[ROIDataset] = []
        if isinstance(dataset, ROIDataset):
            data.append(dataset)
        elif isinstance(dataset, ConcatDataset):
            for dataset in dataset.datasets:
                if isinstance(dataset, ROIDataset):
                    data.append(dataset)
                else:
                    raise ValueError(f"Invalid dataset type: {type(dataset)}")
        else:
            raise ValueError(f"Invalid dataset type: {type(dataset)}")
        
        for dataset in data:
            for i in range(len(dataset)):
                image_path = dataset.get_image_path(i)
                label = dataset.get_label(i)
                sample = self._create_classification_sample(image_path, label, class_mapping)
                samples.append(sample)

        self.dataset.add_samples(samples)
        self.save_dataset()

    def import_classification_datamodule(self, root_data_directory: Optional[str] = None, 
                                        batch_size: Optional[int] = None, transforms: Optional[dict] = None, 
                                        load_as_single_class: Optional[bool] = None, 
                                        background_class_name: Optional[str] = None, 
                                        single_class_name: Optional[str] = None, 
                                        keep_classes: Optional[list[str]] = None, 
                                        discard_classes: Optional[list[str]] = None, 
                                        config_path: Optional[str] = None, 
                                        split: str = "val"):
        """User-facing API: Load and import a classification dataset for visualization."""
        stage = 'fit' if split == "train" else 'validate' if split == "val" else 'test'
        datamodule = self.load_classification_datamodule(
            root_data_directory=root_data_directory,
            batch_size=batch_size,
            transforms=transforms,
            load_as_single_class=load_as_single_class,
            background_class_name=background_class_name,
            single_class_name=single_class_name,
            keep_classes=keep_classes,
            discard_classes=discard_classes,
            stage=stage,
            config_path=config_path,
        )
        self.import_classification_dataset(datamodule, split=split)
    
    def _create_yolo_sample(self, image_path:str,image_data:np.ndarray, detections:sv.Detections, classes:list[str]):

        if len(detections.xyxy) == 0:
            sample = fo.Sample(
                filepath=str(image_path),
            )
            return sample

        detections_list = []
        for i,box in enumerate(detections.xyxy):
            box[:,[0,2]] = box[:,[0,2]]/image_data.shape[1]
            box[:,[1,3]] = box[:,[1,3]]/image_data.shape[0]
            class_id = detections.class_id[i]
            label = classes[class_id]
            fo_detection = fo.Detection(
                label=label,
                bounding_box=box,
            )
            detections_list.append(fo_detection)

        sample = fo.Sample(
            filepath=str(image_path),
            ground_truth=fo.Detections(detections)
        )
        return sample
    
    def import_yolo_dataset(self, data_yaml_path: str , split:str, is_obb: bool = False, force_mask: bool = False):
        data_config = OmegaConf.load(data_yaml_path)
        root_path = data_config.get("path")
        images_directory_path = os.path.join(root_path, data_config.get(split))
        annotations_directory_path = os.path.join(root_path, data_config.get(split))

        assert os.path.exists(images_directory_path), f"Images directory path does not exist: {images_directory_path}"
        assert os.path.exists(annotations_directory_path), f"Annotations directory path does not exist: {annotations_directory_path}"

        dataset = load_yolo_dataset(images_directory_path=images_directory_path,
                                                    annotations_directory_path=annotations_directory_path,
                                                    data_yaml_path=data_yaml_path,
                                                    is_obb=is_obb,
                                                    force_mask=force_mask)
        
        self._ensure_dataset_initialized()

        samples = []
        for file_path,image_data,detections in dataset:
            samples.append(self._create_yolo_sample(file_path,image_data,detections,dataset.classes))

        self.dataset.add_samples(samples)
        self.save_dataset()


    def send_predictions_to_labelstudio(
        self, annot_key: str,label_map: dict, dotenv_path: Optional[str] = None,label_type="detections"
    ):
        """Launch the FiftyOne annotation app."""
        if dotenv_path is not None:
            load_dotenv(dotenv_path, override=True)

        classes = [label_map[i] for i in sorted(label_map.keys())]

        try:
            dataset = fo.load_dataset(self.dataset_name)
            dataset.annotate(
                annot_key,
                backend="labelstudio",
                label_field=self.prediction_field,
                label_type=label_type,
                classes=classes,
                api_key=os.environ["FIFTYONE_LABELSTUDIO_API_KEY"],
                url=os.environ["FIFTYONE_LABELSTUDIO_URL"],
            )
        except Exception:
            logger.error(f"Error exporting to LabelStudio: {traceback.format_exc()}")
            raise
     
    def save_dataset(self):
        """Save the dataset to disk."""
        self._ensure_dataset_initialized()

        if self.dataset is None:
            logger.error("Dataset is not initialized")
            return

        try:
            self.dataset.save()
            logger.info("Dataset saved successfully")
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")

    def close(self):
        """Close the dataset."""
        if self.dataset:
            self.dataset.close()
            logger.info("Dataset closed")

    def add_predictions_from_classifier(self, 
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

        self._ensure_dataset_initialized()
        try:
            model = GenericClassifier.load_from_checkpoint(checkpoint_path,map_location=device)
        except Exception:
            model = GenericClassifier.load_from_checkpoint(checkpoint_path,map_location="cpu")
        model.to(device)

        samples = list(self.dataset)
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
