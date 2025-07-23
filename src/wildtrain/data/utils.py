from PIL import Image
from torchvision import transforms as T
import torch
from omegaconf import DictConfig, OmegaConf
import os
from supervision.dataset.core import DetectionDataset
from typing import Union

def load_image(path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return T.PILToTensor()(image)

def load_coco_dataset(images_directory_path:str, annotations_path:str,):
    return DetectionDataset.from_coco(images_directory_path=images_directory_path,
                                      annotations_path=annotations_path)


def load_yolo_dataset(images_directory_path:str, annotations_directory_path:str,data_yaml_path:str,force_mask:bool=False,is_obb:bool=False):
    return DetectionDataset.from_yolo(images_directory_path=images_directory_path,
                                        annotations_directory_path=annotations_directory_path,
                                        data_yaml_path=data_yaml_path,
                                        force_mask=force_mask,
                                        is_obb=is_obb
                                        )


def load_dataset(config:Union[DictConfig,str,dict]):
    """
    Load a dataset from a config.
    """

    if isinstance(config,DictConfig):
        config = OmegaConf.to_container(config)
    elif isinstance(config,str):
        config = OmegaConf.load(config)
    elif isinstance(config,dict):
        config = DictConfig(config)
    else:
        raise ValueError(f"Invalid config type: {type(config)}")
    
    if config.get("type") == "coco":
        return load_coco_dataset(config.get("images_directory_path"),config.get("annotations_path"))
        
    elif config.get("type") == "yolo":
        return load_yolo_dataset(config.get("images_directory_path"),
                                config.get("annotations_directory_path"),
                                config.get("data_yaml_path"),
                                config.get("force_mask",False),
                                config.get("is_obb",False)
                            )
    else:
        raise ValueError(f"Invalid dataset type: {config.get('type')}")