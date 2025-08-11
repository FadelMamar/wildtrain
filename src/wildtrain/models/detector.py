from ast import Tuple
import torch
import supervision as sv
import torchvision.ops as ops
import numpy as np
from typing import Optional, Union, Any, Dict, List
from copy import deepcopy
from collections import defaultdict
from omegaconf import DictConfig
import requests
import base64

from .classifier import GenericClassifier
from .localizer import ObjectLocalizer, UltralyticsLocalizer
from ..shared.models import YoloConfig, MMDetConfig
from ..utils.mlflow import load_registered_model


class Detector(object):
    """
    Two-stage detector for inference: localizes objects and classifies each ROI.
    Args:
        localizer (ObjectLocalizer): The object localization model.
        classifier (GenericClassifier): The classifier for each ROI.
    """

    def __init__(self, localizer: ObjectLocalizer, classifier: Optional[GenericClassifier]=None):
        self.localizer = localizer
        self.classifier = classifier
        self.metadata: Optional[Dict[str,Any]] = None
    
    @property
    def input_shape(self,)->Tuple:
        if self.metadata is None:
            return None
        return (self.metadata["batch"],3,self.metadata['imgsz'],self.metadata['imgsz'])
    
    @property
    def class_mapping(self):
        return self.localizer.class_mapping
    
    @classmethod
    def from_config(cls,localizer_config:Union[YoloConfig,MMDetConfig, DictConfig],classifier_ckpt:Optional[str]=None):
        if isinstance(localizer_config,MMDetConfig):
            raise NotImplementedError("Support only Yolo.")
        localizer = UltralyticsLocalizer.from_config(localizer_config)
        classifier = None
        if isinstance(classifier_ckpt,str):
            classifier = GenericClassifier.load_from_checkpoint(classifier_ckpt,map_location=localizer_config.device) 
        return cls(localizer=localizer,classifier=classifier)
    
    @classmethod
    def from_mlflow(cls,name:str,alias:str,dwnd_location:str=None,mlflow_tracking_uri:str="http://localhost:5000")->"Detector":
        model,metadata = load_registered_model(alias=alias,name=name,
                                                load_unwrapped=True,
                                                dwnd_location=dwnd_location,
                                                mlflow_tracking_url=mlflow_tracking_uri)
        model.metadata = metadata
        return model
    
    def _pad_if_needed(self, batch: torch.Tensor) -> torch.Tensor:
        if self.input_shape is None:
            return batch
        assert len(self.input_shape) == len(batch.shape), f"Expected {len(self.input_shape)}D tensor but received {len(batch.shape)}D tensor"
        assert all([a <= b for a, b in zip(batch.shape[1:], self.input_shape[1:])]), f"Expected {self.input_shape[1:]} but received {batch.shape[1:]}"
        if batch.shape[0] < self.input_shape[0]:
            b, c, h, w = batch.shape
            padded = torch.zeros(self.input_shape)
            padded[:b, :c, :h, :w] = batch.clone()
            batch = padded
        elif batch.shape[0] > self.input_shape[0]:
            raise ValueError(f"Expected {self.input_shape[0]} >= {batch.shape[0]}")

        return batch
    
    def resize_bbox(self,bbox: np.ndarray,roi_size:int,images_width:int,images_height:int) -> np.ndarray:
            
            if bbox.size == 0:
                return bbox
            
            bbox = bbox.copy()
            h = roi_size
            w = roi_size

            bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2.0 - w / 2.0
            bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2.0 - h / 2.0
            bbox[:, 2] = bbox[:, 0] + w
            bbox[:, 3] = bbox[:, 1] + h

            bbox[:, 0] = np.clip(bbox[:, 0], 0, images_width - 1)
            bbox[:, 1] = np.clip(bbox[:, 1], 0, images_height - 1)
            bbox[:, 2] = np.clip(bbox[:, 2], 0, images_width - 1)
            bbox[:, 3] = np.clip(bbox[:, 3], 0, images_height - 1)
            return bbox
          
    @staticmethod
    def predict_inference_service(
        batch: torch.Tensor,url:str="http://localhost:4141/predict",timeout:int=15
    ) -> List[List[Dict]]:
        as_bytes = batch.cpu().numpy().tobytes()
        payload = {
            "tensor": base64.b64encode(as_bytes).decode("utf-8"),
            "shape": list(batch.shape),
        }
        res = requests.post(
            url=url, json=payload, timeout=timeout
        ).json()
        detections = res.get("detections")
        if detections is None:
            raise ValueError(f"Inference service failed: {res}")
        return detections

    def predict(self, images: torch.Tensor,return_as_dict:bool=True) -> list:
        """Detects objects in a batch of images and classifies each ROI."""
        b = images.shape[0]
        detections: list[sv.Detections] = self.localizer.predict(self._pad_if_needed(images))[:b]

        if self.classifier is None:
            return detections

        roi_size = self.classifier.input_size.item()
        _,_,images_height,images_width = images.shape
        boxes = []

        for i in range(images.shape[0]):
            det = self.resize_bbox(detections[i].xyxy,roi_size=roi_size,images_width=images_width,images_height=images_height)
            if det.size == 0:
                continue
            det = torch.Tensor(det)
            roi_boxes = torch.cat(
                [torch.full((det.shape[0], 1), i, dtype=torch.long), det], dim=1
            )
            boxes.append(roi_boxes)
        
        if len(boxes)==0:
            return detections
        
        boxes = torch.cat(boxes, dim=0)
        crops = ops.roi_align(
            images,
            boxes,
            output_size=roi_size,
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=True,
        )

        cls_results = self.classifier.predict(crops)  # (N, num_classes)
        results = deepcopy(detections)
        class_mapping = self.classifier.label_to_class_map

        # Map from image index to list of (detection index in boxes, classifier result)
        img_to_det_indices = defaultdict(list)
        for det_idx, img_idx in enumerate(boxes[:, 0].tolist()):
            img_to_det_indices[int(img_idx)].append(det_idx)

        # Prepare per-image updates
        for img_idx, det_indices in img_to_det_indices.items():
            if len(det_indices) == 0:
                continue
            det = detections[img_idx]
            # Gather classifier results for all detections in this image
            class_ids = np.array([cls_results[j]['class_id'] for j in det_indices])
            scores = np.array([cls_results[j]['score'] for j in det_indices])
            results[img_idx] = sv.Detections(
                xyxy=det.xyxy,
                confidence=det.confidence * scores,
                class_id=class_ids,
                metadata={"class_mapping":class_mapping},
            )
        
        if return_as_dict:
            results = [vars(result) for result in results]
            results = [{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items()} for result in results]
            return results

        return results
