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
from pathlib import Path

from .classifier import GenericClassifier
from .localizer import ObjectLocalizer, UltralyticsLocalizer
from ..shared.models import YoloConfig, MMDetConfig, RegistrationBase
from ..utils.mlflow import load_registered_model
from ..utils.logging import get_logger

logger = get_logger(__name__)

class Detector(torch.nn.Module):
    """
    Two-stage detector for inference: localizes objects and classifies each ROI.
    Args:
        localizer (ObjectLocalizer): The object localization model.
        classifier (GenericClassifier): The classifier for each ROI.
    """

    def __init__(self, localizer: ObjectLocalizer, classifier: Optional[GenericClassifier]=None):
        super().__init__()
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
    def from_config(cls,
    localizer_config:Union[YoloConfig,MMDetConfig, DictConfig],
    classifier_ckpt:Optional[Union[str,Path]]=None,
    classifier_export_kwargs:Optional[RegistrationBase]=None
    ):
        if isinstance(localizer_config,MMDetConfig):
            raise NotImplementedError("Support only Yolo.")
        localizer = UltralyticsLocalizer.from_config(localizer_config)
        classifier = None
        if classifier_ckpt is not None:
            assert isinstance(classifier_ckpt,(Path,str)), "classifier_ckpt must be a Path object or a string"
            classifier = GenericClassifier.load_from_checkpoint(classifier_ckpt,map_location=localizer_config.device)
            if classifier_export_kwargs is not None:
                assert isinstance(classifier_export_kwargs,(RegistrationBase,DictConfig)), f"classifier_export_kwargs must be a RegistrationBase object. Received {type(classifier_export_kwargs)}"
                classifier = classifier.export(mode=classifier_export_kwargs.export_format,
                                              batch_size=classifier_export_kwargs.batch_size,
                                              output_path = Path(classifier_ckpt).with_suffix(f".{classifier_export_kwargs.export_format}").as_posix()
                                            )

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
          
    def _to_dict(self,results:List[sv.Detections])->List[Dict]:
        results = [vars(result) for result in results]
        results_as_dict = []
        for result in results:
            result_as_dict = {}
            for k,v in result.items():
                if isinstance(v,np.ndarray):
                    result_as_dict[k] = v.tolist()
                elif isinstance(v,dict):
                    result_as_dict[k] = {k:v[k].tolist() if isinstance(v[k],np.ndarray) else v[k] for k in v}
                else:
                    result_as_dict[k] = v
            results_as_dict.append(result_as_dict)
        return results_as_dict
    
    @staticmethod
    def _to_sv_detections(results:List[Dict])->List[sv.Detections]:
        detections = []
        for result in results:
            detections.append(sv.Detections(
                xyxy=np.array(result["xyxy"]).reshape(-1,4),
                confidence=np.array(result["confidence"]),
                class_id=np.array(result["class_id"]),
                metadata={"class_mapping":result["metadata"].get("class_mapping",{})},
            ))
        return detections

    @staticmethod
    def predict_inference_service(
        batch: torch.Tensor,url:str="http://localhost:4141/predict",timeout:int=15
    ) -> List[sv.Detections]:
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

        return Detector._to_sv_detections(detections)

    def predict(self, images: torch.Tensor,return_as_dict:bool=False) -> Union[List[sv.Detections],List[Dict]]:
        """Detects objects in a batch of images and classifies each ROI."""
        b = images.shape[0]
        detections: list[sv.Detections] = self.localizer.predict(self._pad_if_needed(images))[:b]
        
        if self.classifier is None:
            if return_as_dict:
                detections = self._to_dict(detections)
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
        
        # If no boxes are found, return the detections as is
        if len(boxes)==0:
            if return_as_dict:
                detections = self._to_dict(detections)
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
            results = self._to_dict(results)
            return results

        return results

    def forward(self,images:torch.Tensor)->List[Dict]:
        return self.predict(images,return_as_dict=True)