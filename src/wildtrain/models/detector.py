import torch
import supervision as sv
import torchvision.ops as ops
import numpy as np
from typing import Optional, Union
from copy import deepcopy
from collections import defaultdict
from omegaconf import DictConfig

from .classifier import GenericClassifier
from .localizer import ObjectLocalizer, UltralyticsLocalizer
from ..shared.models import YoloConfig, MMDetConfig


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
    
    @classmethod
    def from_config(cls,localizer_config:Union[YoloConfig,MMDetConfig, DictConfig],classifier_ckpt:Optional[str]=None):
        if isinstance(localizer_config,MMDetConfig):
            raise NotImplementedError("Support only Yolo.")
        localizer = UltralyticsLocalizer.from_config(localizer_config)
        classifier = None
        if isinstance(classifier_ckpt,str):
            classifier = GenericClassifier.load_from_checkpoint(classifier_ckpt,map_location=localizer_config.device) 
        return cls(localizer=localizer,classifier=classifier)

    def predict(self, images: torch.Tensor,return_as_dict:bool=True) -> list[sv.Detections]:
        """Detects objects in a batch of images and classifies each ROI."""
        assert images.dim() == 4, "Input images must be a batch of images"
        assert images.shape[1] == 3, "Input images must be RGB"

        detections: list[sv.Detections] = self.localizer.predict(images)

        if self.classifier is None:
            return detections

        roi_size = self.classifier.input_size.item()
        def resize_bbox(bbox: np.ndarray) -> np.ndarray:
            
            if bbox.size == 0:
                return bbox
            
            bbox = bbox.copy()
            h = roi_size
            w = roi_size

            bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2.0 - w / 2.0
            bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2.0 - h / 2.0
            bbox[:, 2] = bbox[:, 0] + w
            bbox[:, 3] = bbox[:, 1] + h

            _, _, H, W = images.shape
            bbox[:, 0] = np.clip(bbox[:, 0], 0, W - 1)
            bbox[:, 1] = np.clip(bbox[:, 1], 0, H - 1)
            bbox[:, 2] = np.clip(bbox[:, 2], 0, W - 1)
            bbox[:, 3] = np.clip(bbox[:, 3], 0, H - 1)
            return bbox
        
        boxes = []
        for i in range(images.shape[0]):
            det = resize_bbox(detections[i].xyxy)
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
            #results = [{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items()} for result in results]
            return results


        return results
