import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod
import supervision as sv
from ultralytics import YOLO


class ObjectLocalizer(ABC):
    """
    Abstract base class for object localizers.
    Should implement a forward method that returns bounding boxes for detected objects in a batch of images.
    """

    @abstractmethod
    def predict(self, images: torch.Tensor) -> list[sv.Detections]:
        """ """
        pass


class UltralyticsLocalizer(ObjectLocalizer):
    """
    Object localizer using Ultralytics YOLO models.
    Args:
        weights (str): Path to YOLO weights file or model name.
        device (str): Device to run inference on ('cpu' or 'cuda').
        conf_thres (float): Confidence threshold for detections.
    """
    
    def __init__(
        self,
        weights: str,
        imgsz:int,
        device: str = "cpu",
        conf_thres: float = 0.25,
        iou_thres: float = 0.5,
        overlap_metric='IOU', 
        task="detect",
        max_det=300,
    ):
        super().__init__()

        self.model = YOLO(weights, task=task)
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.overlap_metric = overlap_metric
        self.imgsz = imgsz
        
        self.class_agnostic = False
        
        assert overlap_metric in ["IOU","IOS"]
        self.metrics = {"IOU":sv.detection.utils.iou_and_nms.OverlapMetric.IOU,
                   "IOS":sv.detection.utils.iou_and_nms.OverlapMetric.IOS
                   }
        

    def predict(self, images: torch.Tensor) -> list[sv.Detections]:
        """
        Args:
            images (torch.Tensor): Batch of images, shape (B, C, H, W)
        Returns:
            List (length B) of detections per image. Each detection is a tuple:
                (x1, y1, x2, y2, score, class)
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)
            
        assert images.min()>=0. and images.max()<=1., "Images must be normalized to [0,1]"
            
        predictions = self.model.predict(
            images,
            imgsz=self.imgsz,
            verbose=False,
            conf=0.1,
            iou=0.1,
            device=self.device,
            max_det=self.max_det
        )
        detections = [sv.Detections.from_ultralytics(pred) for pred in predictions]
        detections = [det.with_nms(threshold=self.iou_thres,
                                   overlap_metric=self.metrics[self.overlap_metric],
                                   class_agnostic=self.class_agnostic) for det in detections]
        return detections
