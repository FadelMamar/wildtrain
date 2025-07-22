import torch
import supervision as sv
import torchvision.ops as ops
import numpy as np

from .classifier import GenericClassifier
from .localizer import ObjectLocalizer


class TwoStageDetector(object):
    """
    Two-stage detector for inference: localizes objects and classifies each ROI.
    Args:
        localizer (ObjectLocalizer): The object localization model.
        classifier (GenericClassifier): The classifier for each ROI.
    """

    def __init__(self, localizer: ObjectLocalizer, classifier: GenericClassifier):
        self.localizer = localizer
        self.classifier = classifier

    def predict(self, images: torch.Tensor) -> list[sv.Detections]:
        """Detects objects in a batch of images and classifies each ROI."""
        assert images.dim() == 4, "Input images must be a batch of images"
        assert images.shape[1] == 3, "Input images must be RGB"

        roi_size = self.classifier.input_size.item()

        def resize_bbox(bbox: np.ndarray) -> np.ndarray:
            bbox = bbox.copy()
            h = roi_size[0]
            w = roi_size[1]

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

        detections: list[sv.Detections] = self.localizer.forward(images)
        boxes = []
        for i in range(images.shape[0]):
            det = resize_bbox(detections[i].xyxy)
            roi_boxes = torch.cat(
                [torch.full((det.shape[0], 1), i, dtype=torch.long), det], dim=1
            )
            boxes.append(roi_boxes)
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
        results = []
        for i in range(images.shape[0]):
            det = detections[i]
            class_id = cls_results[i]["class"]
            score = cls_results[i]["score"]
            metadata = det.metadata.copy()
            metadata.update({"class": class_id, "score": score})
            updated_det = sv.Detections(
                xyxy=det.xyxy,
                confidence=det.confidence * score,
                class_id=class_id,
                metadata=metadata,
            )
            results.append(updated_det)

        return results
