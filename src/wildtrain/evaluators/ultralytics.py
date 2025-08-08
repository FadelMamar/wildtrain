from typing import Any, Dict, Tuple, Union, List, Generator

import torch
from ultralytics.data.build import build_yolo_dataset
from ultralytics.utils import IterableSimpleNamespace
from torch.utils.data import DataLoader
from ultralytics import YOLO
import os
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import supervision as sv

from .base import BaseEvaluator
from ..models.detector import Detector
from ..models.localizer import UltralyticsLocalizer
from ..models.classifier import GenericClassifier


class UltralyticsEvaluator(BaseEvaluator):
    """
    Evaluator for Ultralytics YOLO models.
    Implements evaluation logic using Ultralytics metrics.
    """

    def __init__(self, config: Union[DictConfig, str, dict]):
        super().__init__(config)
        self.model = self._load_model()
        self.dataloader = self._create_dataloader()    

    def _load_model(self) -> Any:
        # Create a flattened config for the localizer that includes eval parameters
        localizer_config = OmegaConf.create({
            'weights': str(self.config.weights.localizer),
            'imgsz': self.config.eval.imgsz,
            'device': self.config.device,
            'conf_thres': self.config.eval.conf,
            'iou_thres': self.config.eval.iou,
            'overlap_metric': 'IOU',
            'task': self.config.eval.task,
            'max_det': self.config.eval.max_det,
        })
        
        localizer = UltralyticsLocalizer.from_config(localizer_config)
        
        classifier = None
        if self.config.weights.classifier:
            classifier = GenericClassifier.load_from_checkpoint(
                self.config.weights.classifier, map_location=self.config.device
            )

        return Detector(localizer, classifier)

    def _create_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for evaluation using Ultralytics' build_yolo_dataset utility.

        Args:
            data_config: Dictionary from YOLO data YAML (contains names, path, val/test, etc.).
            eval_config: Dictionary of evaluation parameters (imgsz, batch_size, etc.).

        Returns:
            PyTorch DataLoader for the evaluation split.
        """
        data_config = DictConfig(OmegaConf.load(self.config.data))
        # Convert DictConfig to dict properly to preserve all fields
        eval_config = OmegaConf.to_container(self.config.eval, resolve=True)

        names = data_config.get("names")
        root_path = data_config.get("path")
        split = eval_config.get("split")
        if split is None:
            raise ValueError(f"No 'split' found in eval_config: {eval_config}")
        img_path = os.path.join(root_path, data_config.get(split))
        batch_size = eval_config.get("batch_size", 16)
        mode = "val" if "val" in data_config else "test"

        assert os.path.exists(img_path), f"{img_path} does not exist."

        # Prepare cfg for build_yolo_dataset
        cfg = IterableSimpleNamespace(mask_ratio=1, overlap_mask=False, **eval_config)

        dataset = build_yolo_dataset(
            cfg=cfg,
            img_path=img_path,
            batch=batch_size,
            data={"names": names, "channels": 3},
            mode=mode,
            rect=getattr(cfg, "rect", False),
            stride=getattr(cfg, "stride", 32),
            multi_modal=getattr(cfg, "multi_modal", False),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=eval_config.get("num_workers", 0),
            pin_memory=torch.cuda.is_available(),
            collate_fn=self._collate_fn,
        )
        return dataloader

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function for YOLO evaluation batches.
        Stacks images into a batch tensor; collects all other fields as lists.
        Discards 'batch_idx'.
        """
        imgs = torch.stack([item["img"] for item in batch], dim=0).float() / 255.0
        im_files = [item["im_file"] for item in batch]
        #ori_shapes = [item["ori_shape"] for item in batch]
        #resized_shapes = [item["resized_shape"] for item in batch]
        #ratio_pads = [item["ratio_pad"] for item in batch]
        cls = [item["cls"] for item in batch]

        bboxes = []
        for item in batch:
            bbox = item["bboxes"]
            if bbox.numel() > 0:  # convert bbox to xyxy format
                bbox[:, [0, 2]] = bbox[:, [0, 2]] * item["resized_shape"][1]  # x w
                bbox[:, [1, 3]] = bbox[:, [1, 3]] * item["resized_shape"][0]  # y h
                bbox[:, [0, 1]] = bbox[:, [0, 1]] - bbox[:, [2, 3]] / 2  # x y
                bbox[:, [2, 3]] = bbox[:, [2, 3]] + bbox[:, [0, 1]]  # x y x y
                bboxes.append(bbox)
            else:
                bboxes.append(bbox)

        return {
            "img": imgs,
            "im_file": im_files,
            # "ori_shape": ori_shapes,
            # "resized_shape": resized_shapes,
            # "ratio_pad": ratio_pads,
            "cls": cls,
            "bboxes": bboxes,
        }

    def _run_inference(self) -> Generator[Dict[str, List[sv.Detections]], None, None]:
        for batch in tqdm(self.dataloader, desc="Running inference"):
            predictions = self.model.predict(batch["img"])

            offset = 0
            if self.config.eval.single_cls:
                # because ultralytics:   0 -> negative class
                # while we use 1 -> positive class
                offset = 1 

            # convert ultralytics detections to supervision detections
            gt_detections = [
                sv.Detections(
                    xyxy=gt.cpu().numpy(),
                    class_id=cls.cpu().int().flatten().numpy() + offset,
                    metadata=dict(file_path=file_path),
                )
                for gt, cls, file_path in zip(
                    batch["bboxes"], batch["cls"], batch["im_file"]
                )
            ]


            yield dict(predictions=predictions, ground_truth=gt_detections)
