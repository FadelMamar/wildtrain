import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Any, Optional
from omegaconf import DictConfig, OmegaConf
import mlflow
from dotenv import load_dotenv

import mmdet
from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from ..utils.logging import ROOT, get_logger
from .base import ModelTrainer

logger = get_logger(__name__)


class MMDetectionTrainer(ModelTrainer):
    """
    Trainer class for object detection models using MMDetection.
    
    This class handles the training and evaluation of detection models
    using MMDetection framework with MLflow for experiment tracking.
    """
    
    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.mmdet_cfg = Config.fromfile(self.config.model.config_file)
        self.class_mapping = dict()

    def _setup(self) -> None:
        
        with open(self.config.dataset.dataset_info,"r") as file:
            dataset_info = json.load(file)
        
        self.class_mapping = {item['id']:item['name']  for item in dataset_info['classes']}
        
        # number of classes
        if self.config.dataset.load_as_single_class:
            num_classes = 1
        else:
            num_classes = len(self.class_mapping)
            
        self.mmdet_cfg.model.roi_head.bbox_head.num_classes = num_classes
        
        # training runtime
        self.mmdet_cfg.train_cfg.max_epochs = self.config.train.epochs
        self.mmdet_cfg.train_cfg.val_interval = self.config.train.val_interval
        
        setattr(self.mmdet_cfg.optim_wrapper, "optimizer", self.config.train.optimizer)
        setattr(self.mmdet_cfg, "param_scheduler", self.config.train.param_scheduler)
        setattr(self.mmdet_cfg.default_hooks, "checkpoint", self.config.train.checkpointer)
        
        self.mmdet_cfg.visualizer.vis_backends = [{'type': 'MLflowVisBackend', 
                                                   'tracking_uri': self.config.mlflow.tracking_uri, 
                                                   'save_dir': self.config.work_dir, 
                                                   "run_name": self.config.mlflow.run_name,
                                                   'exp_name': self.config.mlflow.experiment_name}
                                                  ]
        
        # Region proposal
        self.mmdet_cfg.model.train_cfg.rpn_proposal.nms = 0.6
        self.mmdet_cfg.model.train_cfg.rpn_proposal.max_per_img = 300
        self.mmdet_cfg.model.train_cfg.rcnn.assigner = dict(type='MaxIoUAssigner',
                                                            pos_iou_thr=self.config.pos_iou_thr,
                                                            neg_iou_thr=self.config.neg_iou_thr,
                                                            min_pos_iou=self.config.min_pos_iou,
                                                            match_low_quality=False,
                                                            ignore_iof_thr=-1)
        
        self.mmdet_cfg.model.test_cfg.rpn.nms.iou_threshold = 0.7
        self.mmdet_cfg.model.test_cfg.rpn.max_per_img = 300
        self.mmdet_cfg.model.test_cfg.rcnn.nms = dict(type='nms', iou_threshold=0.5)
        self.mmdet_cfg.model.test_cfg.rcnn.max_per_img = 300
        
        
        
        # setting dataloaders values
        for name in ["batch_size","num_workers","persistent_workers"]:
            value = getattr(self.config.dataloader, name)
            if value is not None:
                setattr(self.mmdet_cfg.train_dataloader, name, value)
                setattr(self.mmdet_cfg.val_dataloader, name, value)
        
        # setting dataset values
        ## Train loader
        self.mmdet_cfg.train_dataloader.dataset.data_root = self.config.ROOT_DATASET
        self.mmdet_cfg.train_dataloader.dataset.ann_file = self.config.dataset.train_ann
        self.mmdet_cfg.train_dataloader.dataset.data_prefix.img = ""
        self.mmdet_cfg.train_dataloader.dataset.filter_cfg.filter_empty_gt = self.config.dataset.filter_empty_gt.train
        
        ## Val loader
        self.mmdet_cfg.val_dataloader.dataset.data_root = self.config.ROOT_DATASET
        self.mmdet_cfg.val_dataloader.dataset.ann_file = self.config.dataset.val_ann
        self.mmdet_cfg.val_dataloader.dataset.test_mode = True
        self.mmdet_cfg.val_dataloader.dataset.data_prefix.img = ""
        self.mmdet_cfg.val_dataloader.dataset.filter_cfg.filter_empty_gt = self.config.dataset.filter_empty_gt.val
        
        ## evaluator
        
        for name in ["type","ann_file","metric","format_only"]:
            value = getattr(self.config.val_evaluator, name)
            if value is not None:
                setattr(self.mmdet_cfg.val_evaluator, name, value)
        
        
        if self.config.train.amp == True:
            self.mmdet_cfg.optim_wrapper.type = "AmpOptimWrapper"
            self.mmdet_cfg.optim_wrapper.loss_scale = 'dynamic'

        if self.config.train.resume == 'auto':
            self.mmdet_cfg.resume = True
            self.mmdet_cfg.load_from = None
        elif self.config.train.resume is not None:
            self.mmdet_cfg.resume = True
            self.mmdet_cfg.load_from = self.config.train.resume
        
        if self.config.work_dir is not None:
            self.mmdet_cfg.work_dir = self.config.work_dir
        else:
            self.mmdet_cfg.work_dir = str(ROOT / 'work_dirs' / 'mmdet')
        Path(self.mmdet_cfg.work_dir).mkdir(parents=True, exist_ok=True)
        
        # build the runner from config
        if 'runner_type' not in self.mmdet_cfg:
            self.runner = Runner.from_cfg(self.mmdet_cfg)
        else:
            self.runner = RUNNERS.build(self.mmdet_cfg)
        
    def run(self,debug: bool = False) -> None:
        """
        Run object detection training or evaluation using MMDetection.
        
        Args:
            debug: If True, run with limited batches for debugging
        """

        self._setup()
        self.runner.train()
        

