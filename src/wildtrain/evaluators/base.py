from abc import ABC, abstractmethod
from typing import Any, Dict, Union, List, Generator
from omegaconf import OmegaConf, DictConfig
import pandas as pd
from supervision.metrics import (
    MeanAveragePrecision,
    MeanAverageRecall,
)
import supervision as sv
from copy import deepcopy
from logging import getLogger
import traceback
from .metrics import MyPrecision,MyRecall,MyF1Score

logger = getLogger(__name__)

class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.
    Subclasses must implement the evaluate method.
    """

    def __init__(self, config: Union[Dict[str, Any], str]):
        if isinstance(config, str):
            self.config = DictConfig(OmegaConf.load(config))
        elif isinstance(config, dict):
            self.config = DictConfig(config)
        else:
            raise ValueError(f"Invalid config type: {type(config)}")

        boxes = sv.metrics.core.MetricTarget.BOXES
        average = getattr(sv.metrics.core.AveragingMethod,self.config.metrics.average.upper())

        self.metrics = dict(
            mAP=MeanAveragePrecision(
                boxes,
                class_agnostic=self.config.metrics.class_agnostic,
                class_mapping=None,
            ),
            mAR=MeanAverageRecall(boxes),
            precision=MyPrecision(boxes, averaging_method=average),
            recall=MyRecall(boxes, averaging_method=average),
            f1=MyF1Score(boxes, averaging_method=average),
        )
        
        self.per_image_metrics = deepcopy(self.metrics)
        self.per_image_results = dict()

        self.report: Dict[str, float] = dict()

    def evaluate(
        self,debug:bool=False
    ) -> Dict[str, Any]:
        """
        Evaluate model using parameters from config dict passed via kwargs.
        """
        count = 0
        for results in self._run_inference():
            self._compute_metrics(results)
            count += 1
            if debug and count > 10:
                break

        results = self._get_results()
        try:
            self.report = self._get_report(results)
        except Exception:
            logger.error(f"Error generating report: {traceback.format_exc()}")
            self.report = dict()

        self._reset()
        return results

    @abstractmethod
    def _run_inference(self) -> Generator[Dict[str, List[sv.Detections]], None, None]:
        """
        Run inference and return predictions and ground truth.
        """
        pass

    def _compute_metrics(self, results: Dict[str, List[sv.Detections]]) -> None:
        """
        Compute metrics for a batch of results.
        """
        for metric_name, metric in self.metrics.items():
            for pred, gt in zip(results["predictions"], results["ground_truth"]):
                metric.update(pred, gt)
                
                try:
                    self._record_per_image_stats(pred, gt, metric_name)
                except Exception:
                    logger.info(gt)
                    logger.info(pred)
                    logger.error(f"Error recording per image stats for {gt.metadata['file_path']}. {traceback.format_exc()}")
                    raise 
                
    def _record_per_image_stats(self,pred,gt,metric_name):
        
        # compute per image metrics
        if (pred.xyxy.size == 0) and (gt.xyxy.size == 0):
            self.per_image_results[(gt.metadata['file_path'],metric_name)] = "True-Negative"
            return
            
        elif pred.xyxy.size == 0:
            self.per_image_results[(gt.metadata['file_path'],metric_name)] = ["False-Negative",gt]
            return

        elif gt.xyxy.size == 0:
            self.per_image_results[(gt.metadata['file_path'],metric_name)] = ["False-Positive",pred]
            return

        metric_i = self.per_image_metrics[metric_name]
        metric_i.reset()
        result_i = metric_i.update(pred, gt).compute()
        self.per_image_results[(gt.metadata['file_path'],metric_name)] = result_i.to_pandas().to_dict(orient='records')
        

    def _reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
        self.per_image_results = dict()

    def _get_report(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Generate a summary evaluation report as a pandas DataFrame.
        Includes mAP@50, mAP@75, mAR@1, Precision@50, Recall@50, F1@50.
        Only summary metrics are included, no per-class metrics.
        """

        dfs = {}
        for name, result in results.items():
            df = result.to_pandas()
            for record in df.to_dict(orient='records'):
                dfs.update(record)

        for name in ["f1","precision","recall"]:
            argmax = getattr(results[name],f"{name}_scores").argmax()
            best_score = getattr(results[name],f"{name}_scores")[argmax]
            best_iou = results[name].iou_thresholds[argmax]
            dfs[f'best_{name}'] = {f'{name}_at_{best_iou}': best_score}
            dfs[f"{name}_scores"] = list(zip(results[name].iou_thresholds, getattr(results[name],f"{name}_scores")))

        return dfs

    def _get_results(self) -> Dict[str, Any]:
        return {name: metric.compute() for name, metric in self.metrics.items()}
