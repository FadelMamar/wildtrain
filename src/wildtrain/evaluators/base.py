from abc import ABC, abstractmethod
from typing import Any, Dict, Union, List, Generator
from omegaconf import OmegaConf, DictConfig
import pandas as pd
from supervision.metrics import (
    MeanAveragePrecision,
    MeanAverageRecall,
    Precision,
    Recall,
    F1Score,
)
import supervision as sv


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

        self.metrics = dict(
            mAP=MeanAveragePrecision(
                "boxes",
                class_agnostic=self.config.metrics.class_agnostic,
                class_mapping=None,
            ),
            mAR=MeanAverageRecall("boxes"),
            p=Precision("boxes", averaging_method=self.config.metrics.average),
            r=Recall("boxes", averaging_method=self.config.metrics.average),
            f1=F1Score("boxes", averaging_method=self.config.metrics.average),
        )

        self.report: pd.DataFrame = None

    def evaluate(
        self,
    ) -> Dict[str, Any]:
        """
        Evaluate model using parameters from config dict passed via kwargs.
        """
        for results in self._run_inference():
            self._compute_metrics(results)
        results = self._get_results()
        try:
            self.report = self._get_report(results)
        except Exception as e:
            print(f"Error generating report: {e}")
            self.report = None

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
        for metric in self.metrics.values():
            for pred, gt in zip(results["predictions"], results["ground_truth"]):
                metric.update(pred, gt)

    def _reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()

    def _get_report(self, results: Dict[str, Any]):
        """
        Generate a summary evaluation report as a pandas DataFrame.
        Includes mAP@50, mAP@75, mAR@1, Precision@50, Recall@50, F1@50.
        Only summary metrics are included, no per-class metrics.
        """

        dfs = []
        for name, result in results.items():
            df = result.to_pandas()
            df = df.add_prefix(f"{name}_")
            dfs.append(df)
        report_df = pd.concat(dfs, axis=1)
        return report_df

    def _get_results(self) -> Dict[str, Any]:
        return {name: metric.compute() for name, metric in self.metrics.items()}
