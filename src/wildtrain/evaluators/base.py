from abc import ABC, abstractmethod
from typing import Any, Dict, Union, List
from omegaconf import OmegaConf, DictConfig
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

    @abstractmethod
    def evaluate(self,) -> List[Dict[str, Any]]:
        """
        Evaluate predictions against ground truth.

        Args:
            predictions: Model predictions.
            ground_truth: Ground truth annotations.
            **kwargs: Additional arguments for evaluation.

        Returns:
            Dictionary of evaluation metrics.
        """
        pass

    def reset(self) -> None:
        """
        Reset any internal state (optional for stateful evaluators).
        """
        pass

    def get_results(self) -> Dict[str, Any]:
        """
        Retrieve computed metrics (optional for stateful evaluators).

        Returns:
            Dictionary of evaluation metrics.
        """
        raise NotImplementedError("get_results is not implemented for this evaluator.") 