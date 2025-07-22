import os
import json
from typing import Optional
from omegaconf import OmegaConf, DictConfig
from wildtrain.trainers.classification_trainer import ClassifierTrainer
from wildtrain.evaluators.classification import ClassificationEvaluator
from wildtrain.utils.logging import get_logger

logger = get_logger(__name__)

class ClassificationPipeline:
    """
    Orchestrates the full classification pipeline: training, evaluation, and report saving.
    """
    def __init__(self, config_path: str):
        self.config = OmegaConf.load(config_path)
        self.results_dir = os.path.join("results", "classification")
        os.makedirs(self.results_dir, exist_ok=True)
        self.config_path = config_path
        self.best_model_path: Optional[str] = None

    def train(self):
        logger.info("[Pipeline] Starting classification training...")
        trainer = ClassifierTrainer(DictConfig(self.config))
        trainer.run(debug=self.config.debug)
        logger.info("[Pipeline] Training completed.")
        self.best_model_path = trainer.best_model_path

    def evaluate(self):
        logger.info("[Pipeline] Starting classification evaluation...")
        assert self.best_model_path is not None, "Best model path is not set by trainer"
        # Prepare evaluation config
        eval_config = DictConfig({
            **self.config,
            "classifier": self.best_model_path,
            "device": self.config.eval.device,
            "split": self.config.eval.split,
            "batch_size": self.config.eval.batch_size,
        })
        evaluator = ClassificationEvaluator(eval_config)
        results = evaluator.evaluate(debug=self.config.debug)
        report_path = os.path.join(self.results_dir, "eval_report.json")
        evaluator.save_report(report_path)
        logger.info("[Pipeline] Evaluation completed.")
        return results

    def run(self):
        if self.config.mode == "train":
            self.train()
        results = self.evaluate()
        return results
