import os
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from wildtrain.trainers.detection_trainer import UltralyticsDetectionTrainer
from wildtrain.evaluators.ultralytics import UltralyticsEvaluator
from wildtrain.utils.logging import get_logger

logger = get_logger(__name__)

class DetectionPipeline:
    """
    Orchestrates the full object detection pipeline: training, evaluation, and report saving.
    """
    def __init__(self, config_path: str):
        self.config = OmegaConf.load(config_path)
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)

    def train(self):
        logger.info("[Pipeline] Starting training...")
        train_config = OmegaConf.load(self.config.train.config)
        trainer = UltralyticsDetectionTrainer(DictConfig(train_config))
        trainer.run(debug=self.config.train.debug)
        logger.info("[Pipeline] Training completed.")

    def evaluate(self):
        logger.info("[Pipeline] Starting evaluation...")
        eval_config = OmegaConf.load(self.config.eval.config)
        evaluator = UltralyticsEvaluator(config=DictConfig(eval_config))
        results = evaluator.evaluate(debug=self.config.eval.debug)
        logger.info("[Pipeline] Evaluation completed.")
        
        report_path = os.path.join(self.config.results_dir, "eval_report.json")
        evaluator.save_report(report_path)
        return results

    def run(self):
        if not self.config.disable_train:
            self.train()
        else:
            logger.info("[Pipeline] Training disabled.")

        if not self.config.disable_eval:
            results = self.evaluate()
            return results
        else:
            logger.info("[Pipeline] Evaluation disabled.")
            return None
