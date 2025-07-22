import os
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
        self.results_dir = self.config.output.results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def train(self):
        logger.info("[Pipeline] Starting training...")
        trainer = UltralyticsDetectionTrainer(DictConfig(self.config))
        trainer.run()
        logger.info("[Pipeline] Training completed.")

    def evaluate(self):
        logger.info("[Pipeline] Starting evaluation...")
        evaluator = UltralyticsEvaluator(config=DictConfig(self.config))
        results = evaluator.evaluate()
        logger.info("[Pipeline] Evaluation completed.")
        report_path = os.path.join(self.results_dir, "eval_report.json")
        evaluator.save_report(report_path)
        return results

    def run(self):
        self.train()
        results = self.evaluate()
        return results
