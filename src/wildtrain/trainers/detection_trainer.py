from omegaconf import DictConfig
from .base import ModelTrainer


class DetectorTrainer(ModelTrainer):
    """
    Trainer class for object detection models.
    
    This class handles the training and evaluation of detection models
    using PyTorch Lightning and MLflow for experiment tracking.
    """
    
    def run(self) -> None:
        """
        Run object detection training or evaluation based on config.
        """
        self.validate_config()
        
        # TODO: Implement detection training logic
        # This is a placeholder for the detection training implementation
        # The actual implementation would be similar to ClassifierTrainer
        # but with detection-specific models, data modules, and callbacks
        
        raise NotImplementedError("Detection training is not yet implemented")


# Legacy function for backward compatibility
def run_detection(cfg: DictConfig) -> None:
    """
    Run object detection training or evaluation based on config.
    
    This function is kept for backward compatibility. New code should use
    DetectorTrainer class directly.
    """
    trainer = DetectorTrainer(cfg)
    trainer.run() 