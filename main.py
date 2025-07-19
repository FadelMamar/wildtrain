import hydra
from omegaconf import DictConfig
from wildtrain.utils.logging import setup_logging
from wildtrain.trainers.detection_trainer import run_detection
from wildtrain.trainers.classification_trainer import run_classification

@hydra.main(config_path="configs", config_name="main", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for WildTrain. Selects and runs the appropriate task based on config.
    """
    setup_logging(cfg)
    task = cfg.task
    if task == "detection":
        run_detection(cfg)
    elif task == "classification":
        run_classification(cfg)
    else:
        raise ValueError(f"Unknown task: {task}")

if __name__ == "__main__":
    main()
