from typing import Any, Dict, Optional, List
from omegaconf import DictConfig, OmegaConf
import mlflow
from ultralytics import YOLO
from ultralytics import settings
import yaml
import os
import pandas as pd
from pathlib import Path

from ..utils.logging import ROOT, get_logger
from .base import ModelTrainer
from .yolo_utils import CustomYOLO, CustomTrainer

logger = get_logger(__name__)

def load_yaml(path: str):
    with open(path, "r") as file:
        return yaml.safe_load(file)

def save_yaml(cfg: dict, save_path: str, mode="w"):
    with open(save_path, mode, encoding="utf-8") as file:
        yaml.dump(cfg, file)

def save_yolo_yaml_cfg(
    root_dir: str,
    labels_map: dict,
    yolo_train: list | str,
    yolo_val: list | str,
    save_path: str,
    mode="w",
) -> None:
    cfg_dict = {
        "path": root_dir,
        "names": labels_map,
        "train": yolo_train,
        "val": yolo_val,
        "nc": len(labels_map),
    }

    save_yaml(cfg=cfg_dict, save_path=save_path, mode=mode)

def remove_label_cache(data_config_yaml: str):
    """
    Remove the labels.cache files from the dataset directories specified in the YOLO data config YAML.

    Args:
        data_config_yaml (str): Path to the YOLO data config YAML file.
    """
    # Remove labels.cache
    yolo_config = load_yaml(data_config_yaml)
    root = yolo_config["path"]
    for split in ["train", "val", "test"]:
        # try:
        if split in yolo_config.keys():
            for p in yolo_config[split]:
                path = os.path.join(root, p, "../labels.cache")
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Removing: {os.path.join(root, p, '../labels.cache')}")
        else:
            logger.info(f"split={split} does not exist.")

def sample_pos_neg(images_paths: list, ratio: float, seed: int = 41):
    """
    Sample positive and negative image paths based on the ratio of empty to non-empty samples.

    Args:
        images_paths (list): List of image paths.
        ratio (float): Ratio defined as num_empty/num_non_empty.
        seed (int, optional): Random seed. Defaults to 41.

    Returns:
        list: Selected image paths.
    """

    # build dataframe
    is_empty = [
        1 - Path(str(p).replace("images", "labels")).with_suffix(".txt").exists()
        for p in images_paths
    ]
    data = pd.DataFrame.from_dict(
        {"image_paths": images_paths, "is_empty": is_empty}, orient="columns"
    )
    # get empty and non empty
    num_empty = (data["is_empty"] == 1).sum()
    num_non_empty = len(data) - num_empty
    if num_empty == 0:
        logger.info("contains only positive samples")
    num_sampled_empty = min(int(num_non_empty * ratio), num_empty)
    sampled_empty = data.loc[data["is_empty"] == 1].sample(
        n=num_sampled_empty, random_state=seed
    )
    # concatenate
    sampled_data = pd.concat([sampled_empty, data.loc[data["is_empty"] == 0]])

    logger.info(f"Sampling: pos={num_non_empty} & neg={num_sampled_empty}")

    return sampled_data["image_paths"].to_list()

def get_data_cfg_paths_for_cl(
    ratio: float,
    data_config_yaml: str,
    cl_save_dir: str,
    seed: int = 41,
    split: str = "train",
    pattern_glob: str = "*",
):
    """
    Generate and save a YOLO data config YAML for continual learning with sampled images.

    Args:
        ratio (float): Ratio for sampling.
        data_config_yaml (str): Path to YOLO data config YAML.
        cl_save_dir (str): Directory to save sampled images and config.
        seed (int, optional): Random seed. Defaults to 41.
        split (str, optional): Dataset split. Defaults to 'train'.
        pattern_glob (str, optional): Glob pattern for images. Defaults to '*'.

    Returns:
        str: Path to the saved config YAML.
    """

    yolo_config = load_yaml(data_config_yaml)

    root = yolo_config["path"]
    dirs_images = [os.path.join(root, p) for p in yolo_config[split]]

    # sample positive and negative images
    sampled_imgs_paths = []
    for dir_images in dirs_images:
        logger.info(f"Sampling positive and negative samples from {dir_images}")
        paths = sample_pos_neg(
            images_paths=list(Path(dir_images).glob(pattern_glob)),
            ratio=ratio,
            seed=seed,
        )
        sampled_imgs_paths = sampled_imgs_paths + paths

    # save selected images in txt file
    save_path_samples = os.path.join(
        cl_save_dir, f"{split}_ratio_{ratio}-seed_{seed}.txt"
    )
    pd.Series(sampled_imgs_paths).to_csv(save_path_samples, index=False, header=False)
    logger.info(f"Saving {len(sampled_imgs_paths)} sampled images.")

    # save config
    save_path_cfg = Path(save_path_samples).with_suffix(".yaml")
    cfg = dict(root_dir=root, save_path=save_path_cfg, labels_map=yolo_config["names"])
    if split == "train":
        cfg["yolo_val"] = yolo_config["val"]
        cfg["yolo_train"] = os.path.relpath(save_path_samples, start=root)

    elif split == "val":
        cfg["yolo_val"] = os.path.relpath(save_path_samples, start=root)
        cfg["yolo_train"] = yolo_config["train"]

    else:
        raise NotImplementedError

    # save yolo data cfg
    save_yolo_yaml_cfg(mode="w", **cfg)

    logger.info(
        f"Saving samples at: {save_path_samples} and data_cfg at {save_path_cfg}",
    )

    return str(save_path_cfg)


class UltralyticsDetectionTrainer(ModelTrainer):
    """
    Trainer class for object detection models using Ultralytics YOLO.
    This class handles training using parameters from a DictConfig (e.g., from yolo.yaml).
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.model: Optional[YOLO] = None

    def validate_config(self) -> None:
        if (
            self.config.model.architecture_file is None
            and self.config.model.weights is None
        ):
            raise ValueError("Either architecture_file or weights must be provided")

        if self.config.dataset.data_cfg is None:
            raise ValueError("data_cfg must be provided")
    
    def pretrain(self,debug:bool=False):
        """
        Run pretraining phase for the model if enabled in configuration.
        """
        if not os.path.exists(self.config.pretraining.data_cfg):
            raise FileNotFoundError(f"Pretraining data config file not found: {self.config.pretraining.data_cfg}")
        
        logger.info("\n\n------------ Pretraining ----------\n")
        remove_label_cache(self.config.pretraining.data_cfg)
        self.config.name += f"-PTR_freeze_{self.config.train.freeze}"
        self.config.train.epochs = self.config.train.ptr_epochs
        self.config.train.lr0 = self.config.train.ptr_lr0
        self.config.train.lrf = self.config.train.ptr_lrf
        self.config.train.freeze = self.config.train.ptr_freeze

        self.run(debug=debug)
    
    def curriculum_learning(self, img_glob_pattern: str = "*", debug:bool=False):
        """
        Run continual learning strategy for the model.

        Args:
            img_glob_pattern (str, optional): Glob pattern for images.
        """
        if not os.path.exists(self.config.curriculum.data_cfg):
            raise FileNotFoundError(f"Continual learning data config file not found: {self.config.curriculum.data_cfg}")
        
        logger.info("\n\n------------ Continual Learning ----------\n")
        remove_label_cache(self.config.curriculum.data_cfg)

        if self.config.curriculum.save_dir is None:
            self.config.curriculum.save_dir = ROOT / "data" / "curriculum_learning"

        self.config.curriculum.save_dir.mkdir(parents=True, exist_ok=True)

        for flag in (self.config.curriculum.ratios, self.config.curriculum.epochs, self.config.curriculum.freeze):
            assert len(flag) == len(self.config.curriculum.lr0s), (
                f"All cl_* flags should match length. {len(flag)} != {len(self.config.curriculum.lr0s)}"
            )

        original_run_name = self.config.name
        for lr, ratio, num_epochs, freeze in zip(
            self.config.curriculum.lr0s, self.config.curriculum.ratios, self.config.curriculum.epochs, self.config.curriculum.freeze,
        ):
            cl_cfg_path = get_data_cfg_paths_for_cl(
                ratio=ratio,
                data_config_yaml=self.config.curriculum.data_cfg,
                cl_save_dir=self.config.curriculum.save_dir,
                seed=self.config.train.seed,
                split="train",
                pattern_glob=img_glob_pattern,
            )
            self.config.name = f"{original_run_name}-cl_ratio-{ratio}_freeze-{freeze}"
            self.config.train.freeze = freeze
            self.config.train.lr0 = lr
            self.config.train.epochs = num_epochs
            self.config.dataset.data_cfg = cl_cfg_path
            self.run(debug=debug)

    def get_model(self,):
        """Returns a customized detection model instance configured with specified config and weights."""

        # Load model
        if self.config.use_custom_yolo:
            model = CustomYOLO(model=self.config.model.architecture_file or self.config.model.weights,
            task="detect",
            **self.config.custom_yolo_kwargs,
            )
        else:
            model = YOLO(
                model=self.config.model.architecture_file or self.config.model.weights,
            task="detect",
        )

        if self.config.model.weights is not None:
            model.load(self.config.model.weights)

        return model



    def run(self,debug:bool=False) -> None:
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        
        settings.update({"mlflow": True})

        self.validate_config()

        # Load model
        self.model = self.get_model()
        

        # Training parameters
        train_cfg = dict(self.config.train)

        # Run training
        self.model.train(
            single_cls=self.config.dataset.load_as_single_class,
            trainer=CustomTrainer,
            data=self.config.dataset.data_cfg,
            name=self.config.name,
            project=self.config.project,
            time=2/60 if debug else None,
            save=True,
            **train_cfg,
        )
