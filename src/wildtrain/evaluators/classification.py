import json
import traceback
from typing import Union, Optional
from omegaconf import OmegaConf, DictConfig
import torch
from pathlib import Path
from wildtrain.models.classifier import GenericClassifier
from wildtrain.data import ClassificationDataModule
from wildtrain.trainers.classification_trainer import create_transforms
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC
from tqdm import tqdm
from wildtrain.utils.logging import get_logger

logger = get_logger(__name__)


class ClassificationEvaluator:
    def __init__(self, config:Union[str, Path, DictConfig]):
        if isinstance(config, (str, Path)):
            self.config = OmegaConf.load(config)
            assert self.config.split in ["val", "test"], "split must be either val or test"
        else:
            self.config = config    

    def _load_model(self):
        try:
            model = GenericClassifier.load_from_lightning_ckpt(self.config.classifier, map_location=self.config.device)
        except Exception:
            model = GenericClassifier.load_from_checkpoint(self.config.classifier, map_location=self.config.device)
        model.eval()
        return model

    def _load_data(self):
        transforms = create_transforms(self.config.dataset.transforms)
        datamodule = ClassificationDataModule(
            root_data_directory=self.config.dataset.root_data_directory,
            batch_size=self.config.batch_size,
            transforms=transforms,
            load_as_single_class=self.config.dataset.single_class.enable,
            background_class_name=self.config.dataset.single_class.background_class_name,
            single_class_name=self.config.dataset.single_class.single_class_name,
            keep_classes=self.config.dataset.single_class.keep_classes,
            discard_classes=self.config.dataset.single_class.discard_classes,
        )
        datamodule.setup(stage="test" if self.config.split == "test" else "validate")
        if self.config.split == "test":
            return datamodule.test_dataloader()
        else:
            return datamodule.val_dataloader()

    def evaluate(self,debug:bool=False, save_path:Optional[str]=None):
        model = self._load_model().to(self.config.device)
        dataloader = self._load_data()
        num_classes = len(model.label_to_class_map)
        metrics = {
            "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
            "precision": Precision(task="multiclass", num_classes=num_classes, average=None),
            "recall": Recall(task="multiclass", num_classes=num_classes, average=None),
            "f1": F1Score(task="multiclass", num_classes=num_classes, average=None),
            "auroc": AUROC(task="multiclass", num_classes=num_classes),
        }
        for metric in metrics.values():
            metric.to(self.config.device)
        with torch.no_grad():
            count = 0
            for x, y in tqdm(dataloader, desc="Evaluating"):
                x, y = x.to(self.config.device), y.to(self.config.device).long().squeeze(1)
                logits = model(x)
                for metric in metrics.values():
                    metric.update(logits, y)
                count += 1
                if count > 10 and debug:
                    break

        results = {}
        for name, metric in metrics.items():
            score = metric.compute().cpu().flatten()
            results[name] = score.tolist()
            for i, score in enumerate(score):
                cls_name = model.label_to_class_map.get(i, i)
                results[f"{name}_class_{cls_name}"] = score.item()
        
        if save_path:
            try:
                with open(save_path, "w") as f:
                    json.dump(results, f, indent=2)
            except Exception:
                logger.error(f"Error saving report to {save_path}: {traceback.format_exc()}")

        return results
