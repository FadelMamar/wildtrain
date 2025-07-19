import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from contextlib import nullcontext

from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC
from typing import Any, Optional, Tuple

class GenericClassifier(nn.Module):
    """
    Generic classifier model supporting timm and MMPretrain backbones.
    """
    def __init__(self, 
    num_classes: int, 
    label_to_class_map: dict,
    backbone: str = 'resnet18', 
    backbone_source: str = 'timm', 
    pretrained: bool = True, 
    dropout: float = 0.2,
    no_grad_backbone: bool = True):

        super().__init__()
        self.backbone = self._get_backbone(backbone, backbone_source, pretrained)
        self.fc = torch.nn.Sequential(
            torch.nn.LazyLinear(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.LazyLinear(num_classes),
        )
        self.no_grad_backbone = no_grad_backbone
        self.num_classes = num_classes
        self.label_to_class_map = label_to_class_map

    def _get_backbone(self, backbone: str, source: str, pretrained: bool)->nn.Module:
        if source == 'timm':
            import timm
            model = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool='avg')
            return model
        elif source == 'mmpretrain':
            from mmpretrain import get_model
            model = get_model(backbone, pretrained=pretrained)
            if hasattr(model, 'head'):
                model.head = nn.Identity()
            else:
                raise ValueError("MMPretrain model does not have a 'head' attribute.")
            return model
        else:
            raise ValueError(f"Unsupported backbone source: {source}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad() if self.no_grad_backbone else nullcontext():
            x = self.backbone(x)
        x = self.fc(x)
        return x

class Classifier(L.LightningModule):
    def __init__(
        self,
        epochs: int,
        model: GenericClassifier,
        threshold: float = 0.5,
        label_smoothing: float = 0.0,
        lr: float = 1e-3,
        lrf: float = 1e-2,
        weight_decay: float = 5e-3,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["threshold", "model"])

        self.model = model
        self.num_classes = model.num_classes
        self.label_to_class_map = model.label_to_class_map

        # metrics
        cfg = dict(task="multiclass", num_classes=self.num_classes, average=None)
        self.accuracy = Accuracy(**cfg)
        self.precision = Precision(threshold=threshold, **cfg)
        self.recall = Recall(threshold=threshold, **cfg)
        self.f1score = F1Score(threshold=threshold, **cfg)
        self.ap = AUROC(**cfg)

        self.metrics = dict(
            accuracy=self.accuracy,
            precision=self.precision,
            recall=self.recall,
            f1score=self.f1score,
        )

        self.label_smoothing = label_smoothing
        

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch:Tuple[torch.Tensor, torch.Tensor], batch_idx:int):
        x, y = batch

        classes = y.cpu().flatten().tolist()
        weight = [
            len(classes) / (classes.count(i) + 1e-6) for i in range(self.num_classes)
        ]
        weight = torch.Tensor(weight).float().clamp(1.0, 1e2).to(y.device)

        logits = self(x)
        loss = F.cross_entropy(
            logits,
            y.long().squeeze(1),
            label_smoothing=self.label_smoothing,
            weight=weight,
        )

        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch:Tuple[torch.Tensor, torch.Tensor], batch_idx:int):
        x, y = batch
        y = y.long().squeeze(1)

        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)

        for name, metric in self.metrics.items():
            metric.update(logits, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        for name, metric in self.metrics.items():
            score = metric.compute().cpu()
            self.log(f"val_{name}", score.mean())
            for i, score in enumerate(score):
                cls_name = self.label_to_class_map.get(i, i)
                self.log(f"val_{name}_class_{cls_name}", score)

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.hparams.epochs,
            T_mult=1,
            eta_min=self.hparams.lr * self.hparams.lrf,
        )
        return [optimizer], [lr_scheduler]