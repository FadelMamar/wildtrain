import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
import timm
from typing import Optional,Tuple

from ..utils.logging import get_logger

logger = get_logger(__name__)


class GenericClassifier(nn.Module):
    """
    Generic classifier model supporting timm and MMPretrain backbones.
    """

    def __init__(
        self,
        label_to_class_map: dict,
        backbone: str = "timm/vit_base_patch14_reg4_dinov2.lvd142m",
        backbone_source: str = "timm",
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = True,
        input_size: int = 224,
        mean: list[float] = [0.554, 0.469, 0.348],
        std: list[float] = [0.203, 0.173, 0.144],
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.label_to_class_map = label_to_class_map

        # register buffers
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
        self.register_buffer("input_size", torch.tensor([input_size]).int())
        self.register_buffer(
            "num_classes", torch.tensor([len(label_to_class_map)]).int()
        )

        self._check_arguments()

        self.backbone = backbone
        self.backbone_source = backbone_source
        self.pretrained = pretrained

        self.backbone_trfs,self.feature_extractor = self._get_backbone()
        self.fc = torch.nn.Sequential(
            torch.nn.LazyLinear(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.LazyLinear(self.num_classes.item()),
        )
        # Initialize preprocessing
        self.preprocessing = self._set_preprocessing()
        return None

    def _set_preprocessing(
        self,
    ):
        preprocessing = torch.nn.Sequential(
            T.Resize([self.input_size.item(), self.input_size.item()], interpolation=T.InterpolationMode.BICUBIC),
            T.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
        )
        return preprocessing

    def _check_arguments(
        self,
    ):
        if self.num_classes is None:
            raise ValueError("num_classes must be provided")
        if not isinstance(self.input_size, torch.Tensor):
            raise ValueError("input_size must be a tensor")
        if not isinstance(self.mean, torch.Tensor):
            raise ValueError("mean must be a tensor")
        if not isinstance(self.std, torch.Tensor):
            raise ValueError("std must be a tensor")

    def _get_backbone(
        self,
    ) -> Tuple[nn.Module,nn.Module]:
        if self.backbone_source == "timm":
            model = timm.create_model(
                self.backbone, pretrained=self.pretrained, num_classes=0
            )
            data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
            transform = timm.data.create_transform(**data_cfg)
            trfs = nn.Sequential(*[t for t in transform.transforms if isinstance(t, T.Normalize)])

            try:
                model.set_input_size((self.input_size.item(),self.input_size.item()))
            except Exception:
                logger.info(f"Backbone {self.backbone} does not support setting input size")
                pass
        else:
            raise ValueError(f"Unsupported backbone source: {self.backbone_source}")

        if self.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            #model.eval()
            logger.info(f"Backbone {self.backbone} frozen")
        return trfs,model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocessing(x.float())
        x = self.backbone_trfs(x)
        if "vit" in self.backbone: # get CLS token for ViT models
            x = self.feature_extractor.forward_features(x)[:,0,:]
        else:
            x = self.feature_extractor(x)
        return self.fc(x)
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> list[dict]:
        self.eval()
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        labels = probs.argmax(dim=1).cpu().tolist()
        classes = [
            self.label_to_class_map[i] for i in labels
        ]
        scores = probs.max(dim=1).values.cpu().tolist()
        return [{"class": c, "score": s, "class_id":l} for c, s, l in zip(classes, scores, labels)]
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, map_location: str = "cpu"):
        try:
            return cls._load_from_lightning_ckpt(checkpoint_path=checkpoint_path, map_location=map_location)
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path} using lightning. {e}")
            return cls._load_from_checkpoint(checkpoint_path=checkpoint_path, map_location=map_location,)

    @classmethod
    def _load_from_lightning_ckpt(cls, checkpoint_path: str, map_location: str = "cpu"):
        """Load from a PyTorch Lightning checkpoint by extracting the underlying model."""
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Check if this is a Lightning checkpoint
        if 'state_dict' in checkpoint and 'hyper_parameters' in checkpoint:
            # Extract the model state dict from the Lightning checkpoint
            state_dict = checkpoint['state_dict']
            
            # Remove the 'model.' prefix from state dict keys
            model_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
            
            # Get hyperparameters from the Lightning module
            hparams = checkpoint['hyper_parameters']
            
            # Create model instance using saved hyperparameters
            model = cls(
                label_to_class_map=hparams['label_to_class_map'],
                backbone=hparams['backbone'],
                backbone_source=hparams['backbone_source'],
                input_size=hparams['input_size'],
                mean=hparams['mean'],
                std=hparams['std'],
            )
            # Initialize the model
            model(torch.rand(1, 3, hparams['input_size'], hparams['input_size']))  
            
            # Load the model weights
            model.load_state_dict(model_state_dict)
            return model
        else:
            raise ValueError("Checkpoint does not appear to be a valid PyTorch Lightning checkpoint")

    @classmethod
    def _load_from_checkpoint(cls, checkpoint_path: Optional[str]=None, map_location: str = "cpu",state_dict:Optional[dict]=None) -> 'GenericClassifier':
        
        if state_dict is None:
            assert checkpoint_path is not None, "checkpoint_path must be provided if state_dict is not provided"
            state_dict = torch.load(checkpoint_path, map_location=map_location)

        label_to_class_map = state_dict["label_to_class_map"]
        backbone = state_dict["backbone"]
        backbone_source = state_dict["backbone_source"]
        input_size = state_dict["input_size"].item()
        model = cls(
            label_to_class_map=label_to_class_map,
            backbone=backbone,
            backbone_source=backbone_source,
            input_size=input_size,
        )

        # Initialize the model
        model(torch.rand(1, 3,input_size, input_size))

        model.load_state_dict(state_dict)

        return model
