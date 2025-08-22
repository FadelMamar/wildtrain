import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
import torch.nn.functional as F
from torch.export import Dim

import timm
from typing import Optional,Tuple

from ..utils.logging import get_logger

logger = get_logger(__name__)

class ResizeNormalize(nn.Module):
    def __init__(self, input_size: int, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer('mean', mean.view(1, 3, 1, 1))
        self.register_buffer('std', std.view(1, 3, 1, 1))
        self.input_size = input_size
        assert isinstance(self.input_size,int), f"input_size must be an integer, received {type(self.input_size)}"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x, 
            size=(self.input_size, self.input_size), 
            mode='bicubic', 
            align_corners=False
        )
        # Normalize using registered buffers
        x = (x - self.mean) / self.std
        return x

class BackboneNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', mean.view(1, 3, 1, 1))
        self.register_buffer('std', std.view(1, 3, 1, 1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

class ClassifierHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.2,hidden_dim:int=128,num_layers:int=2):
        super().__init__()
        layers = []
        for i in range(num_layers-1):
            if i == 0:
                layers.append(nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout)
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout)
                ))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

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
        hidden_dim: int = 128,
        num_layers: int = 2,
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
        self.register_buffer("num_layers", torch.tensor([num_layers]).int())
        self.register_buffer("hidden_dim", torch.tensor([hidden_dim]).int())
        self.register_buffer("dropout", torch.tensor([dropout]).float())

        self._check_arguments()

        self.backbone = backbone
        self.backbone_source = backbone_source
        self.pretrained = pretrained

        self.backbone_trfs,self.feature_extractor = self._get_backbone()
        self.fc = ClassifierHead(input_dim=self.feature_extractor.num_features, 
                                num_classes=self.num_classes.item(), 
                                dropout=self.dropout.item(), 
                                hidden_dim=self.hidden_dim.item(), 
                                num_layers=self.num_layers.item())

        # Initialize preprocessing
        self.preprocessing = ResizeNormalize(input_size=self.input_size.item(), mean=self.mean, std=self.std)

        self._onnx_program: Optional[torch.onnx.ONNXProgram] = None
        self._onnx_initialized = False
        return None

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

        if self.backbone_source != "timm":
            ValueError(f"Unsupported backbone source: {self.backbone_source}")

        # Create timm model
        model = timm.create_model(
            self.backbone, pretrained=self.pretrained, num_classes=0,global_pool=''
        )
        data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        norm_mean = torch.tensor(data_cfg.get('mean', self.mean))
        norm_std = torch.tensor(data_cfg.get('std', self.std))
        trfs = BackboneNormalize(norm_mean, norm_std)

        # Set input size    
        try:
            model.set_input_size((self.input_size.item(),self.input_size.item()))
        except Exception:
            logger.info(f"Backbone {self.backbone} does not support setting input size")

        # Freeze backbone
        if self.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            logger.info(f"Backbone {self.backbone} frozen")
        return trfs,model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocessing(x.float())
        x = self.backbone_trfs(x)
        if "vit" in self.backbone: # get CLS token for ViT models
            x = self.feature_extractor(x)[:,0]
        else:
            x = self.feature_extractor(x)
        return self.fc(x)
    
    def export(self,mode:str,batch_size:int=8,dynamic:bool=True)->"GenericClassifier":
        if mode == "torchscript":
            return self.to_torchscript()
        elif mode == "onnx":
            return self.to_onnx(batch_size=batch_size,dynamic=dynamic)
        else:
            raise ValueError(f"Unsupported export mode: {mode}")

    def to_torchscript(self)->"GenericClassifier":
        self.eval()
        for module in [self.feature_extractor, self.fc, self.preprocessing, self.backbone_trfs]:
            if isinstance(module, nn.Module):
                module = torch.jit.script(module)
        return self
    
    def to_onnx(self, 
                output_path: Optional[str]=None,
                input_names=["input"],
                output_names=["output"],
                report:bool=False,
                batch_size:int=8,
                dynamic:bool=True
    )->"GenericClassifier":
        self.eval()
        x = torch.rand(batch_size, 3, self.input_size.item(), self.input_size.item(),dtype=torch.float32)
        self.forward(x)
        cfg = dict(dynamic_shapes={"input": {0: Dim("batch_size")}}) if dynamic else {}
        onnx_program = torch.onnx.export(self, x, output_path, input_names=input_names,
                        output_names=output_names,
                        dynamo=True,
                        report=report,
                        **cfg
                        )
        onnx_program.optimize()
        if output_path is not None:
            onnx_program.save(output_path)
        self._onnx_program = onnx_program
        return self
    
    def _predict_onnx(self, x: torch.Tensor) -> torch.Tensor:
        assert self._onnx_program is not None, "ONNX program not created. Call to_onnx() first."
        if not self._onnx_initialized:
            self._onnx_program.initialize_inference_session()
            self._onnx_initialized = True
        return self._onnx_program(x)

    def predict(self, x: torch.Tensor) -> list[dict]:
        if self._onnx_program is not None:
            return self._predict_onnx(x)
        self.eval()
        with torch.no_grad():
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
                dropout=hparams['dropout'],
                freeze_backbone=hparams['freeze_backbone'],
                num_layers=hparams['num_layers'],
                hidden_dim=hparams['hidden_dim'],
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
