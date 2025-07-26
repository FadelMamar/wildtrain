from PIL import Image
from torchvision import transforms as T
import torch
from omegaconf import DictConfig, OmegaConf
import os
from typing import Union

def load_image(path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return T.PILToTensor()(image)