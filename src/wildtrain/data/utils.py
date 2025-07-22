from PIL import Image
from torchvision import transforms as T
import torch


def load_image(path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return T.PILToTensor()(image)