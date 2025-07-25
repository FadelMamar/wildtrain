# Hard sample mining strategies and filters
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchvision.ops as tvops
from PIL import Image



# --- Miner Abstract Base ---
class Miner(ABC):
    def __init__(self):
        self.reset()

    @abstractmethod
    def ingest_batch(
        self, images: List[Dict[str, Any]], predictions: List[Dict[str, Any]]
    ):
        pass

    @abstractmethod
    def finalize(self) -> Dict[Any, float]:
        pass

    def reset(self):
        pass


