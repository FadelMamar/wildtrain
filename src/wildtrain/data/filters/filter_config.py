"""
FilterConfig Pydantic models and loader for filter and feature extractor settings.
"""

from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel


class FeatureExtractorConfig(BaseModel):
    model_name: str = "timm/vit_small_patch16_224.dino"
    device: str = "cpu"


class ClusteringFilterConfig(BaseModel):
    enabled: bool = False
    n_clusters: int = 7
    samples_per_cluster: int = 5
    method: str = "kmeans"  # or "agglomerative", etc.
    x_percent: float = 0.3  # Fraction of data to keep after filtering


#TODO
class HardSampleMiningConfig(BaseModel):
    enabled: bool = False
   


class FilterConfig(BaseModel):
    feature_extractor: FeatureExtractorConfig = FeatureExtractorConfig()
    clustering: ClusteringFilterConfig = ClusteringFilterConfig()
    # hard_sample_mining: HardSampleMiningConfig = HardSampleMiningConfig()

    @classmethod
    def from_yaml(cls, path: str) -> "FilterConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "FilterConfig":
        return cls.model_validate(data)
