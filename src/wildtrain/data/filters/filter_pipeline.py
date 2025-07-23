"""
Filter pipeline for orchestrating multiple data filtering steps on COCO data.
"""


from typing import Any, Dict, List, Optional

from .algorithms import ClusteringFilter, BaseFilter
from .feature_extractor import FeatureExtractor
from .filter_config import FilterConfig



class FilterPipeline:
    """
    Pipeline for orchestrating multiple data filtering steps on COCO data.
    """

    def __init__(self, filters: Optional[List[BaseFilter]] = None):
        self.filters = filters or []
        self.filter_history: List[Dict[str, Any]] = []

    def add_filter(self, filter_obj: BaseFilter) -> None:
        self.filters.append(filter_obj)

    def clear_filters(self) -> None:
        self.filters.clear()
        self.filter_history.clear()

    def filter(self, coco_data: Dict[str, Any]) -> Dict[str, Any]:
        data = coco_data
        self.filter_history.clear()
        for i, filter_obj in enumerate(self.filters):
            data = filter_obj.filter(data)
            self.filter_history.append(filter_obj.get_filter_info())
        return data

    def get_filter_history(self) -> List[Dict[str, Any]]:
        return self.filter_history

    @classmethod
    def from_config(cls, config: FilterConfig) -> "FilterPipeline":
        filters: List[BaseFilter] = []

        # Clustering filter
        if getattr(config.clustering, "enabled", False):
            filters.append(
                ClusteringFilter(
                    config.clustering,
                    feature_extractor=FeatureExtractor(
                        model_name=config.feature_extractor.model_name,
                        device=config.feature_extractor.device,
                    ),
                )
            )

        return cls(filters)
