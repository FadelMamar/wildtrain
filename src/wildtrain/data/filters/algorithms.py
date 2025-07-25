# NOTE: Some type checkers may report false positives for pandas/numpy/sklearn operations. These are safe for runtime if the correct packages are installed.

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd  # Ensure pandas is installed
from sklearn.cluster import MiniBatchKMeans  # Use MiniBatchKMeans for scalability
from sklearn.metrics import silhouette_score

from wildtrain.utils.logging import get_logger
from .feature_extractor import FeatureExtractor
from .filter_config import ClusteringFilterConfig

logger = get_logger(__name__)


# --- Sampling Strategy Base and Implementation ---
class SamplingStrategy:
    def sample(self, cluster_df: pd.DataFrame, n_samples: int) -> List[int]:
        raise NotImplementedError

class BaseFilter(ABC):
    """
    Abstract base class for all filters.
    """

    @abstractmethod
    def filter(self, coco_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the filter to the input COCO data and return filtered data.
        """
        pass

    def get_filter_info(self) -> Dict[str, Any]:
        """
        Return information about the filter (for logging/debugging).
        """
        return {"filter_type": self.__class__.__name__}

    
class UniformDistanceRandomSamplingStrategy(SamplingStrategy):
    def sample(self, cluster_df: pd.DataFrame, n_samples: int) -> List[int]:
        # Sort by distance to centroid
        cluster_df = cluster_df.sort_values("dist_to_centroid")
        if n_samples >= len(cluster_df):
            return cluster_df.index.tolist()
        # Split into bins and randomly select one from each bin
        bins = np.array_split(cluster_df, n_samples)
        chosen = []
        for b in bins:
            if isinstance(b, pd.DataFrame) and not b.empty:
                chosen.append(
                    b.sample(1, random_state=random.randint(0, 1_000_000)).index[0]
                )
        return chosen



class ClusteringFilter(BaseFilter):
    """
    Clustering-based filter. Ensures embeddings are present in each image as image['_embedding'].
    If not present, computes embeddings using Dinov2Extractor and adds them in batches to avoid memory issues.
    Performs clustering on these embeddings and samples uniformly by distance to centroid, proportional to cluster size and x% reduction target.
    Sampling strategy is pluggable.
    """

    CLUSTER_TRIALS = [3, 5, 7, 9,]

    last_silhouette_scores: Optional[Dict[int, float]]
    last_samples_per_cluster: Optional[List[int]]

    def __init__(
        self,
        config: ClusteringFilterConfig,
        feature_extractor: Optional[FeatureExtractor] = None,
        batch_size: int = 64,
        sampling_strategy: Optional[SamplingStrategy] = None,
    ):
        self.config = config
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.batch_size = batch_size
        self.x_percent = getattr(config, "x_percent", 0.3)  # Default to 30% if not set
        self.sampling_strategy = (
            sampling_strategy or UniformDistanceRandomSamplingStrategy()
        )
        self.last_silhouette_scores = None
        self.last_samples_per_cluster = None

    def filter(self, coco_data: Dict[str, Any]) -> Dict[str, Any]:
        images = coco_data.get("images", [])
        if not images:
            logger.warning("No images found in coco_data")
            return coco_data
        coco_data = self.add_embeddings(coco_data)
        embeddings = np.stack([img["_embedding"] for img in coco_data["images"]])
        image_ids = [img["id"] for img in coco_data["images"]]
        file_names = [img["file_name"] for img in coco_data["images"]]
        # Find best clustering
        best_k, best_labels, silhouette_scores = self._find_best_kmeans(embeddings)
        self.last_silhouette_scores = silhouette_scores
        logger.info(f"Best number of clusters: {best_k}")
        logger.info(f"Silhouette scores for tried clusters: {silhouette_scores}")
        # Create DataFrame for sampling
        df = pd.DataFrame(
            {
                "image_id": image_ids,
                "file_name": file_names,
                "cluster": best_labels,
                "embedding": embeddings.tolist(),
            }
        )  # type: ignore
        # Compute centroids
        kmeans = MiniBatchKMeans(n_clusters=best_k, random_state=42).fit(embeddings)
        centroids = kmeans.cluster_centers_
        df["dist_to_centroid"] = [
            np.linalg.norm(emb - centroids[cl])
            for emb, cl in zip(embeddings, best_labels)
        ]  # type: ignore
        # Log distance distribution
        dist_series = pd.Series(df["dist_to_centroid"])
        logger.info(
            f"Distance to centroid stats: min={dist_series.min():.4f}, max={dist_series.max():.4f}, mean={dist_series.mean():.4f}, std={dist_series.std():.4f}"
        )
        # Determine number of samples to keep
        n_total = len(df)
        n_keep = int(np.ceil(self.x_percent * n_total))
        logger.info(
            f"Target: keeping {n_keep} out of {n_total} images ({self.x_percent*100:.1f}%)"
        )
        # Proportional allocation
        samples_per_cluster = self._allocate_samples_per_cluster(df, n_keep, best_k)
        self.last_samples_per_cluster = samples_per_cluster
        logger.info(f"Samples per cluster: {samples_per_cluster}")
        selected_indices = []
        for cl in range(best_k):
            cluster_df = df[df["cluster"] == cl].copy()
            n_samples = samples_per_cluster[cl]
            if n_samples == 0:
                continue
            chosen = self.sampling_strategy.sample(cluster_df, n_samples)
            selected_indices.extend(chosen)
        # Filter images and annotations
        selected_image_ids = set(df.loc[selected_indices, "image_id"])  # type: ignore
        filtered_images = [
            img for img in coco_data["images"] if img["id"] in selected_image_ids
        ]
        filtered_annotations = [
            ann
            for ann in coco_data.get("annotations", [])
            if ann["image_id"] in selected_image_ids
        ]
        # Clean up embeddings before export
        for img in filtered_images:
            if "_embedding" in img:
                del img["_embedding"]
        filtered_coco = dict(coco_data)
        filtered_coco["images"] = filtered_images
        filtered_coco["annotations"] = filtered_annotations
        return filtered_coco

    def _find_best_kmeans(
        self, embeddings: np.ndarray
    ) -> Tuple[int, np.ndarray, Dict[int, float]]:
        best_score = -1
        best_k = self.CLUSTER_TRIALS[0]
        best_labels = None
        silhouette_scores = {}
        for k in self.CLUSTER_TRIALS:
            if k >= len(embeddings):
                continue
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42).fit(embeddings)
            labels = kmeans.labels_
            if len(set(labels)) == 1:
                continue  # skip degenerate case
            score = silhouette_score(embeddings, labels)
            silhouette_scores[k] = score
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        return best_k, best_labels, silhouette_scores  # type: ignore

    def _allocate_samples_per_cluster(
        self, df: pd.DataFrame, n_keep: int, n_clusters: int
    ) -> List[int]:
        # Proportional allocation, at least 1 per cluster if possible
        cluster_sizes = df["cluster"].value_counts().sort_index().values
        total = cluster_sizes.sum()
        base = (cluster_sizes / total * n_keep).astype(int)  # type: ignore
        # Ensure at least 1 per cluster if possible
        base[base == 0] = 1  # type: ignore
        # Adjust to match n_keep
        while base.sum() > n_keep:
            idx = int(np.argmax(base))  # type: ignore
            base[idx] -= 1  # type: ignore
        while base.sum() < n_keep:
            idx = int(np.argmin(base))  # type: ignore
            base[idx] += 1  # type: ignore
        return base.tolist()  # type: ignore

    def add_embeddings(self, coco_data: Dict[str, Any]) -> Dict[str, Any]:
        images = coco_data.get("images", [])
        if not images:
            logger.warning("No images found in coco_data")
            return coco_data
        image_paths: List[Union[str, Path]] = [
            Path(img["file_name"]).resolve() for img in images
        ]
        embeddings = self._compute_embeddings_in_batches(image_paths)
        for img, emb in zip(images, embeddings):  # type: ignore
            img["_embedding"] = emb
        return coco_data

    def _compute_embeddings_in_batches(
        self, image_paths: List[Union[str, Path]]
    ) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            batch_embeddings = self.feature_extractor(batch_paths)  # type: ignore
            all_embeddings.append(batch_embeddings)
        return np.vstack(all_embeddings)

    def get_filter_info(self) -> Dict[str, Any]:
        return {"filter_type": self.__class__.__name__, **vars(self.config)}
