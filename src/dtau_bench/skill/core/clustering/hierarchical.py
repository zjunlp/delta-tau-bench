"""Hierarchical clustering for skill deduplication."""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class CompleteLinkageClustering:
    """
    Agglomerative clustering with complete linkage and cosine distance.

    Complete linkage better matches the desired behavior for skill merging:
    all items inside a cluster should remain mutually close, rather than being
    chained together through bridge points.
    """

    def __init__(
        self,
        distance_threshold: float = 0.10,  # 1 - cosine_similarity_threshold (0.90)
    ):
        self.distance_threshold = distance_threshold

    def fit(self, embeddings: np.ndarray) -> List[List[int]]:
        if len(embeddings) == 0:
            return []

        try:
            from sklearn.cluster import AgglomerativeClustering

            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="complete",
                distance_threshold=self.distance_threshold,
            ).fit(embeddings)

            clusters: Dict[int, List[int]] = {}
            for idx, label in enumerate(clustering.labels_):
                clusters.setdefault(int(label), []).append(idx)

            result = list(clusters.values())
            logger.info(
                "Created %d complete-linkage clusters from %d items",
                len(result),
                len(embeddings),
            )
            return result

        except ImportError:
            logger.error("sklearn not installed, using simple grouping")
            return [[i] for i in range(len(embeddings))]

    def cluster_skills(
        self,
        skills: List[Dict],
        embeddings: np.ndarray,
    ) -> List[List[Dict]]:
        cluster_indices = self.fit(embeddings)
        return [[skills[i] for i in indices] for indices in cluster_indices]
