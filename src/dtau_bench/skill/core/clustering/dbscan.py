"""DBSCAN clustering for skill deduplication."""

import logging
from typing import List, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class DBSCANClustering:
    """
    DBSCAN clustering with cosine similarity.

    Used for grouping similar skills for merging.
    """

    def __init__(
        self,
        eps: float = 0.10,  # 1 - cosine_similarity_threshold (0.90)
        min_samples: int = 1
    ):
        """
        Initialize DBSCAN clustering.

        Args:
            eps: Maximum distance between samples (1 - similarity_threshold)
            min_samples: Minimum samples per cluster
        """
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, embeddings: np.ndarray) -> List[List[int]]:
        """
        Cluster embeddings using DBSCAN.

        Args:
            embeddings: Numpy array of embeddings (n_samples, n_features)

        Returns:
            List of clusters, each cluster is a list of indices
        """
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.metrics.pairwise import cosine_distances

            # Compute cosine distances
            distances = cosine_distances(embeddings)

            # Run DBSCAN
            clustering = DBSCAN(
                eps=self.eps,
                min_samples=self.min_samples,
                metric='precomputed'
            ).fit(distances)

            # Group by cluster
            clusters = {}
            for idx, label in enumerate(clustering.labels_):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(idx)

            # Convert to list (noise points get their own cluster)
            result = []
            for label, indices in clusters.items():
                if label == -1:
                    # Noise points - each as separate cluster
                    for idx in indices:
                        result.append([idx])
                else:
                    result.append(indices)

            logger.info(f"Created {len(result)} clusters from {len(embeddings)} items")
            return result

        except ImportError:
            logger.error("sklearn not installed, using simple grouping")
            return [[i] for i in range(len(embeddings))]

    def cluster_skills(
        self,
        skills: List[Dict],
        embeddings: np.ndarray
    ) -> List[List[Dict]]:
        """
        Cluster skills based on their embeddings.

        Args:
            skills: List of skill dictionaries
            embeddings: Corresponding embeddings

        Returns:
            List of skill clusters
        """
        cluster_indices = self.fit(embeddings)

        clusters = []
        for indices in cluster_indices:
            cluster = [skills[i] for i in indices]
            clusters.append(cluster)

        return clusters
