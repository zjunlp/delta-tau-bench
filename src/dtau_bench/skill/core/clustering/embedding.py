"""Embedding service for skill clustering."""

import asyncio
import json
import logging
from typing import List, Optional
from urllib import request, error

import numpy as np

from dtau_bench.utils.embedding_http import (
    HTTPEmbeddingModelResolver,
    build_openai_api_url,
)

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Embedding service for skill text.

    Supports vLLM and other embedding backends.
    """

    def __init__(
        self,
        model: str = "Qwen3-Embedding-8B",
        base_url: Optional[str] = None,
        batch_size: int = 32,
        api_key: Optional[str] = None,
    ):
        """
        Initialize embedding service.

        Args:
            model: Embedding model name
            base_url: vLLM server URL
            batch_size: Batch size for embedding
        """
        self.model = model
        self.base_url = base_url
        self.batch_size = batch_size
        self.api_key = api_key or ""
        self.endpoint = self._build_endpoint(base_url)
        self._model_resolver = HTTPEmbeddingModelResolver(
            base_url=base_url,
            configured_model=model,
            api_key=self.api_key,
        )

    def _build_endpoint(self, base_url: Optional[str]) -> str:
        if not base_url:
            raise ValueError("EmbeddingService requires a base_url")
        return build_openai_api_url(base_url, "embeddings")

    async def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.zeros((0, 0), dtype=float)
        return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: List[str]) -> np.ndarray:
        payload = {
            "model": self._model_resolver.resolve(),
            "input": texts,
        }
        raw = self._request_embeddings(payload)
        data = raw.get("data")
        if not isinstance(data, list):
            raise ValueError("Embedding response missing data list")

        embeddings = []
        for item in data:
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise ValueError("Embedding response missing embedding vectors")
            embeddings.append(item["embedding"])

        if len(embeddings) != len(texts):
            raise ValueError(
                f"Embedding response count mismatch: expected {len(texts)}, got {len(embeddings)}"
            )

        array = np.asarray(embeddings, dtype=float)
        if array.ndim != 2:
            raise ValueError(f"Embedding array must be 2D, got shape {array.shape}")
        return array

    def _request_embeddings(self, payload: dict) -> dict:
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req) as resp:
                if resp.status != 200:
                    body = resp.read().decode("utf-8", errors="replace")
                    raise ValueError(
                        f"Embedding request failed with status {resp.status}: {body}"
                    )
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ValueError(
                f"Embedding request failed with status {exc.code}: {body}"
            ) from exc

    async def embed_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for large batches.

        Args:
            texts: List of text strings
            show_progress: Whether to show progress

        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.zeros((0, 0), dtype=float)

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = await self.embed(batch)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)
