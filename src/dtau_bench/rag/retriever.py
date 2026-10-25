"""
Tool Retriever — Performs semantic search to find relevant tools.

This module provides functionality to:
1. Encode user queries into embeddings
2. Compute cosine similarity between queries and indexed tools
3. Return the most relevant tools for a given query

Usage:
    retriever = ToolRetriever(index, config)
    results = retriever.retrieve("I need to check my bank account balance")
    for result in results:
        print(f"{result.function_name}: {result.score:.3f}")
"""

import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request

import numpy as np

from dtau_bench.utils.embedding_http import (
    HTTPEmbeddingModelResolver,
    build_openai_api_url,
)

from .config import RAGConfig
from .indexer import ToolDocument

logger = logging.getLogger(__name__)

# Thread-safe model cache for multi-threaded environments
_model_lock = threading.Lock()
_model_cache: Dict[str, Any] = {}


@dataclass
class RetrievalResult:
    """Result of a single tool retrieval.

    Attributes:
        tool_name: Full tool name (same as function_name in dtau_bench).
        function_name: Function name.
        domain: Domain/server this tool belongs to.
        score: Similarity score (cosine similarity, higher is better).
        description: Tool description.
    """

    tool_name: str
    function_name: str
    domain: str
    score: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "function_name": self.function_name,
            "domain": self.domain,
            "score": round(self.score, 4),
            "description": self.description,
        }


class ToolRetriever:
    """Retrieves relevant tools based on semantic similarity.

    Uses pre-built embeddings and cosine similarity for fast retrieval.
    Thread-safe for concurrent rollout workers.

    Args:
        index: Pre-built index containing documents and embeddings.
        config: RAG configuration.
    """

    def __init__(
        self,
        index: Dict[str, Any],
        config: RAGConfig,
    ):
        self.documents: List[ToolDocument] = index["documents"]
        self.embeddings: np.ndarray = np.asarray(index["embeddings"], dtype=np.float32)
        self.config = config
        self._embedding_model = None
        self._http_model_resolver = (
            HTTPEmbeddingModelResolver(
                base_url=config.embedding_api_base,
                configured_model=config.embedding_model,
                api_key=config.embedding_api_key,
            )
            if config.embedding_backend == "openai_http"
            and config.embedding_api_base
            else None
        )

        # Pre-normalize embeddings for fast cosine similarity
        self._normalized_embeddings = self._normalize_embeddings(self.embeddings)

        logger.info(f"ToolRetriever initialized with {len(self.documents)} tools")

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit vectors for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms

    def _get_embedding_model(self):
        """Lazy initialization of local embedding model with thread-safe caching."""
        if self.config.embedding_backend != "local":
            raise RuntimeError(
                "_get_embedding_model is only available for local embedding backend"
            )
        if self._embedding_model is None:
            cache_key = (
                self.config.embedding_model,
                self.config.use_flash_attention,
            )

            with _model_lock:
                if cache_key in _model_cache:
                    self._embedding_model = _model_cache[cache_key]
                    logger.debug(
                        f"Reusing cached embedding model: {self.config.embedding_model}"
                    )
                else:
                    try:
                        from sentence_transformers import SentenceTransformer

                        model_kwargs = {}
                        tokenizer_kwargs = {}

                        if self.config.use_flash_attention:
                            model_kwargs["attn_implementation"] = "flash_attention_2"
                            model_kwargs["device_map"] = "auto"
                            tokenizer_kwargs["padding_side"] = "left"

                        self._embedding_model = SentenceTransformer(
                            self.config.embedding_model,
                            model_kwargs=model_kwargs if model_kwargs else None,
                            tokenizer_kwargs=(
                                tokenizer_kwargs if tokenizer_kwargs else None
                            ),
                        )
                        logger.info(
                            f"Loaded embedding model: {self.config.embedding_model}"
                        )
                    except ImportError:
                        raise ImportError(
                            "sentence-transformers is required for local embeddings. "
                            "From the code repository, install with: "
                            "pip install '.[local-embeddings]'"
                        )

                    _model_cache[cache_key] = self._embedding_model
        return self._embedding_model

    def _format_query_text(self, text: str) -> str:
        """Format query text with instruction prefix."""
        instruction = (self.config.query_instruction or "").strip()
        if not instruction:
            return text
        return f"Instruct: {instruction}\nQuery: {text}"

    def _embed_query_via_http(self, query: str) -> np.ndarray:
        """Embed a query via OpenAI-compatible HTTP API."""
        if self._http_model_resolver is None:
            raise RuntimeError(
                "embedding_api_base is required for the openai_http backend"
            )
        payload = {
            "model": self._http_model_resolver.resolve(),
            "input": [self._format_query_text(query)],
        }
        headers = {"Content-Type": "application/json"}
        if self.config.embedding_api_key:
            headers["Authorization"] = f"Bearer {self.config.embedding_api_key}"

        url = build_openai_api_url(
            self.config.embedding_api_base,
            "embeddings",
        )

        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"Embedding API request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Failed to reach embedding API at "
                f"{self.config.embedding_api_base}: {exc}"
            ) from exc

        items = data.get("data")
        if not isinstance(items, list) or len(items) != 1:
            raise RuntimeError(
                "Embedding API query response must contain exactly one embedding"
            )
        embedding = items[0].get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError(
                "Embedding API query response missing embedding vector"
            )
        return np.array(embedding, dtype=np.float32)

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a query text."""
        if self.config.embedding_backend == "openai_http":
            return self._embed_query_via_http(query)

        # Local backend
        model = self._get_embedding_model()
        if self.config.query_prompt_name:
            embedding = model.encode(
                query,
                prompt_name=self.config.query_prompt_name,
                convert_to_numpy=True,
            )
        else:
            embedding = model.encode(query, convert_to_numpy=True)
        return embedding

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        domains: Optional[List[str]] = None,
        min_score: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """Retrieve top-k most relevant tools for a query.

        Args:
            query: User's natural language query.
            top_k: Number of tools to retrieve (default: from config).
            domains: Optional filter to specific domains.
            min_score: Minimum similarity score (default: from config).

        Returns:
            List of RetrievalResult objects, sorted by score (descending).
        """
        top_k = top_k or self.config.top_k
        min_score = (
            min_score if min_score is not None else self.config.similarity_threshold
        )

        # Get query embedding and normalize
        query_embedding = self._get_query_embedding(query)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_normalized = query_embedding / query_norm
        else:
            query_normalized = query_embedding

        # Compute cosine similarities (dot product of normalized vectors)
        similarities = np.dot(self._normalized_embeddings, query_normalized)

        # Filter by domain if specified
        if domains:
            domain_set = set(domains)
            domain_mask = np.array(
                [doc.domain in domain_set for doc in self.documents]
            )
            similarities = np.where(domain_mask, similarities, -np.inf)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < min_score:
                continue

            doc = self.documents[idx]
            results.append(
                RetrievalResult(
                    tool_name=doc.tool_name,
                    function_name=doc.function_name,
                    domain=doc.domain,
                    score=score,
                    description=doc.description,
                )
            )

        return results

    def retrieve_tool_names(
        self,
        query: str,
        top_k: Optional[int] = None,
        domains: Optional[List[str]] = None,
        include_scores: bool = False,
    ) -> List[str] | Tuple[List[str], Dict[str, float]]:
        """Retrieve tool names for a query.

        Args:
            query: User query.
            top_k: Number of tools to retrieve.
            domains: Optional domain filter.
            include_scores: If True, also return similarity scores.

        Returns:
            List of function names, or tuple of (names, scores) if include_scores.
        """
        results = self.retrieve(query, top_k, domains)
        names = [r.function_name for r in results]

        if include_scores:
            scores = {r.function_name: r.score for r in results}
            return names, scores

        return names

    def list_all_tools(self) -> List[str]:
        """List all tool names in the index."""
        return [doc.function_name for doc in self.documents]

    def list_domains(self) -> List[str]:
        """List all domains in the index."""
        return sorted(set(doc.domain for doc in self.documents))

    @property
    def total_tools(self) -> int:
        """Total number of tools in the index."""
        return len(self.documents)
