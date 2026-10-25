"""
Tool Indexer — Builds and manages embeddings for all tools.

This module provides functionality to:
1. Load tool definitions from data/tools/{domain}/{version}_tools.json
2. Create semantic representations (name + description + parameters)
3. Generate and cache embeddings using embedding models
4. Load pre-built indices for fast retrieval

Usage:
    # Build new index
    indexer = ToolIndexer(config, tools_dir=Path("data/tools"), version="v1")
    index = indexer.build_index()
    indexer.save_index(index, "./rag_index_cache/v1")

    # Load pre-built index (recommended for rollout)
    indexer = ToolIndexer(config)
    index = indexer.load_index("./rag_index_cache/v1")
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error, request

import numpy as np

from dtau_bench.utils.embedding_http import (
    HTTPEmbeddingModelResolver,
    build_openai_api_url,
)

from .config import RAGConfig

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_BATCH_SIZE = 64


@dataclass
class ToolDocument:
    """Represents a tool as a searchable document.

    Attributes:
        tool_name: Full tool name (function_name, without domain prefix).
        function_name: The function name (same as tool_name).
        domain: The domain/server this tool belongs to.
        description: Tool description.
        parameters_text: Flattened parameter descriptions.
        full_text: Combined text for embedding (what gets embedded).
        embedding: The embedding vector (None until embedded).
    """

    tool_name: str
    function_name: str
    domain: str
    description: str
    parameters_text: str
    full_text: str
    embedding: Optional[List[float]] = None

    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "tool_name": self.tool_name,
            "function_name": self.function_name,
            "domain": self.domain,
            "description": self.description,
            "parameters_text": self.parameters_text,
            "full_text": self.full_text,
        }
        if include_embedding and self.embedding is not None:
            result["embedding"] = self.embedding
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolDocument":
        """Create from dictionary."""
        return cls(
            tool_name=data["tool_name"],
            function_name=data["function_name"],
            domain=data["domain"],
            description=data["description"],
            parameters_text=data["parameters_text"],
            full_text=data["full_text"],
            embedding=data.get("embedding"),
        )


class ToolIndexer:
    """Builds and manages the tool embedding index.

    Reads ``data/tools/{domain}/{version}_tools.json``.

    Args:
        config: RAG configuration.
        tools_dir: Path to data/tools/.
        version: Catalog version (e.g., "v1").
        embedding_batch_size: Maximum texts per embedding request. ``None``
            uses the historical default of 64.
    """

    def __init__(
        self,
        config: RAGConfig,
        tools_dir: Optional[Path] = None,
        version: str = "v1",
        embedding_batch_size: Optional[int] = None,
    ):
        if embedding_batch_size is not None and (
            isinstance(embedding_batch_size, bool)
            or not isinstance(embedding_batch_size, int)
            or embedding_batch_size <= 0
        ):
            raise ValueError(
                "embedding_batch_size must be a positive integer, "
                f"got {embedding_batch_size!r}"
            )
        self.config = config
        self.tools_dir = tools_dir
        self.version = version
        # ``None`` preserves compatibility for runtime callers which do not
        # control index construction. Build CLIs always pass an explicit value,
        # so their validation detects a models.yaml/CLI batch-size change.
        self._embedding_batch_size_explicit = embedding_batch_size is not None
        self.embedding_batch_size = (
            embedding_batch_size
            if embedding_batch_size is not None
            else DEFAULT_EMBEDDING_BATCH_SIZE
        )
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

    # ─────────────────────────────────────────────────────────────────────────
    # Document Loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_documents_from_tools_dir(self) -> List[ToolDocument]:
        """Load tools from data/tools/{domain}/{version}_tools.json layout."""
        if self.tools_dir is None:
            raise ValueError("tools_dir is required for dtau_bench standard layout")

        documents: List[ToolDocument] = []
        domain_dirs = sorted(
            [p for p in self.tools_dir.iterdir() if p.is_dir()]
        )

        for domain_dir in domain_dirs:
            tools_file = domain_dir / f"{self.version}_tools.json"
            if not tools_file.exists():
                logger.warning(
                    f"Skipping {domain_dir.name}: "
                    f"tools file not found at {tools_file}"
                )
                continue

            logger.info(
                f"Loading tools from {domain_dir.name}/{tools_file.name}"
            )
            with open(tools_file, encoding="utf-8") as f:
                tools = json.load(f)

            for tool_def in tools:
                try:
                    doc = self._create_tool_document(tool_def, domain_dir.name)
                    documents.append(doc)
                except Exception as e:
                    logger.warning(
                        f"Failed to process tool in {domain_dir.name}: {e}"
                    )
                    continue

        return documents


    # ─────────────────────────────────────────────────────────────────────────
    # Document Creation
    # ─────────────────────────────────────────────────────────────────────────

    def _create_tool_document(
        self,
        tool_def: Dict[str, Any],
        domain: str,
    ) -> ToolDocument:
        """Create a searchable document from a tool definition.

        Combines:
        - Function name
        - Description
        - Parameter names and descriptions
        """
        # Handle various tool definition formats
        if "tool" in tool_def:
            tool_def = tool_def["tool"]
        if "function" in tool_def:
            func = tool_def["function"]
        else:
            func = tool_def

        name = func.get("name", "")
        description = func.get("description", "")

        # Build parameter text
        params = func.get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])

        param_parts = []
        for param_name, param_info in properties.items():
            param_desc = param_info.get("description", "")
            param_type = param_info.get("type", "any")
            is_required = "required" if param_name in required else "optional"
            param_parts.append(
                f"- {param_name} ({param_type}, {is_required}): {param_desc}"
            )

        parameters_text = "\n".join(param_parts) if param_parts else "No parameters"

        # Full text for embedding - optimized for semantic search
        full_text = (
            f"Tool: {name}\n"
            f"Domain: {domain}\n"
            f"Description: {description}\n"
            f"Parameters:\n{parameters_text}"
        )

        return ToolDocument(
            tool_name=name,
            function_name=name,
            domain=domain,
            description=description,
            parameters_text=parameters_text,
            full_text=full_text,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Embedding Generation
    # ─────────────────────────────────────────────────────────────────────────

    def _get_embedding_model(self):
        """Lazy initialization of local embedding model."""
        if self.config.embedding_backend != "local":
            raise RuntimeError(
                "_get_embedding_model is only available for local embedding backend"
            )
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                model_kwargs = {}
                tokenizer_kwargs = {}

                if self.config.use_flash_attention:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    tokenizer_kwargs["padding_side"] = "left"

                self._embedding_model = SentenceTransformer(
                    self.config.embedding_model,
                    model_kwargs=model_kwargs if model_kwargs else None,
                    tokenizer_kwargs=tokenizer_kwargs if tokenizer_kwargs else None,
                )
                logger.info(f"Loaded embedding model: {self.config.embedding_model}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "From the code repository, install with: "
                    "pip install '.[local-embeddings]'"
                )
        return self._embedding_model

    def _format_query_text(self, text: str) -> str:
        """Format text with instruction prefix (for openai_http backend)."""
        instruction = (self.config.query_instruction or "").strip()
        if not instruction:
            return text
        return f"Instruct: {instruction}\nQuery: {text}"

    def _embed_via_http(
        self,
        texts: List[str],
        *,
        offline_build: bool = True,
    ) -> List[List[float]]:
        """Generate embeddings via OpenAI-compatible HTTP API."""
        if self._http_model_resolver is None:
            raise RuntimeError(
                "embedding_api_base is required for the openai_http backend"
            )
        payload = {
            "model": self._http_model_resolver.resolve(),
            "input": texts,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.embedding_api_key:
            headers["Authorization"] = f"Bearer {self.config.embedding_api_key}"

        url = build_openai_api_url(
            self.config.embedding_api_base,
            "embeddings",
        )

        req = request.Request(
            url,
            data=body,
            headers=headers,
            method="POST",
        )
        # Long CPU batches occur only while building an index. Runtime queries
        # contain one short text and must retain a bounded latency instead of
        # inheriting the offline builder's long timeout and retry budget.
        timeout = (
            self.config.embedding_http_timeout
            if offline_build
            else min(self.config.embedding_http_timeout, 120.0)
        )
        max_attempts = (
            self.config.embedding_http_max_retries + 1
            if offline_build
            else 1
        )
        raw = ""
        for attempt in range(1, max_attempts + 1):
            try:
                with request.urlopen(
                    req,
                    timeout=timeout,
                ) as resp:
                    raw = resp.read().decode("utf-8")
                break
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                retryable = exc.code in {408, 429} or 500 <= exc.code < 600
                message = (
                    f"Embedding API request failed with HTTP {exc.code}: {detail}"
                )
                if not retryable or attempt >= max_attempts:
                    raise RuntimeError(message) from exc
                cause: BaseException = exc
            except (TimeoutError, error.URLError) as exc:
                message = (
                    "Embedding API request timed out"
                    if isinstance(exc, TimeoutError)
                    else "Failed to reach embedding API"
                )
                message = (
                    f"{message} at {self.config.embedding_api_base} "
                    f"after {timeout:g}s"
                )
                if attempt >= max_attempts:
                    raise RuntimeError(message) from exc
                cause = exc

            delay = self.config.embedding_http_retry_delay * attempt
            logger.warning(
                "%s; retrying embedding request (%d/%d) in %.1fs: %s",
                message,
                attempt + 1,
                max_attempts,
                delay,
                cause,
            )
            if delay:
                time.sleep(delay)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Embedding API returned invalid JSON: {raw[:500]}"
            ) from exc

        items = data.get("data")
        if not isinstance(items, list):
            raise RuntimeError("Embedding API response missing 'data' list")
        if len(items) != len(texts):
            raise RuntimeError(
                f"Embedding API returned {len(items)} embeddings for {len(texts)} inputs"
            )

        embeddings: List[List[float]] = []
        for idx, item in enumerate(items):
            embedding = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(embedding, list) or not embedding:
                raise RuntimeError(
                    f"Embedding API item {idx} missing embedding vector"
                )
            embeddings.append(embedding)

        return embeddings

    def _get_embeddings_batch(
        self, texts: List[str], is_query: bool = False
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            is_query: If True, use query prompt/instruction.
        """
        if self.config.embedding_backend == "openai_http":
            # For openai_http, only format queries with instruction
            formatted = (
                [self._format_query_text(t) for t in texts] if is_query else texts
            )
            return self._embed_via_http(formatted, offline_build=not is_query)

        # Local backend
        model = self._get_embedding_model()
        if is_query and self.config.query_prompt_name:
            embeddings = model.encode(
                texts,
                prompt_name=self.config.query_prompt_name,
                convert_to_numpy=True,
            )
        else:
            embeddings = model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]

    # ─────────────────────────────────────────────────────────────────────────
    # Build / Save / Load Index
    # ─────────────────────────────────────────────────────────────────────────

    def build_index(self) -> Dict[str, Any]:
        """Build the complete tool index.

        Scans tools from the configured source and generates embeddings.

        Returns:
            Dictionary containing:
            - documents: List of ToolDocument objects with embeddings
            - embeddings: numpy array of shape (N, dim)
            - metadata: Index metadata (model, timestamp, etc.)
        """
        if self.tools_dir is None:
            raise ValueError("tools_dir is required for building the index")
        documents = self._load_documents_from_tools_dir()

        if not documents:
            raise ValueError(
                f"No tools found. Check the source directory and version={self.version}"
            )

        logger.info(f"Found {len(documents)} tools, generating embeddings...")

        # Generate embeddings in batches
        all_embeddings: List[List[float]] = []

        for i in range(0, len(documents), self.embedding_batch_size):
            batch = documents[i : i + self.embedding_batch_size]
            texts = [d.full_text for d in batch]
            embeddings = self._get_embeddings_batch(texts, is_query=False)
            all_embeddings.extend(embeddings)

            processed = min(i + self.embedding_batch_size, len(documents))
            if processed % 100 == 0 or processed >= len(documents):
                logger.info(f"Embedded {processed}/{len(documents)} tools")

        # Assign embeddings to documents
        for doc, embedding in zip(documents, all_embeddings):
            doc.embedding = embedding

        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        # Build metadata
        metadata = {
            "embedding_model": self.config.embedding_model,
            "embedding_backend": self.config.embedding_backend,
            "embedding_api_base": (
                self.config.embedding_api_base
                if self.config.embedding_backend == "openai_http"
                else None
            ),
            "query_instruction": self.config.query_instruction,
            "query_format_version": self.config.query_format_version,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_dim": embeddings_array.shape[1] if embeddings_array.ndim > 1 else 0,
            "total_tools": len(documents),
            "domains": sorted(set(d.domain for d in documents)),
            "source_version": self.version,
            "created_at": datetime.now().isoformat(),
        }

        logger.info(
            f"Index built: {len(documents)} tools from "
            f"{len(metadata['domains'])} domains"
        )

        return {
            "documents": documents,
            "embeddings": embeddings_array,
            "metadata": metadata,
        }

    def save_index(self, index: Dict[str, Any], path: str) -> None:
        """Save index to disk.

        Creates three files:
        - embeddings.npy: numpy array of embeddings
        - documents.json: tool document info (without embeddings)
        - metadata.json: index metadata
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save embeddings as numpy array
        embeddings_path = save_path / "embeddings.npy"
        np.save(embeddings_path, index["embeddings"])
        logger.info(f"Saved embeddings to {embeddings_path}")

        # Save documents (without embeddings to save space)
        docs_path = save_path / "documents.json"
        docs_data = [d.to_dict(include_embedding=False) for d in index["documents"]]
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(docs_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(docs_data)} documents to {docs_path}")

        # Save metadata
        metadata_path = save_path / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(index["metadata"], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata to {metadata_path}")

    def load_index(self, path: str) -> Dict[str, Any]:
        """Load pre-built index from disk.

        Args:
            path: Directory containing the index files.

        Returns:
            Dictionary with documents, embeddings, and metadata.
        """
        load_path = Path(path)

        if not load_path.exists():
            raise FileNotFoundError(f"Index not found at {path}")

        # Load embeddings
        embeddings_path = load_path / "embeddings.npy"
        if not embeddings_path.exists():
            raise FileNotFoundError(f"embeddings.npy not found in {path}")
        embeddings = np.load(embeddings_path)

        # Load documents
        docs_path = load_path / "documents.json"
        if not docs_path.exists():
            raise FileNotFoundError(f"documents.json not found in {path}")
        with open(docs_path, encoding="utf-8") as f:
            docs_data = json.load(f)

        # Reconstruct documents with embeddings
        documents = []
        for i, d in enumerate(docs_data):
            doc = ToolDocument.from_dict(d)
            doc.embedding = embeddings[i].tolist()
            documents.append(doc)

        # Load metadata
        metadata_path = load_path / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)

        logger.info(
            f"Loaded index: {len(documents)} tools from "
            f"{len(metadata.get('domains', []))} domains"
        )

        return {
            "documents": documents,
            "embeddings": embeddings,
            "metadata": metadata,
        }

    def validate_index(self, index: Dict[str, Any]) -> bool:
        """Validate that an index is compatible with current config.

        Checks embedding model, backend, instruction, and dimension.
        Returns True if compatible, False (with warnings) otherwise.
        """
        metadata = index.get("metadata", {})

        # Check embedding model matches
        if metadata.get("embedding_model") != self.config.embedding_model:
            logger.warning(
                f"Index was built with model '{metadata.get('embedding_model')}', "
                f"but config specifies '{self.config.embedding_model}'"
            )
            return False

        if metadata.get("embedding_backend", "openai_http") != self.config.embedding_backend:
            logger.warning(
                f"Index was built with backend '{metadata.get('embedding_backend')}', "
                f"but config specifies '{self.config.embedding_backend}'"
            )
            return False

        if metadata.get("query_instruction", "") != (self.config.query_instruction or ""):
            logger.warning("Index query_instruction does not match current config")
            return False

        if (
            metadata.get("query_format_version", "instruct_query_v1")
            != self.config.query_format_version
        ):
            logger.warning("Index query_format_version does not match current config")
            return False

        # Indices created before this field was added used the historical
        # default of 64. Runtime callers without an explicit construction batch
        # accept any persisted value because batching does not affect queries;
        # build/validate CLIs pass a value and therefore detect config changes.
        index_batch_size = metadata.get(
            "embedding_batch_size", DEFAULT_EMBEDDING_BATCH_SIZE
        )
        if (
            self._embedding_batch_size_explicit
            and index_batch_size != self.embedding_batch_size
        ):
            logger.warning(
                f"Index was built with embedding batch size {index_batch_size!r}, "
                f"but config specifies {self.embedding_batch_size}"
            )
            return False

        # Check embedding dimensions
        embeddings = index.get("embeddings")
        if embeddings is not None and hasattr(embeddings, "shape"):
            expected_dim = self.config.get_embedding_dim()
            actual_dim = (
                embeddings.shape[1] if len(embeddings.shape) > 1 else 0
            )
            if actual_dim and actual_dim != expected_dim:
                logger.warning(
                    f"Index has embedding dimension {actual_dim}, "
                    f"but model expects {expected_dim}"
                )
                return False

        return True
