"""
Configuration for RAG tool retrieval.

Supports either local sentence-transformers inference or an OpenAI-compatible
HTTP embeddings API (e.g., vLLM, TEI, or similar).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RAGConfig:
    """Configuration for RAG tool retrieval.

    Attributes:
        embedding_model: Name or path of the embedding model.
            E.g., "Qwen3-Embedding-0.6B" or local path.
        embedding_backend: Either "local" (sentence-transformers) or "openai_http".
        embedding_api_base: Base URL for OpenAI-compatible embedding API.
        embedding_api_key: Optional API key for embedding API.
        query_instruction: Optional instruction prefix for query encoding.
        top_k: Default number of tools to retrieve.
        similarity_threshold: Minimum similarity score to include a tool.
        index_cache_dir: Directory to store/load pre-built indices.
        query_prompt_name: Prompt name for local query encoding (sentence-transformers).
        use_flash_attention: Whether to use flash_attention_2 (local backend only).
        query_format_version: Version of the query formatting logic.
        embedding_http_timeout: Per-request timeout for HTTP embedding calls.
        embedding_http_max_retries: Retries after a retryable HTTP embedding failure.
        embedding_http_retry_delay: Base delay between HTTP embedding retries.
    """

    embedding_model: str = "Qwen3-Embedding-0.6B"
    embedding_backend: str = "openai_http"
    embedding_api_base: Optional[str] = None
    embedding_api_key: Optional[str] = None
    query_instruction: str = "Given a user request, retrieve relevant tools that can help complete the task"
    top_k: int = 15
    similarity_threshold: float = 0.0
    index_cache_dir: str = "./rag_index_cache"
    query_prompt_name: str = "query"  # Only used by local backend
    use_flash_attention: bool = False
    query_format_version: str = "instruct_query_v1"
    embedding_http_timeout: float = 600.0
    embedding_http_max_retries: int = 2
    embedding_http_retry_delay: float = 1.0

    # Known embedding models and their dimensions
    KNOWN_MODELS: Dict[str, int] = None  # type: ignore

    def __post_init__(self):
        if (
            isinstance(self.embedding_http_timeout, bool)
            or not isinstance(self.embedding_http_timeout, (int, float))
            or self.embedding_http_timeout <= 0
        ):
            raise ValueError(
                "embedding_http_timeout must be a positive number, "
                f"got {self.embedding_http_timeout!r}"
            )
        if (
            isinstance(self.embedding_http_max_retries, bool)
            or not isinstance(self.embedding_http_max_retries, int)
            or self.embedding_http_max_retries < 0
        ):
            raise ValueError(
                "embedding_http_max_retries must be a non-negative integer, "
                f"got {self.embedding_http_max_retries!r}"
            )
        if (
            isinstance(self.embedding_http_retry_delay, bool)
            or not isinstance(self.embedding_http_retry_delay, (int, float))
            or self.embedding_http_retry_delay < 0
        ):
            raise ValueError(
                "embedding_http_retry_delay must be a non-negative number, "
                f"got {self.embedding_http_retry_delay!r}"
            )
        if self.KNOWN_MODELS is None:
            object.__setattr__(self, "KNOWN_MODELS", {
                "Qwen3-Embedding-0.6B": 1024,
                "Qwen/Qwen3-Embedding-0.6B": 1024,
                "Qwen3-Embedding-4B": 2560,
                "Qwen/Qwen3-Embedding-4B": 2560,
            })

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension for the current model."""
        if self.KNOWN_MODELS and self.embedding_model in self.KNOWN_MODELS:
            return self.KNOWN_MODELS[self.embedding_model]
        # Check if model name is a suffix match (e.g., path/to/Qwen3-Embedding-0.6B)
        for model_name, dim in (self.KNOWN_MODELS or {}).items():
            if self.embedding_model.endswith(model_name):
                return dim
        return 1024  # Default for Qwen 0.6B

    def validate(self) -> bool:
        """Validate the configuration. Raises ValueError on failure."""
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")

        if self.embedding_backend not in {"local", "openai_http"}:
            raise ValueError(
                f"embedding_backend must be one of {{'local', 'openai_http'}}, "
                f"got {self.embedding_backend}"
            )

        if self.embedding_backend == "openai_http" and not self.embedding_api_base:
            raise ValueError(
                "embedding_api_base is required when embedding_backend='openai_http'"
            )

        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError(
                f"similarity_threshold must be between 0 and 1, "
                f"got {self.similarity_threshold}"
            )

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "embedding_model": self.embedding_model,
            "embedding_backend": self.embedding_backend,
            "embedding_api_base": self.embedding_api_base,
            "embedding_api_key": self.embedding_api_key,
            "query_instruction": self.query_instruction,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "index_cache_dir": self.index_cache_dir,
            "query_prompt_name": self.query_prompt_name,
            "use_flash_attention": self.use_flash_attention,
            "query_format_version": self.query_format_version,
            "embedding_http_timeout": self.embedding_http_timeout,
            "embedding_http_max_retries": self.embedding_http_max_retries,
            "embedding_http_retry_delay": self.embedding_http_retry_delay,
            "embedding_dim": self.get_embedding_dim(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGConfig":
        """Create config from dictionary."""
        return cls(
            embedding_model=data.get("embedding_model", "Qwen3-Embedding-0.6B"),
            embedding_backend=data.get("embedding_backend", "openai_http"),
            embedding_api_base=data.get("embedding_api_base"),
            embedding_api_key=data.get("embedding_api_key"),
            query_instruction=data.get(
                "query_instruction",
                "Given a user request, retrieve relevant tools that can help complete the task",
            ),
            top_k=data.get("top_k", 15),
            similarity_threshold=data.get("similarity_threshold", 0.0),
            index_cache_dir=data.get("index_cache_dir", "./rag_index_cache"),
            query_prompt_name=data.get("query_prompt_name", "query"),
            use_flash_attention=data.get("use_flash_attention", False),
            query_format_version=data.get("query_format_version", "instruct_query_v1"),
            embedding_http_timeout=data.get("embedding_http_timeout", 600.0),
            embedding_http_max_retries=data.get("embedding_http_max_retries", 2),
            embedding_http_retry_delay=data.get("embedding_http_retry_delay", 1.0),
        )
