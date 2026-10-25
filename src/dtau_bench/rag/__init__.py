"""
RAG (Retrieval-Augmented Generation) module for tool retrieval.

Provides embedding-based semantic search to find relevant tools
from a large tool pool. Used by iterative RAG strategies to
dynamically expand the agent's visible tool set.

Components:
- RAGConfig: Configuration dataclass for embedding models and thresholds
- ToolIndexer: Builds and manages embedding indices for tools
- ToolDocument: Data class representing a tool as a searchable document
- ToolRetriever: Performs cosine similarity search over indexed tools
- ToolRAGRuntime: High-level runtime interface used by strategies
"""

from .config import RAGConfig
from .indexer import ToolIndexer, ToolDocument
from .retriever import ToolRetriever, RetrievalResult
from .runtime import ToolRAGRuntime

__all__ = [
    "RAGConfig",
    "ToolIndexer",
    "ToolDocument",
    "ToolRetriever",
    "RetrievalResult",
    "ToolRAGRuntime",
]
