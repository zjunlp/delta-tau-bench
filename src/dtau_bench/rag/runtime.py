"""
RAG Runtime — High-level interface for tool retrieval used by strategies.

This is the primary interface between the strategy layer and the RAG indexer/retriever.
Strategies call `ToolRAGRuntime.retrieve_tools()` to get relevant tools for a query.

Usage:
    runtime = ToolRAGRuntime.from_config(config_dict)
    result = runtime.retrieve_tools(
        query_text="I need to check account balance",
        all_tools_info=all_openai_tools,
        top_k=15,
    )
    # result["tool_names"]: List of retrieved tool names
    # result["tools_info"]: Filtered OpenAI-format tool defs
    # result["trace"]: Retrieval trace for artifacts
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import RAGConfig
from .indexer import ToolIndexer
from .retriever import ToolRetriever

logger = logging.getLogger(__name__)


def _tool_name_from_info(item: Dict[str, Any]) -> Optional[str]:
    """Extract tool name from OpenAI-format tool definition."""
    function = item.get("function") if isinstance(item, dict) else None
    if not isinstance(function, dict):
        return None
    name = str(function.get("name") or "").strip()
    return name or None


class ToolRAGRuntime:
    """High-level RAG runtime for tool retrieval.

    Wraps ToolRetriever with:
    - Tool info filtering (maps retrieved names back to full OpenAI tool defs)
    - Trace/debug output for artifacts
    - Configuration-based factory method

    Args:
        retriever: Initialized ToolRetriever with loaded index.
        config: The raw retrieval config dict (for trace metadata).
    """

    def __init__(self, retriever: ToolRetriever, config: Dict[str, Any]):
        self.retriever = retriever
        self.config = config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ToolRAGRuntime":
        """Create a ToolRAGRuntime from a strategy retrieval config dict.

        Expected config keys:
            - embedding_model: str
            - embedding_backend: str (default "openai_http")
            - embedding_api_base: str (required for openai_http)
            - embedding_api_key: str (optional)
            - query_instruction: str (optional)
            - top_k / top_k_init: int (default 15)
            - similarity_threshold: float (default 0.0)
            - index_dir: str (path to pre-built index directory)

        Args:
            config: Retrieval configuration dict.

        Returns:
            Initialized ToolRAGRuntime.

        Raises:
            ValueError: If index_dir is missing or index is incompatible.
            FileNotFoundError: If index directory doesn't exist.
        """
        rag_config = RAGConfig(
            embedding_model=config.get("embedding_model", "Qwen3-Embedding-0.6B"),
            embedding_backend=config.get("embedding_backend", "openai_http"),
            embedding_api_base=config.get("embedding_api_base"),
            embedding_api_key=config.get("embedding_api_key"),
            query_instruction=config.get(
                "query_instruction",
                "Given a user request, retrieve relevant tools that can help complete the task",
            ),
            query_prompt_name=config.get("query_prompt_name", "query"),
            top_k=int(config.get("top_k", config.get("top_k_init", 15))),
            similarity_threshold=float(config.get("similarity_threshold", 0.0)),
            use_flash_attention=bool(config.get("use_flash_attention", False)),
        )

        index_dir = config.get("index_dir")
        if not index_dir:
            raise ValueError(
                "RAG config missing 'index_dir': path to pre-built RAG index "
                "(use python -m dtau_bench build-tool-index to create one)"
            )

        index_path = Path(index_dir)
        if not index_path.exists():
            raise FileNotFoundError(
                f"RAG index directory not found: {index_path}\n"
                f"Build the index first with: python -m dtau_bench build-tool-index "
                f"--version <version> --output-dir {index_path}"
            )

        # Load index
        indexer = ToolIndexer(config=rag_config)
        index = indexer.load_index(str(index_path))

        # Validate compatibility
        if not indexer.validate_index(index):
            logger.warning(
                "RAG index metadata does not fully match current config. "
                "Results may be degraded. Consider rebuilding the index."
            )

        retriever = ToolRetriever(index=index, config=rag_config)
        logger.info(
            f"ToolRAGRuntime initialized: {retriever.total_tools} tools, "
            f"backend={rag_config.embedding_backend}"
        )
        return cls(retriever=retriever, config=config)

    def retrieve_tools(
        self,
        query_text: str,
        all_tools_info: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Retrieve relevant tools for a query.

        Args:
            query_text: The user's natural language query.
            all_tools_info: All available tools in OpenAI format (for filtering).
            top_k: Number of tools to retrieve (overrides config default).

        Returns:
            Dict with keys:
            - tool_names: List[str] — retrieved tool function names
            - tools_info: List[Dict] — filtered OpenAI-format tool definitions
            - debug: Dict — debug information
            - trace: Dict — retrieval trace for artifacts output
        """
        effective_top_k = top_k or int(self.config.get("top_k", 15))

        # Perform retrieval
        results = self.retriever.retrieve(
            query=query_text,
            top_k=effective_top_k,
            min_score=float(self.config.get("similarity_threshold", 0.0)),
        )

        # Map retrieved names back to OpenAI tool definitions
        selected_names = [r.function_name for r in results]
        selected_set = set(selected_names)

        tools_info: List[Dict[str, Any]] = []
        seen: set = set()
        for item in all_tools_info:
            name = _tool_name_from_info(item)
            if name and name in selected_set and name not in seen:
                tools_info.append(item)
                seen.add(name)

        # Build debug info
        debug = {
            "rag_query": query_text,
            "rag_selected_tool_names": selected_names,
            "rag_selected_tool_scores": {
                r.function_name: round(r.score, 4) for r in results
            },
            "rag_selected_tool_count": len(selected_names),
        }

        # Build trace for artifacts output
        trace = {
            "type": "tool_rag",
            "query": query_text,
            "top_k": effective_top_k,
            "similarity_threshold": float(
                self.config.get("similarity_threshold", 0.0)
            ),
            "retrieved_tools": [
                {"name": r.function_name, "score": round(r.score, 4)}
                for r in results
            ],
            "final_visible_tool_names": selected_names,
            "final_visible_tool_count": len(selected_names),
        }

        return {
            "tool_names": selected_names,
            "tools_info": tools_info,
            "debug": debug,
            "trace": trace,
        }
