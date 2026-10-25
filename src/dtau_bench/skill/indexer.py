"""
skill context Indexer — Build and load embedding indices for skill retrieval.

Creates two separate indices:
- Planning index: embeddings of (task, plan) pairs from plan_library.json
- Skills index: embeddings of functional + atomic skills

The index format (per sub-index):
  {index_root}/{planning,skills}/
    embeddings.npy    — shape (N, dim), float32
    documents.json    — serialized PlanningDocument/SkillDocument list
    metadata.json     — index metadata (model, dim, counts, etc.)

Usage:
    # Build indices
    indexer = SkillIndexer.from_rag_config(config)
    indices = indexer.build_indices("/path/to/skills")
    indexer.save_indices(indices, "/path/to/index_cache")

    # Load indices
    planning_index = indexer.load_index("/path/to/index_cache/planning", kind="planning")
    skills_index = indexer.load_index("/path/to/index_cache/skills", kind="skills")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from dtau_bench.rag.config import RAGConfig
from dtau_bench.rag.indexer import ToolIndexer

from .loader import SkillLibraryLoader
from .models import PlanningDocument, SkillDocument

logger = logging.getLogger(__name__)


def _as_tool_list(value):
    """``tools`` is a stringified list in some tool_skills.prepared.jsonl files."""
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item]
    text = str(value or "").strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        import ast

        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return [text]
        if isinstance(parsed, (list, tuple, set)):
            return [str(item) for item in parsed if item]
        return [str(parsed)]
    return [text]


def _tool_skill_document(item):
    """Build a SkillDocument from one tool_skills.prepared.jsonl row.

    Layout: ``{"embedding_text": ..., "skill": {"name","document","content","tools"}}``
    """
    if not isinstance(item, dict):
        return None
    skill = item.get("skill") if isinstance(item.get("skill"), dict) else item
    name = str(skill.get("name") or "").strip()
    if not name:
        return None
    embedding_text = str(item.get("embedding_text") or "").strip()
    if not embedding_text:
        embedding_text = f"{name}\n{skill.get('document') or ''}".strip()
    return SkillDocument(
        skill_type="tool",
        name=name,
        document=str(skill.get("document") or ""),
        content=str(skill.get("content") or ""),
        tools=_as_tool_list(skill.get("tools")) or [name],
        embedding_text=embedding_text,
        raw_skill=skill,
        metadata={k: v for k, v in item.items() if k not in {"skill", "embedding_text"}},
    )


class SkillIndexer:
    """Builds and loads skill context planning and skill indices.

    Delegates actual embedding generation to the RAG ToolIndexer,
    which handles both local and openai_http backends.

    Args:
        config: RAG config (embedding model, backend, API base, etc.)
        embedding_batch_size: Maximum texts per embedding request. ``None``
            uses the historical default of 64.
    """

    def __init__(
        self,
        config: RAGConfig,
        embedding_batch_size: Optional[int] = None,
    ):
        self.config = config
        # Re-use the ToolIndexer for embedding generation & HTTP calls
        self._tool_indexer = ToolIndexer(
            config=config,
            embedding_batch_size=embedding_batch_size,
        )
        self.embedding_batch_size = self._tool_indexer.embedding_batch_size

    @classmethod
    def from_rag_config(
        cls,
        config: RAGConfig,
        embedding_batch_size: Optional[int] = None,
    ) -> "SkillIndexer":
        """Create indexer from RAGConfig."""
        return cls(
            config=config,
            embedding_batch_size=embedding_batch_size,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Build Indices
    # ─────────────────────────────────────────────────────────────────────────

    def build_indices(self, skill_library_dir: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
        """Build planning and skills indices from a skill library directory.

        Args:
            skill_library_dir: Path to skill library directory.

        Returns:
            Dict with "planning" and "skills" keys, each containing:
            - documents: list of PlanningDocument/SkillDocument
            - embeddings: numpy array (N, dim)
            - metadata: dict
        """
        artifacts = SkillLibraryLoader.load(skill_library_dir, mode="normal")

        # Convert plans to PlanningDocuments
        planning_docs = [
            PlanningDocument(
                task=item["task"],
                plan=item["plan"],
                embedding_text=item["embedding_text"],
                metadata=item.get("metadata", {}),
            )
            for item in artifacts.plans
        ]

        # Convert skills to SkillDocuments
        skill_docs = [
            SkillDocument(
                skill_type=item["skill_type"],
                name=item["name"],
                document=item["document"],
                content=item["content"],
                tools=item["tools"],
                embedding_text=item["embedding_text"],
                raw_skill=item.get("raw_skill", {}),
                metadata=item.get("metadata", {}),
            )
            for item in artifacts.skills
        ]

        logger.info(
            f"Building skill context indices: {len(planning_docs)} plans, "
            f"{len(skill_docs)} skills"
        )

        indices = {
            "planning": self._build_index(
                documents=planning_docs,
                kind="planning",
                source_dir=artifacts.source_dir,
            ),
            "skills": self._build_index(
                documents=skill_docs,
                kind="skills",
                source_dir=artifacts.source_dir,
            ),
        }
        # ── Tool skills (P1): explorer-only index ──
        # Tool skills hold the parameter-level usage rules that plain schemas
        # lack. They are indexed under a SEPARATE key so the main strategy's
        # atomic/functional retrievers (which filter the "skills" index by
        # skill_type) stay byte-identical.
        tool_skill_docs = [
            doc for doc in (
                _tool_skill_document(item) for item in (artifacts.tool_skills or [])
            ) if doc is not None
        ]
        if tool_skill_docs:
            logger.info(f"Building skill context tool-skill index: {len(tool_skill_docs)} skills")
            indices["tool_skills"] = self._build_index(
                documents=tool_skill_docs,
                kind="tool_skills",
                source_dir=artifacts.source_dir,
            )

        return indices

    def _build_index(
        self,
        documents: List[Any],  # PlanningDocument | SkillDocument
        kind: str,
        source_dir: str,
    ) -> Dict[str, Any]:
        """Build a single sub-index (planning or skills).

        Args:
            documents: List of document objects.
            kind: "planning" or "skills".
            source_dir: Source directory (for metadata).

        Returns:
            Dict with documents, embeddings array, and metadata.
        """
        if not documents:
            raise ValueError(f"No skill context {kind} documents found in {source_dir}")

        texts = [doc.embedding_text for doc in documents]

        # Generate embeddings in batches
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.embedding_batch_size):
            batch = texts[i:i + self.embedding_batch_size]
            embeddings = self._tool_indexer._get_embeddings_batch(batch, is_query=False)
            all_embeddings.extend(embeddings)
            processed = min(i + self.embedding_batch_size, len(texts))
            if processed % 100 == 0 or processed >= len(texts):
                logger.info(f"  [{kind}] Embedded {processed}/{len(texts)}")

        # Assign embeddings to documents
        for doc, embedding in zip(documents, all_embeddings):
            doc.embedding = embedding

        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        metadata = {
            "kind": kind,
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
            "embedding_dim": self.config.get_embedding_dim(),
            "total_documents": len(documents),
            "created_at": datetime.now().isoformat(),
            "source_dir": source_dir,
        }

        return {
            "documents": documents,
            "embeddings": embeddings_array,
            "metadata": metadata,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Save / Load Indices
    # ─────────────────────────────────────────────────────────────────────────

    def save_indices(self, indices: Dict[str, Dict[str, Any]], output_root: Union[str, Path]) -> None:
        """Save both planning and skills indices to disk.

        Creates:
            {output_root}/planning/ — planning index
            {output_root}/skills/   — skills index
        """
        root = Path(output_root)
        for kind, index in indices.items():
            self._save_index(index, root / kind)
        logger.info(f"Saved skill context indices to {root}")

    def _save_index(self, index: Dict[str, Any], path: Path) -> None:
        """Save a single sub-index to disk."""
        path.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        np.save(path / "embeddings.npy", index["embeddings"])

        # Save documents (without embeddings to save space)
        docs = [doc.to_dict(include_embedding=False) for doc in index["documents"]]
        with (path / "documents.json").open("w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)

        # Save metadata
        with (path / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(index["metadata"], f, ensure_ascii=False, indent=2)

        logger.info(
            f"  Saved {index['metadata']['kind']} index: "
            f"{index['metadata']['total_documents']} documents → {path}"
        )

    def load_index(self, path: Union[str, Path], kind: str) -> Dict[str, Any]:
        """Load a pre-built sub-index from disk.

        Args:
            path: Directory containing the index files.
            kind: "planning" or "skills" (determines document type).

        Returns:
            Dict with documents, embeddings, and metadata.
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"skill context {kind} index not found at {load_path}")

        # Load embeddings
        embeddings = np.load(load_path / "embeddings.npy")

        # Load documents
        with (load_path / "documents.json").open("r", encoding="utf-8") as f:
            docs_data = json.load(f)

        # Reconstruct documents
        if kind == "planning":
            factory = PlanningDocument.from_dict
        else:
            factory = SkillDocument.from_dict

        documents = []
        for i, data in enumerate(docs_data):
            doc = factory(data)
            doc.embedding = embeddings[i].tolist()
            documents.append(doc)

        # Load metadata
        with (load_path / "metadata.json").open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        logger.info(
            f"Loaded skill context {kind} index: {len(documents)} documents from {load_path}"
        )

        return {
            "documents": documents,
            "embeddings": embeddings,
            "metadata": metadata,
        }

    def validate_index(self, index: Dict[str, Any]) -> bool:
        """Validate that a loaded index is compatible with current config.

        Checks embedding model, backend, and instruction match.
        """
        return self._tool_indexer.validate_index(index)

    # ─────────────────────────────────────────────────────────────────────────
    # Embedding Access (for runtime query embedding)
    # ─────────────────────────────────────────────────────────────────────────

    def get_query_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a query text (with instruction prefix if configured).

        This is used by the SkillContextRuntime to embed step/query text during retrieval.

        Args:
            text: Query text to embed.

        Returns:
            numpy array of shape (dim,), dtype float32.
        """
        embeddings = self._tool_indexer._get_embeddings_batch([text], is_query=True)
        return np.array(embeddings[0], dtype=np.float32)
