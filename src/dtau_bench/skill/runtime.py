"""
skill context Runtime — End-to-end skill retrieval and prompt injection.

The SkillContextRuntime orchestrates the full skill retrieval pipeline:
1. Planning retrieval: embed query → retrieve top_k planning skills
2. Pseudo-plan rewrite: LLM rewrites plans into task-specific steps
3. Per-step skill retrieval: embed each step → retrieve atomic/functional skills
4. Graph prioritization: (if enabled) prefer graph-linked atomic skills
5. Dedup: union-find clustering by embedding similarity
6. MMR selection: Maximal Marginal Relevance for diversity
7. LLM selection: final skill selection by LLM
8. Tool clarifications: attach linked tool skills via graph edges
9. Prompt construction: build system message injecting selected skills

Two main entry points:
- build_context_for_query(): Single-domain mode (returns prompt message + stats)
- retrieve_skills_for_step(): Lightweight per-step retrieval for iterative strategy

Usage:
    runtime = SkillContextRuntime.from_config(config, llm_client)
    prompt_msg, stats = runtime.build_context_for_query(query_text, tools_info)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from dtau_bench.rag.config import RAGConfig
from dtau_bench.utils.llm import LLMClient, LLMConfig

from .indexer import SkillIndexer
from .loader import SkillLibraryLoader
from .models import PlanningDocument, SkillDocument, SkillGraphState

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Data Classes
# ============================================================================


@dataclass
class RetrievalHit:
    """Vector retrieval result."""
    score: float
    document: Any  # PlanningDocument | SkillDocument
    embedding: Optional[np.ndarray] = None


@dataclass
class StepSkillHit:
    """Skill candidate retrieved for a pseudo-plan step."""
    score: float
    query_text: str
    document: SkillDocument
    document_embedding: np.ndarray
    relevance_score: float = 0.0


# ============================================================================
# Vector Retriever (cosine similarity over precomputed embeddings)
# ============================================================================


class VectorRetriever:
    """Simple cosine-similarity retriever over precomputed embeddings."""

    def __init__(self, index: Dict[str, Any]):
        self.documents = index["documents"]
        self.embeddings = index["embeddings"]
        if len(self.documents) == 0:
            self.normalized_embeddings = self.embeddings
            return
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.normalized_embeddings = self.embeddings / norms

    def retrieve(self, query_embedding: np.ndarray, top_k: int) -> List[RetrievalHit]:
        """Retrieve top_k most similar documents."""
        if top_k <= 0 or len(self.documents) == 0:
            return []
        similarities = self._compute_similarities(query_embedding)
        if similarities is None:
            return []
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [
            RetrievalHit(
                score=float(similarities[idx]),
                document=self.documents[idx],
                embedding=self.normalized_embeddings[idx],
            )
            for idx in top_indices
        ]

    def retrieve_above_threshold(
        self,
        query_embedding: np.ndarray,
        min_similarity: float,
        top_k: Optional[int] = None,
    ) -> List[RetrievalHit]:
        """Retrieve the highest-scoring documents above a threshold.

        ``top_k`` is applied after thresholding and score ordering.  Leaving it
        unset preserves the unbounded behavior used by older callers.
        """
        if top_k is not None and top_k <= 0:
            return []
        if len(self.documents) == 0:
            return []
        similarities = self._compute_similarities(query_embedding)
        if similarities is None:
            return []
        matched_indices = [
            idx for idx, score in enumerate(similarities) if float(score) >= min_similarity
        ]
        matched_indices.sort(key=lambda idx: float(similarities[idx]), reverse=True)
        if top_k is not None:
            matched_indices = matched_indices[:top_k]
        return [
            RetrievalHit(
                score=float(similarities[idx]),
                document=self.documents[idx],
                embedding=self.normalized_embeddings[idx],
            )
            for idx in matched_indices
        ]

    def _compute_similarities(self, query_embedding: np.ndarray) -> Optional[np.ndarray]:
        """Compute cosine similarities between query and all documents."""
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return None
        query_normalized = query_embedding / query_norm
        return np.dot(self.normalized_embeddings, query_normalized)


# ============================================================================
# skill context Runtime
# ============================================================================


class SkillContextRuntime:
    """End-to-end skill context runtime for skill retrieval and prompt injection.

    Used by the strategy layer to:
    1. Retrieve relevant skills for a query (single-domain or iterative)
    2. Build a skill context prompt to inject into the Agent's conversation

    The runtime is stateless across conversations (all session state is in
    StrategySessionState). It holds the loaded indices and configuration.
    """

    def __init__(
        self,
        planning_retriever: VectorRetriever,
        functional_retriever: VectorRetriever,
        atomic_retriever: VectorRetriever,
        indexer: SkillIndexer,
        llm_client: LLMClient,
        config: Dict[str, Any],
        graph_state: Optional[SkillGraphState] = None,
        tool_skill_retriever: Optional[VectorRetriever] = None,
    ):
        self.planning_retriever = planning_retriever
        self.functional_retriever = functional_retriever
        self.atomic_retriever = atomic_retriever
        # Explorer-only (P1). Deliberately NOT used by any main-path retrieval
        # method, so enabling it cannot change the strategy's behaviour.
        self.tool_skill_retriever = tool_skill_retriever
        self.indexer = indexer
        self.llm_client = llm_client
        self.config = config
        self.graph_state = graph_state
        self._apply_default_config_values()

    @classmethod
    def from_config(
        cls,
        skill_context_config: Dict[str, Any],
        llm_client: LLMClient,
        embedding_config: Optional[Dict[str, Any]] = None,
    ) -> "SkillContextRuntime":
        """Create a SkillContextRuntime from configuration.

        Args:
            skill_context_config: skill context strategy configuration containing:
                - skill_library_dir: Path to skill library directory
                - index_dir: Path to pre-built index directory
                - plan_top_k: Number of plans to retrieve
                - skill_top_k_per_step: Maximum skills retrieved for each pseudo-plan step
                - final_max_skills: Max skills to select
                - dedup_similarity_threshold: Threshold for dedup
                - mmr_lambda: MMR diversity parameter
                - candidate_min_similarity: Min similarity for skill candidates
                - enable_skill_graph: Whether to use skill graph
                - embedding_model: Embedding model name
                - embedding_backend: "openai_http" or "local"
                - embedding_api_base: API base URL
            llm_client: LLM client for pseudo-plan rewrite and skill selection.
            embedding_config: Optional embedding config override (from models.yaml).

        Returns:
            Initialized SkillContextRuntime.
        """
        skill_library_dir = Path(skill_context_config["skill_library_dir"])
        if not skill_library_dir.exists():
            raise FileNotFoundError(f"skill context directory not found: {skill_library_dir}")

        index_dir = Path(skill_context_config["index_dir"])

        # Build RAGConfig for embeddings
        emb_cfg = embedding_config or {}
        rag_config = RAGConfig(
            embedding_model=skill_context_config.get(
                "embedding_model", emb_cfg.get("model", "Qwen3-Embedding-0.6B")
            ),
            embedding_backend=skill_context_config.get(
                "embedding_backend", emb_cfg.get("backend", "openai_http")
            ),
            embedding_api_base=skill_context_config.get(
                "embedding_api_base", emb_cfg.get("base_url")
            ),
            embedding_api_key=skill_context_config.get(
                "embedding_api_key", emb_cfg.get("api_key")
            ),
            query_instruction=skill_context_config.get(
                "query_instruction",
                "Given a novel and complex agent task, retrieve relevant reusable skills that can guide solving this task",
            ),
            query_prompt_name=skill_context_config.get("query_prompt_name", "query"),
            use_flash_attention=bool(skill_context_config.get("use_flash_attention", False)),
        )

        # Normalize backend name
        if rag_config.embedding_backend == "openai_compatible":
            rag_config.embedding_backend = "openai_http"

        indexer = SkillIndexer(config=rag_config)

        # Load indices
        planning_path = index_dir / "planning"
        skills_path = index_dir / "skills"
        if not planning_path.exists() or not skills_path.exists():
            raise FileNotFoundError(
                f"skill context indices not found at {index_dir}. "
                f"Expected {planning_path} and {skills_path}. "
                "Run python -m dtau_bench build-skill-index first."
            )

        planning_index = indexer.load_index(planning_path, kind="planning")
        skills_index = indexer.load_index(skills_path, kind="skills")
        # Optional, explorer-only (P1). Absent for every library built before
        # this index kind existed; the explorer degrades with an explicit error
        # rather than silently returning no results.
        tool_skill_path = index_dir / "tool_skills"
        tool_skill_index = None
        if tool_skill_path.exists():
            try:
                tool_skill_index = indexer.load_index(tool_skill_path, kind="tool_skills")
            except Exception as exc:  # pragma: no cover - corrupt/partial index
                logger.warning("Failed to load tool-skill index at %s: %r", tool_skill_path, exc)

        # Validate
        if not indexer.validate_index(planning_index):
            logger.warning(
                "skill context planning index metadata does not match current config. "
                "Consider rebuilding."
            )
        if not indexer.validate_index(skills_index):
            logger.warning(
                "skill context skills index metadata does not match current config. "
                "Consider rebuilding."
            )

        # Build graph state if enabled
        graph_state = None
        if bool(skill_context_config.get("enable_skill_graph", False)):
            graph_dir = Path(skill_context_config.get("graph_dir", skill_context_config["skill_library_dir"]))
            graph_path = graph_dir / "skill_graph.json"
            if graph_path.exists():
                artifacts = SkillLibraryLoader.load(graph_dir)
                if artifacts.skill_graph:
                    graph_state = cls._build_graph_state(
                        graph_payload=artifacts.skill_graph,
                        tool_skills=artifacts.tool_skills,
                    )
                    if not graph_state.planning_to_atomic and not graph_state.atomic_to_tool:
                        # An edgeless graph silently disables graph
                        # prioritization and the explorer's expand_graph action
                        # while every trace still reports graph_enabled=true.
                        logger.warning(
                            "Skill graph at %s has NO edges: graph prioritization "
                            "and expand_graph will have no effect. Rebuild it.",
                            graph_path,
                        )
                    logger.info(
                        f"Loaded skill graph: "
                        f"{len(graph_state.planning_to_atomic)} plan→atomic edges, "
                        f"{len(graph_state.atomic_to_tool)} atomic→tool edges"
                    )
            else:
                logger.warning(f"Skill graph enabled but not found at {graph_path}")

        return cls(
            planning_retriever=VectorRetriever(planning_index),
            functional_retriever=VectorRetriever(
                cls._filter_skill_index(skills_index, skill_type="functional")
            ),
            atomic_retriever=VectorRetriever(
                cls._filter_skill_index(skills_index, skill_type="atomic")
            ),
            tool_skill_retriever=(
                VectorRetriever(tool_skill_index) if tool_skill_index else None
            ),
            indexer=indexer,
            llm_client=llm_client,
            config=skill_context_config,
            graph_state=graph_state,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Main Entry Point: Single-Domain Skill Context
    # ─────────────────────────────────────────────────────────────────────────

    def build_context_for_query(
        self,
        query_text: str,
        tools_info: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[Dict[str, str]], Dict[str, Any]]:
        """Build skill context prompt for a single-domain query.

        Full pipeline: plan retrieval → pseudo-plan → skill retrieval →
        graph prioritization → dedup → MMR → LLM select → prompt.

        Args:
            query_text: The user's query text.
            tools_info: Available tools (for pseudo-plan context). Optional.

        Returns:
            Tuple of:
            - Prompt message dict {"role": "system", "content": ...} or None
            - Stats/trace dict for artifacts output
        """
        selected_skills, pseudo_plan, pseudo_steps, stats, trace = (
            self._retrieve_selected_skills(query_text=query_text, tools_info=tools_info)
        )

        # Attach tool clarifications (via graph)
        tool_clarifications = self._attach_tool_clarifications(selected_skills)
        prompt = self._build_skill_prompt(selected_skills, tool_clarifications=tool_clarifications)

        stats["attached_tool_skill_names"] = [item["name"] for item in tool_clarifications]
        stats["attached_tool_skill_count"] = len(tool_clarifications)
        stats["retrieval_trace"] = trace

        if not prompt:
            return None, stats

        return {"role": "system", "content": prompt}, stats

    # ─────────────────────────────────────────────────────────────────────────
    # Main Entry Point: All-Tools Skill + Tool RAG Context
    # ─────────────────────────────────────────────────────────────────────────

    def build_all_tools_context_for_query(
        self,
        query_text: str,
        all_tools_info: List[Dict[str, Any]],
        tool_rag_runtime: Any = None,
        tool_top_k_per_step: int = 3,
    ) -> Tuple[Optional[Dict[str, str]], List[str], Dict[str, Any]]:
        """Build skill context AND initial tool set for all-tools mode.

        Combines skill context retrieval with per-step tool RAG:
        1. Full skill retrieval pipeline (same as build_context_for_query)
        2. Collect tool names from selected skills (skill_tool_names)
        3. Per-step tool RAG retrieval using pseudo-plan steps
        4. Merge: skill-recommended tools ∪ per-step RAG tools = initial visible tools

        Args:
            query_text: The user's query text.
            all_tools_info: All available tools in OpenAI format (cross-domain).
            tool_rag_runtime: ToolRAGRuntime instance for per-step tool retrieval.
                If None, only skill-recommended tools are used.
            tool_top_k_per_step: Number of tools to retrieve per pseudo-plan step.

        Returns:
            Tuple of:
            - Prompt message dict {"role": "system", "content": ...} or None
            - List of initial visible tool names (from skills + per-step RAG)
            - Stats/trace dict for artifacts output
        """
        # Run the full skill retrieval pipeline
        selected_skills, pseudo_plan, pseudo_steps, stats, trace = (
            self._retrieve_selected_skills(query_text=query_text, tools_info=all_tools_info)
        )

        # Attach tool clarifications (via graph)
        tool_clarifications = self._attach_tool_clarifications(selected_skills)
        prompt = self._build_skill_prompt(selected_skills, tool_clarifications=tool_clarifications)

        stats["attached_tool_skill_names"] = [item["name"] for item in tool_clarifications]
        stats["attached_tool_skill_count"] = len(tool_clarifications)
        stats["retrieval_trace"] = trace

        # --- Collect initial tool names ---
        # Source 1: Tool names from selected skills
        skill_tool_names = self._collect_skill_tool_names(selected_skills)

        # Also include tool names from tool clarifications
        for tc in tool_clarifications:
            for tool_name in tc.get("tools", []):
                normalized = str(tool_name).strip()
                if normalized and normalized not in skill_tool_names:
                    skill_tool_names.append(normalized)

        # Source 2: Per-step tool RAG retrieval
        per_step_rag_tools: List[str] = []
        per_step_rag_trace: List[Dict[str, Any]] = []
        if tool_rag_runtime is not None and pseudo_steps:
            all_tool_names_set = {
                self._extract_tool_name_from_info(item)
                for item in all_tools_info
                if self._extract_tool_name_from_info(item)
            }
            seen_rag_tools: set = set()
            for step in pseudo_steps:
                try:
                    rag_result = tool_rag_runtime.retrieve_tools(
                        query_text=step,
                        all_tools_info=all_tools_info,
                        top_k=tool_top_k_per_step,
                    )
                    step_tool_names = rag_result.get("tool_names", [])
                    newly_added = [
                        name for name in step_tool_names
                        if name not in seen_rag_tools
                    ]
                    seen_rag_tools.update(newly_added)
                    per_step_rag_tools.extend(newly_added)
                    per_step_rag_trace.append({
                        "step": step,
                        "retrieved_tools": step_tool_names,
                        "newly_added": newly_added,
                    })
                except Exception as e:
                    logger.warning(f"Per-step tool RAG failed for step '{step}': {e}")
                    per_step_rag_trace.append({
                        "step": step,
                        "error": str(e),
                        "retrieved_tools": [],
                        "newly_added": [],
                    })

        # Merge: skill tools ∪ per-step RAG tools (preserve order, deduplicate)
        initial_tool_names: List[str] = []
        seen_names: set = set()
        for name in skill_tool_names + per_step_rag_tools:
            if name not in seen_names:
                seen_names.add(name)
                initial_tool_names.append(name)

        # Validate against all available tools
        all_available_tool_names = {
            self._extract_tool_name_from_info(item)
            for item in all_tools_info
            if self._extract_tool_name_from_info(item)
        }
        initial_tool_names = [
            name for name in initial_tool_names
            if name in all_available_tool_names
        ]

        # Enrich stats
        stats["skill_tool_names"] = skill_tool_names
        stats["skill_tool_count"] = len(skill_tool_names)
        stats["per_step_rag_tool_names"] = per_step_rag_tools
        stats["per_step_rag_tool_count"] = len(per_step_rag_tools)
        stats["initial_tool_names"] = initial_tool_names
        stats["initial_tool_count"] = len(initial_tool_names)
        stats["pseudo_steps"] = pseudo_steps

        # Enrich trace
        trace["tool_retrieval"] = {
            "skill_tool_names": skill_tool_names,
            "skill_tool_count": len(skill_tool_names),
            "per_step_rag": per_step_rag_trace,
            "per_step_rag_tool_names": per_step_rag_tools,
            "per_step_rag_tool_count": len(per_step_rag_tools),
            "initial_tool_names": initial_tool_names,
            "initial_tool_count": len(initial_tool_names),
            "tool_top_k_per_step": tool_top_k_per_step,
        }

        prompt_msg = {"role": "system", "content": prompt} if prompt else None
        return prompt_msg, initial_tool_names, stats

    # ─────────────────────────────────────────────────────────────────────────
    # Iterative Skill Retrieval (per-step)
    # ─────────────────────────────────────────────────────────────────────────

    def retrieve_skills_for_step(
        self,
        step_text: str,
        top_k: int = 3,
        excluded_atomic_names: Optional[Set[str]] = None,
        excluded_tool_skill_names: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Retrieve atomic skills for a single step (iterative mode).

        Lightweight per-step retrieval used by the iterative skill RAG strategy:
        1. Embed step text
        2. Retrieve atomic skills above threshold
        3. Filter already-seen skills
        4. Prioritize graph-linked atomics
        5. Take top_k
        6. Attach linked tool skills via graph

        Args:
            step_text: The pseudo-plan step to retrieve skills for.
            top_k: Number of skills to return.
            excluded_atomic_names: Already-seen atomic skill names to skip.
            excluded_tool_skill_names: Already-seen tool skill names to skip.

        Returns:
            Dict with keys:
            - atomic_skills: list of SkillDocument
            - tool_skills: list of tool skill dicts
            - trace: retrieval trace
        """
        if excluded_atomic_names is None:
            excluded_atomic_names = set()
        if excluded_tool_skill_names is None:
            excluded_tool_skill_names = set()

        step_embedding = self.indexer.get_query_embedding(step_text)
        min_similarity = float(self.config.get("candidate_min_similarity", 0.45))

        # Retrieve atomic candidates above threshold
        atomic_hits = self.atomic_retriever.retrieve_above_threshold(
            step_embedding, min_similarity
        )

        # Filter excluded
        filtered_hits: List[RetrievalHit] = []
        for hit in atomic_hits:
            if not isinstance(hit.document, SkillDocument):
                continue
            if hit.document.name in excluded_atomic_names:
                continue
            filtered_hits.append(hit)

        # Graph prioritization
        if self._skill_graph_enabled() and self.graph_state is not None:
            all_linked_atomics: Set[str] = set()
            for atomic_set in self.graph_state.planning_to_atomic.values():
                all_linked_atomics.update(atomic_set)
            filtered_hits.sort(
                key=lambda hit: (
                    1 if hit.document.name in all_linked_atomics else 0,
                    hit.score,
                ),
                reverse=True,
            )

        # Take top_k
        selected_hits = filtered_hits[:top_k]
        selected_atomic_skills: List[SkillDocument] = [hit.document for hit in selected_hits]

        # Attach linked tool skills via graph
        new_tool_skills: List[Dict[str, Any]] = []
        if self._skill_graph_enabled() and self.graph_state is not None:
            for skill in selected_atomic_skills:
                linked_tool_names = self.graph_state.atomic_to_tool.get(skill.name, set())
                for tool_name in linked_tool_names:
                    if tool_name in excluded_tool_skill_names:
                        continue
                    tool_skill = self.graph_state.tool_skill_by_tool_name.get(tool_name)
                    if tool_skill is None:
                        continue
                    excluded_tool_skill_names.add(tool_name)
                    new_tool_skills.append({
                        "name": tool_name,
                        "document": tool_skill.get("document", ""),
                        "content": tool_skill.get("content", ""),
                        "tools": list(tool_skill.get("tools", [])),
                    })

        # Build trace
        trace = {
            "type": "skill_retrieval_turn",
            "step": step_text,
            "all_atomic_hits_count": len(atomic_hits),
            "filtered_hits_count": len(filtered_hits),
            "selected_atomic_skills": [
                {"name": hit.document.name, "score": hit.score, "skill_type": hit.document.skill_type}
                for hit in selected_hits
            ],
            "selected_atomic_count": len(selected_atomic_skills),
            "attached_tool_skills": [{"name": ts["name"]} for ts in new_tool_skills],
            "attached_tool_skill_count": len(new_tool_skills),
        }

        return {
            "atomic_skills": selected_atomic_skills,
            "tool_skills": new_tool_skills,
            "trace": trace,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Core Retrieval Pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def _retrieve_selected_skills(
        self,
        query_text: str,
        tools_info: Optional[List[Dict[str, Any]]],
    ) -> Tuple[List[SkillDocument], str, List[str], Dict[str, Any], Dict[str, Any]]:
        """Run the full skill retrieval pipeline.

        Returns:
            Tuple of (selected_skills, pseudo_plan, pseudo_steps, stats, trace)
        """
        query_embedding = self.indexer.get_query_embedding(query_text)
        query_normalized = self._normalize_embedding(query_embedding)

        # Step 1: Planning retrieval
        plan_top_k = int(self.config.get("plan_top_k", 3))
        plan_hits = self.planning_retriever.retrieve(query_embedding, plan_top_k)

        # Step 2: Pseudo-plan rewrite
        pseudo_plan = self._rewrite_pseudo_plan(query_text, plan_hits, tools_info=tools_info)
        pseudo_steps = self.parse_pseudo_plan_steps(pseudo_plan)
        if not pseudo_steps:
            pseudo_steps = [pseudo_plan.strip() or query_text]

        # Step 3: Per-step skill retrieval
        candidate_hits: List[StepSkillHit] = []
        per_step_skill_hits: List[Dict[str, Any]] = []
        functional_enabled = self._functional_skill_retrieval_enabled()
        min_similarity = float(self.config.get("candidate_min_similarity", 0.45))
        skill_top_k_per_step = int(self.config.get("skill_top_k_per_step", 5))

        # Get graph-linked atomic names
        planning_hit_names = [
            hit.document.task
            for hit in plan_hits
            if isinstance(hit.document, PlanningDocument)
        ]
        allowed_atomic_names = self._linked_atomic_names_for_plans(planning_hit_names)

        for step in pseudo_steps:
            step_embedding = self.indexer.get_query_embedding(step)

            functional_hits: List[RetrievalHit] = []
            if functional_enabled:
                functional_hits = self.functional_retriever.retrieve_above_threshold(
                    step_embedding,
                    min_similarity,
                    top_k=skill_top_k_per_step,
                )

            atomic_hits = self.atomic_retriever.retrieve_above_threshold(
                step_embedding,
                min_similarity,
                top_k=skill_top_k_per_step,
            )

            per_step_skill_hits.append({
                "step": step,
                "functional_hits": [
                    {"name": hit.document.name, "skill_type": hit.document.skill_type, "score": hit.score}
                    for hit in functional_hits
                    if isinstance(hit.document, SkillDocument)
                ],
                "atomic_hits": [
                    {"name": hit.document.name, "skill_type": hit.document.skill_type, "score": hit.score}
                    for hit in atomic_hits
                    if isinstance(hit.document, SkillDocument)
                ],
            })

            for hit in functional_hits + atomic_hits:
                if not isinstance(hit.document, SkillDocument) or hit.embedding is None:
                    continue
                candidate_hits.append(
                    StepSkillHit(
                        score=hit.score,
                        query_text=step,
                        document=hit.document,
                        document_embedding=np.array(hit.embedding, dtype=np.float32),
                        relevance_score=float(np.dot(query_normalized, hit.embedding)),
                    )
                )

        # Limit candidates
        candidate_hits = self._limit_candidate_hits(candidate_hits, max_candidates=100)

        # Step 4: Graph prioritization
        if self._skill_graph_enabled():
            candidate_hits = self._prioritize_graph_atomic_hits(candidate_hits, allowed_atomic_names)

        # Step 5: Dedup
        deduped_hits = self._dedup_skill_hits(candidate_hits)

        # Step 6: MMR selection
        mmr_hits = self._mmr_select_hits(deduped_hits, query_normalized)

        # Step 7: LLM selection
        selected_skills = self._select_skills(
            query_text, pseudo_plan, [hit.document for hit in mmr_hits]
        )

        # Build stats and trace
        stats = {
            "plan_candidates_count": len(plan_hits),
            "step_skill_hits_count_before_dedup": len(candidate_hits),
            "skill_candidates_count_after_dedup": len(deduped_hits),
            "mmr_selected_skills_count": len(mmr_hits),
            "selected_skills_count": len(selected_skills),
            "selected_skill_names": [skill.name for skill in selected_skills],
            "pseudo_plan": pseudo_plan,
            "graph_enabled": self._skill_graph_enabled(),
        }

        # Detailed trace
        linked_atomic_names = sorted({
            hit.document.name
            for hit in candidate_hits
            if hit.document.skill_type == "atomic" and hit.document.name in allowed_atomic_names
        })
        selected_atomic_skill_names = [
            skill.name for skill in selected_skills if skill.skill_type == "atomic"
        ]
        tool_clarifications = self._attach_tool_clarifications(selected_skills)

        trace = {
            "type": "skill_context",
            "query": query_text,
            "graph_enabled": self._skill_graph_enabled(),
            "planning_hit_names": planning_hit_names,
            "selected_atomic_skill_names": selected_atomic_skill_names,
            "planning": {
                "top_k": plan_top_k,
                "hits": [
                    {"task": hit.document.task, "score": hit.score}
                    for hit in plan_hits
                ],
            },
            "pseudo_plan": pseudo_plan,
            "pseudo_steps": pseudo_steps,
            "step_skill_retrieval": {
                "enable_functional_skills": functional_enabled,
                "candidate_min_similarity": min_similarity,
                "skill_top_k_per_step": skill_top_k_per_step,
                "max_candidates_before_dedup": 100,
                "per_step_hits": per_step_skill_hits,
                "candidate_hit_count_before_dedup": len(candidate_hits),
                "atomic_candidates_linked_to_plans": linked_atomic_names,
            },
            "dedup": {
                "similarity_threshold": float(self.config.get("dedup_similarity_threshold", 0.95)),
                "remaining_skills": [
                    {"name": hit.document.name, "skill_type": hit.document.skill_type, "score": hit.score}
                    for hit in deduped_hits
                ],
                "remaining_count": len(deduped_hits),
            },
            "mmr": {
                "lambda": float(self.config.get("mmr_lambda", 0.75)),
                "max_selected": int(self.config.get("final_max_skills", 8)),
                "selected_skills": [
                    {"name": hit.document.name, "skill_type": hit.document.skill_type, "relevance_score": hit.relevance_score}
                    for hit in mmr_hits
                ],
                "selected_count": len(mmr_hits),
            },
            "llm_select": {
                "final_max_skills": int(self.config.get("final_max_skills", 8)),
                "selected_skills": [
                    {"name": skill.name, "skill_type": skill.skill_type, "tools": list(skill.tools)}
                    for skill in selected_skills
                ],
                "selected_count": len(selected_skills),
            },
            "tool_clarifications": {
                "attached_tool_skill_names": [item["name"] for item in tool_clarifications],
                "attached_tool_skill_count": len(tool_clarifications),
            },
        }

        return selected_skills, pseudo_plan, pseudo_steps, stats, trace

    # ─────────────────────────────────────────────────────────────────────────
    # Pseudo-Plan Rewrite (via LLM)
    # ─────────────────────────────────────────────────────────────────────────

    def _rewrite_pseudo_plan(
        self,
        query_text: str,
        plan_hits: List[RetrievalHit],
        tools_info: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Rewrite retrieved planning skills into a task-specific pseudo-plan."""
        if not plan_hits:
            return query_text

        retrieved_text = "\n\n".join(
            f"[Plan {idx + 1}] Task: {hit.document.task}\n{hit.document.plan}"
            for idx, hit in enumerate(plan_hits)
        )

        system_prompt = (
            "You are rewriting retrieved planning skills into a task-specific pseudo-plan. "
            "Use the current request and the retrieved plans. "
            'After thinking, return strict JSON only with the shape {"steps":["step 1","step 2"]}. '
            "Each step must be a short actionable string. "
            "Do not include commentary or any text outside the JSON."
        )
        user_prompt = f"Current request:\n{query_text}\n\nRetrieved planning skills:\n{retrieved_text}"

        # Add available tools context for single-domain mode
        available_tools = self._format_available_tools(tools_info)
        if available_tools:
            user_prompt += f"\n\nCurrently available tools:\n{available_tools}"

        response = self._llm_complete(system_prompt, user_prompt)
        return self._normalize_pseudo_plan_text(response, query_text)

    # ─────────────────────────────────────────────────────────────────────────
    # LLM Skill Selection
    # ─────────────────────────────────────────────────────────────────────────

    def _select_skills(
        self,
        query_text: str,
        pseudo_plan: str,
        skills: List[SkillDocument],
    ) -> List[SkillDocument]:
        """Use LLM to select the most applicable skills from MMR candidates."""
        if not skills:
            return []
        final_max = int(self.config.get("final_max_skills", 8))
        if final_max <= 0:
            return []

        candidate_text = "\n\n".join(
            f"[{idx}] Type: {skill.skill_type}\n"
            f"Name: {skill.name}\n"
            f"Document: {skill.document}\n"
            f"Content: {skill.content}\n"
            f"Tools: {', '.join(skill.tools)}"
            for idx, skill in enumerate(skills, start=1)
        )

        system_prompt = (
            "Select the most applicable reusable skills for solving the current request. "
            "Consider the request and pseudo-plan. "
            "Return strict JSON with the shape "
            '{"selected_indices":[1,2,...]}. '
            f"Select at most {final_max} skills."
        )
        user_prompt = (
            f"Current request:\n{query_text}\n\n"
            f"Pseudo-plan:\n{pseudo_plan}\n\n"
            f"Candidate skills:\n{candidate_text}"
        )

        response = self._llm_complete(system_prompt, user_prompt)
        selected_indices = self._parse_selected_indices(response)

        if not selected_indices:
            return skills[:final_max]

        selected: List[SkillDocument] = []
        seen = set()
        for index in selected_indices:
            if 1 <= index <= len(skills) and index not in seen:
                seen.add(index)
                selected.append(skills[index - 1])
        return selected[:final_max]

    # ─────────────────────────────────────────────────────────────────────────
    # Graph & Tool Clarifications
    # ─────────────────────────────────────────────────────────────────────────

    def _linked_atomic_names_for_plans(self, planning_hit_names: List[str]) -> Set[str]:
        """Get atomic skill names linked to retrieved plans via graph."""
        if not self._skill_graph_enabled() or self.graph_state is None:
            return set()
        linked: Set[str] = set()
        for name in planning_hit_names:
            linked.update(self.graph_state.planning_to_atomic.get(name, set()))
        return linked

    def _attach_tool_clarifications(self, selected_skills: List[SkillDocument]) -> List[Dict[str, Any]]:
        """Attach linked tool skills to selected atomic skills via graph."""
        if not self._skill_graph_enabled() or self.graph_state is None:
            return []
        attached: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for skill in selected_skills:
            if skill.skill_type != "atomic":
                continue
            linked_tools = self.graph_state.atomic_to_tool.get(skill.name, set())
            for tool_name in skill.tools:
                candidate_names = [tool_name]
                if linked_tools:
                    candidate_names = [n for n in linked_tools if n == tool_name] or [tool_name]
                for candidate_name in candidate_names:
                    tool_skill = self.graph_state.tool_skill_by_tool_name.get(candidate_name)
                    if not tool_skill or candidate_name in seen:
                        continue
                    seen.add(candidate_name)
                    attached.append({
                        "name": candidate_name,
                        "document": tool_skill.get("document", ""),
                        "content": tool_skill.get("content", ""),
                        "tools": list(tool_skill.get("tools", [])),
                    })
        return attached

    # ─────────────────────────────────────────────────────────────────────────
    # Dedup (Union-Find by Embedding Similarity)
    # ─────────────────────────────────────────────────────────────────────────

    def _dedup_skill_hits(self, hits: List[StepSkillHit]) -> List[StepSkillHit]:
        """Deduplicate skill hits using union-find on embedding similarity."""
        threshold = float(self.config.get("dedup_similarity_threshold", 0.95))
        if not hits:
            return []

        parent = list(range(len(hits)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        for i in range(len(hits)):
            for j in range(i + 1, len(hits)):
                similarity = float(np.dot(hits[i].document_embedding, hits[j].document_embedding))
                if similarity > threshold:
                    union(i, j)

        groups: Dict[int, List[StepSkillHit]] = {}
        for idx, hit in enumerate(hits):
            groups.setdefault(find(idx), []).append(hit)

        deduped: List[StepSkillHit] = []
        for group in groups.values():
            deduped.append(
                max(group, key=lambda h: (h.score, h.relevance_score, h.document.name))
            )

        deduped.sort(key=lambda h: (h.score, h.relevance_score), reverse=True)
        return deduped

    # ─────────────────────────────────────────────────────────────────────────
    # MMR Selection (Maximal Marginal Relevance)
    # ─────────────────────────────────────────────────────────────────────────

    def _mmr_select_hits(
        self,
        hits: List[StepSkillHit],
        query_normalized: np.ndarray,
    ) -> List[StepSkillHit]:
        """Select skills using MMR for diversity."""
        final_max = int(self.config.get("final_max_skills", 8))
        if not hits or final_max <= 0:
            return []

        lambda_weight = float(self.config.get("mmr_lambda", 0.75))
        remaining = list(hits)
        selected: List[StepSkillHit] = []

        # Compute relevance scores
        for hit in remaining:
            hit.relevance_score = float(np.dot(query_normalized, hit.document_embedding))

        while remaining and len(selected) < final_max:
            best_hit: Optional[StepSkillHit] = None
            best_score = -float("inf")
            for hit in remaining:
                redundancy = 0.0
                if selected:
                    redundancy = max(
                        float(np.dot(hit.document_embedding, chosen.document_embedding))
                        for chosen in selected
                    )
                mmr_score = (lambda_weight * hit.relevance_score) - ((1.0 - lambda_weight) * redundancy)
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_hit = hit
            if best_hit is None:
                break
            selected.append(best_hit)
            remaining.remove(best_hit)

        return selected

    # ─────────────────────────────────────────────────────────────────────────
    # Prompt Construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_skill_prompt(
        self,
        skills: List[SkillDocument],
        *,
        tool_clarifications: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build the skill context prompt to inject into Agent's conversation."""
        if not skills and not tool_clarifications:
            return ""

        sections = [
            (
                "Reusable reference skills. Use them only when applicable to the current request. "
                "Any tools mentioned in these skills are only references and may not all be available "
                "in the current environment. You must rely on the actual tool list provided in this session "
                "when deciding which tools can be called."
            )
        ]

        grouped: Dict[str, List[SkillDocument]] = {"functional": [], "atomic": []}
        for skill in skills:
            grouped.setdefault(skill.skill_type, []).append(skill)

        ordered_skill_types = ("atomic",) if self._skill_graph_enabled() else ("functional", "atomic")
        for skill_type in ordered_skill_types:
            items = grouped.get(skill_type, [])
            if not items:
                continue
            title = "Functional Skills" if skill_type == "functional" else "Selected Atomic Skills"
            sections.append(title + ":")
            for skill in items:
                sections.append(
                    "\n".join([
                        f"Name: {skill.name}",
                        f"Document: {skill.document}",
                        f"Content: {skill.content}",
                        f"Tools: {', '.join(skill.tools)}" if skill.tools else "Tools: ",
                    ])
                )

        if tool_clarifications:
            sections.append("Tool Clarifications:")
            for tool_skill in tool_clarifications:
                sections.append(
                    "\n".join([
                        f"Name: {tool_skill.get('name', '')}",
                        f"Document: {tool_skill.get('document', '')}",
                        f"Content: {tool_skill.get('content', '')}",
                        f"Tools: {', '.join(tool_skill.get('tools', []))}" if tool_skill.get("tools") else "Tools: ",
                    ])
                )

        return "\n\n".join(sections).strip()

    # ─────────────────────────────────────────────────────────────────────────
    # LLM Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _llm_complete(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM for structured output (pseudo-plan rewrite, skill selection)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.llm_client.chat(messages=messages)
        return self.llm_client.extract_content(response).strip()

    # ─────────────────────────────────────────────────────────────────────────
    # Configuration Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_default_config_values(self) -> None:
        """Set default config values if not provided."""
        self.config.setdefault("enable_functional_skills", True)
        self.config.setdefault("enable_skill_graph", False)
        self.config.setdefault("plan_top_k", 3)
        self.config.setdefault("skill_top_k_per_step", 5)
        self.config.setdefault("final_max_skills", 8)
        self.config.setdefault("dedup_similarity_threshold", 0.95)
        self.config.setdefault("mmr_lambda", 0.75)
        self.config.setdefault("candidate_min_similarity", 0.45)

    def _functional_skill_retrieval_enabled(self) -> bool:
        """Whether functional skill retrieval is enabled."""
        if self._skill_graph_enabled():
            return False  # When graph is enabled, only atomics are used
        return bool(self.config.get("enable_functional_skills", True))

    def _skill_graph_enabled(self) -> bool:
        """Whether the skill graph is enabled and loaded."""
        return bool(self.config.get("enable_skill_graph", False) and self.graph_state is not None)

    # ─────────────────────────────────────────────────────────────────────────
    # Static Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_graph_state(
        *,
        graph_payload: Dict[str, Any],
        tool_skills: List[Dict[str, Any]],
    ) -> SkillGraphState:
        """Parse skill_graph.json into a SkillGraphState."""
        nodes = graph_payload.get("nodes", [])
        edges = graph_payload.get("edges", [])

        node_map: Dict[str, Dict[str, Any]] = {}
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("node_id", "")).strip()
            if node_id:
                node_map[node_id] = node

        planning_to_atomic: Dict[str, Set[str]] = {}
        atomic_to_tool: Dict[str, Set[str]] = {}

        for edge in edges:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source", "")).strip()
            target = str(edge.get("target", "")).strip()
            edge_type = str(edge.get("edge_type", "")).strip()
            if not source or not target or not edge_type:
                continue

            if edge_type == "plan_to_atomic":
                planning_node = node_map.get(source, {})
                atomic_node = node_map.get(target, {})
                planning_name = str(planning_node.get("name", "")).strip()
                atomic_name = str(atomic_node.get("name", "")).strip()
                if planning_name and atomic_name:
                    planning_to_atomic.setdefault(planning_name, set()).add(atomic_name)
            elif edge_type == "atomic_to_tool":
                atomic_node = node_map.get(source, {})
                tool_node = node_map.get(target, {})
                atomic_name = str(atomic_node.get("name", "")).strip()
                tool_name = str(tool_node.get("name", "")).strip()
                if atomic_name and tool_name:
                    atomic_to_tool.setdefault(atomic_name, set()).add(tool_name)

        # Build tool_skill_by_tool_name lookup
        tool_skill_by_tool_name: Dict[str, Dict[str, Any]] = {}
        for item in tool_skills:
            if not isinstance(item, dict):
                continue
            raw_skill = item.get("raw_skill", {})
            if not isinstance(raw_skill, dict):
                raw_skill = {}
            tool_name = str(item.get("name", "") or raw_skill.get("name", "")).strip()
            if tool_name and tool_name not in tool_skill_by_tool_name:
                tool_skill_by_tool_name[tool_name] = item

        return SkillGraphState(
            planning_to_atomic=planning_to_atomic,
            atomic_to_tool=atomic_to_tool,
            tool_skill_by_tool_name=tool_skill_by_tool_name,
            graph_payload=graph_payload,
        )

    @staticmethod
    def _filter_skill_index(index: Dict[str, Any], skill_type: str) -> Dict[str, Any]:
        """Filter a skills index to only contain one skill_type."""
        filtered_documents: List[SkillDocument] = []
        filtered_embeddings: List[Any] = []

        for idx, document in enumerate(index["documents"]):
            if isinstance(document, SkillDocument) and document.skill_type == skill_type:
                filtered_documents.append(document)
                filtered_embeddings.append(index["embeddings"][idx])

        if filtered_embeddings:
            embeddings = np.array(filtered_embeddings, dtype=np.float32)
        else:
            source_embeddings = index["embeddings"]
            embedding_dim = 0
            if hasattr(source_embeddings, "shape") and len(source_embeddings.shape) > 1:
                embedding_dim = int(source_embeddings.shape[1])
            embeddings = np.zeros((0, embedding_dim), dtype=np.float32)

        metadata = dict(index.get("metadata", {}))
        metadata["skill_type"] = skill_type
        metadata["total_documents"] = len(filtered_documents)

        return {
            "documents": filtered_documents,
            "embeddings": embeddings,
            "metadata": metadata,
        }

    @staticmethod
    def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """L2-normalize an embedding vector."""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    @staticmethod
    def _prioritize_graph_atomic_hits(
        hits: List[StepSkillHit],
        allowed_atomic_names: Set[str],
    ) -> List[StepSkillHit]:
        """Prioritize atomic skills that are linked to retrieved plans."""
        if not allowed_atomic_names:
            return hits
        return sorted(
            hits,
            key=lambda hit: (
                1 if hit.document.skill_type == "atomic" and hit.document.name in allowed_atomic_names else 0,
                hit.score,
                hit.relevance_score,
                hit.document.name,
            ),
            reverse=True,
        )

    @staticmethod
    def _limit_candidate_hits(hits: List[StepSkillHit], max_candidates: int) -> List[StepSkillHit]:
        """Limit number of candidates before dedup."""
        if max_candidates <= 0 or len(hits) <= max_candidates:
            return hits
        return sorted(
            hits,
            key=lambda h: (h.score, h.relevance_score, h.document.name),
            reverse=True,
        )[:max_candidates]

    @staticmethod
    def parse_pseudo_plan_steps(pseudo_plan: str) -> List[str]:
        """Parse a pseudo-plan into individual steps.

        Handles both JSON format {"steps": [...]} and numbered/bulleted text.
        """
        # Try JSON parse first
        parsed_json = SkillContextRuntime._parse_pseudo_plan_json(pseudo_plan)
        if parsed_json is not None:
            return parsed_json

        # Fall back to text parsing
        lines = [line.strip() for line in pseudo_plan.splitlines() if line.strip()]
        if not lines:
            return []

        step_lines = []
        for line in lines:
            normalized = re.sub(r"^\s*(?:[#*-]|\d+[.)])\s*", "", line).strip()
            normalized = re.sub(r"^(?:step\s*\d+\s*:\s*)", "", normalized, flags=re.I).strip()
            if re.match(r"^(?:step|#\s*step)\s*\d+", line, flags=re.I) or re.match(r"^\d+[.)]", line):
                if normalized:
                    step_lines.append(normalized)
            elif line.lower().startswith("# step"):
                step_lines.append(normalized)

        if step_lines:
            return step_lines
        return lines

    @staticmethod
    def _parse_pseudo_plan_json(response_text: str) -> Optional[List[str]]:
        """Try to parse pseudo-plan from JSON response."""
        def extract_steps(payload: Any) -> Optional[List[str]]:
            if not isinstance(payload, dict):
                return None
            steps = payload.get("steps")
            if not isinstance(steps, list):
                return None
            normalized = [str(step).strip() for step in steps if str(step).strip()]
            return normalized or None

        try:
            parsed = json.loads(response_text)
        except Exception:
            match = re.search(r"\{.*\}", response_text, flags=re.S)
            if not match:
                return None
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                return None
        return extract_steps(parsed)

    @classmethod
    def _normalize_pseudo_plan_text(cls, pseudo_plan: str, query_text: str) -> str:
        """Normalize pseudo-plan text, falling back to query if empty."""
        steps = cls.parse_pseudo_plan_steps(pseudo_plan)
        if steps:
            return "\n".join(steps)
        fallback = pseudo_plan.strip()
        return fallback if fallback else query_text

    @staticmethod
    def _format_available_tools(tools_info: Optional[List[Dict[str, Any]]]) -> str:
        """Format available tools as a brief list (name: description)."""
        if not tools_info:
            return ""
        lines: List[str] = []
        for item in tools_info:
            function = item.get("function") if isinstance(item, dict) else None
            if not isinstance(function, dict):
                continue
            name = str(function.get("name") or "").strip()
            if not name:
                continue
            description = str(function.get("description") or "").strip()
            if description:
                lines.append(f"- {name}: {description}")
            else:
                lines.append(f"- {name}")
        return "\n".join(lines)

    @staticmethod
    def _parse_selected_indices(response: str) -> List[int]:
        """Parse LLM response for selected skill indices."""
        try:
            data = json.loads(response)
            indices = data.get("selected_indices", [])
            return [int(i) for i in indices if isinstance(i, int) or str(i).isdigit()]
        except Exception:
            match = re.search(r"\{.*\}", response, flags=re.S)
            if match:
                try:
                    data = json.loads(match.group(0))
                    indices = data.get("selected_indices", [])
                    return [int(i) for i in indices if isinstance(i, int) or str(i).isdigit()]
                except Exception:
                    return []
        return []

    @staticmethod
    def _collect_skill_tool_names(skills: List[SkillDocument]) -> List[str]:
        """Collect unique tool names from selected skills."""
        names: List[str] = []
        seen = set()
        for skill in skills:
            for tool_name in skill.tools:
                normalized = str(tool_name).strip()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    names.append(normalized)
        return names

    @staticmethod
    def _extract_tool_name_from_info(item: Dict[str, Any]) -> Optional[str]:
        """Extract tool name from an OpenAI-format tool definition."""
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict):
            return None
        name = str(function.get("name") or "").strip()
        return name or None
