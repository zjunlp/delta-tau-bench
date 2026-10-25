"""Explorer action backend: side-effect free lookups over skills and tools.

Every method here maps an explorer action to an existing read-only interface
(``SkillContextRuntime`` / ``ToolRAGRuntime`` / the OpenAI tool schemas) and
compresses the result before returning it.

INVARIANT: no method in this module ever executes a real tool.
"""

from __future__ import annotations

import hashlib
import logging
import re
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from dtau_bench.explorer.types import ExplorerConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Addressable refs
#
# Every card the explorer sees carries a stable ``ref`` that read_skill and
# expand_graph resolve against; ``name`` is display only.
# ─────────────────────────────────────────────────────────────────────────────

REF_ATOMIC = "atomic"
REF_TOOL_SKILL = "tool_skill"
REF_PLANNING = "planning"
_REF_KINDS = (REF_ATOMIC, REF_TOOL_SKILL, REF_PLANNING)

# expand_graph edge directions (new names) and the legacy aliases.
EDGE_TOOLS = "tools"
EDGE_ATOMIC_SKILLS = "atomic_skills"
EDGE_PLANS = "plans"
_EDGE_ALIASES = {
    "tool": EDGE_TOOLS,
    "tools": EDGE_TOOLS,
    "atomic": EDGE_ATOMIC_SKILLS,
    "atomic_skills": EDGE_ATOMIC_SKILLS,
    "planning": EDGE_PLANS,
    "plans": EDGE_PLANS,
}
# Which ref kind each edge can start from.
_EDGE_SOURCE_KIND = {
    EDGE_TOOLS: REF_ATOMIC,
    EDGE_ATOMIC_SKILLS: REF_PLANNING,
    EDGE_PLANS: REF_ATOMIC,
}


def make_ref(kind: str, key: str) -> str:
    """Build a ``<kind>:<key>`` ref."""
    return f"{kind}:{key}"


def parse_ref(ref: str) -> Optional[Tuple[str, str]]:
    """Parse a ref into ``(kind, key)``; return None when malformed."""
    text = str(ref or "").strip()
    if ":" not in text:
        return None
    kind, _, key = text.partition(":")
    kind = kind.strip().lower()
    key = key.strip()
    if kind not in _REF_KINDS or not key:
        return None
    return kind, key


def plan_key(task: str) -> str:
    """Stable short key for a planning skill (its task text is not addressable)."""
    return hashlib.sha1(str(task or "").encode("utf-8")).hexdigest()[:8]


def normalize_edge(value: Any) -> Optional[str]:
    """Map a ``to`` / legacy ``direction`` value onto a canonical edge name."""
    text = str(value or "").strip().lower()
    if not text:
        return None
    return _EDGE_ALIASES.get(text)


_ATTR_PATTERN = re.compile(r"\[(?:\s*)((?:cost|latency)\s*=[^\]]*)\]\s*$")

# Lines worth keeping first when a skill body must be truncated.
_SALIENT_HINTS = (
    "step",
    "first",
    "then",
    "next",
    "finally",
    "require",
    "required",
    "must",
    "before",
    "after",
    "precondition",
    "tool",
    "param",
    "id",
    "confirm",
)


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def _first_sentence(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""
    match = re.search(r"(?<=[.!?])\s", text)
    if match and match.start() + 1 <= limit:
        text = text[: match.start() + 1]
    return _truncate(text, limit)


def _parse_tool_attributes(description: str) -> Dict[str, Any]:
    """Recover ``cost`` / ``latency`` injected into the description tail.

    ``ToolDefinition.to_openai_tool(inject_attributes=True)`` appends
    ``[cost=X, latency=Y]`` to the description (see tools/mcp_wrapper.py).
    """
    attributes: Dict[str, Any] = {}
    match = _ATTR_PATTERN.search(description or "")
    if not match:
        return attributes
    for chunk in match.group(1).split(","):
        if "=" not in chunk:
            continue
        key, _, value = chunk.partition("=")
        key = key.strip().lower()
        value = value.strip()
        if key not in {"cost", "latency"}:
            continue
        try:
            attributes[key] = int(value)
        except ValueError:
            try:
                attributes[key] = float(value)
            except ValueError:
                attributes[key] = value
    return attributes


def _strip_attributes(description: str) -> str:
    return _ATTR_PATTERN.sub("", description or "").strip()


def _as_tool_list(value: Any) -> List[str]:
    """Tool skill rows store ``tools`` as a stringified list in some libraries."""
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


class ExplorerBackend:
    """Read-only backend serving the explorer's action space."""

    def __init__(
        self,
        *,
        config: ExplorerConfig,
        all_tools_info: Sequence[Dict[str, Any]],
        skill_context_runtime: Any = None,
        tool_rag_runtime: Any = None,
        tool_to_domain: Optional[Dict[str, str]] = None,
    ):
        self.config = config
        self.skill_context_runtime = skill_context_runtime
        self.tool_rag_runtime = tool_rag_runtime
        self.tool_to_domain = tool_to_domain or {}

        self._tool_info_by_name: Dict[str, Dict[str, Any]] = {}
        for item in all_tools_info or []:
            name = self._tool_name(item)
            if name and name not in self._tool_info_by_name:
                self._tool_info_by_name[name] = item
        self._all_tools_info = list(all_tools_info or [])

        # Observability: what the explorer has looked at (NOT committed).
        self.explored_tool_names: List[str] = []
        self.explored_skill_names: List[str] = []
        # Tools whose full schema was actually inspected (verified candidates).
        self.inspected_tool_names: List[str] = []
        self.explored_skill_refs: List[str] = []

        # ── ref registry (P0) ──
        # Populated whenever a card is emitted, so read_skill / expand_graph can
        # resolve exactly what the explorer was shown.
        self._ref_meta: Dict[str, Dict[str, Any]] = {}
        self._planning_by_key: Dict[str, Any] = {}
        self._catalog_loaded = False
        self._atomic_documents: Dict[str, Any] = {}
        self._tool_skill_names: Set[str] = set()

    # ─────────────────────────────────────────────────────────────────────
    # Public helpers
    # ─────────────────────────────────────────────────────────────────────

    def tool_card(self, name: str, score: Optional[float] = None) -> Dict[str, Any]:
        """Public accessor for a compact tool card."""
        return self._tool_card(name, score)

    def fallback_candidates(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Best-effort candidates when the explorer fails to write a report.

        Prefers tools whose schema was actually inspected; only falls back to
        merely-retrieved tools if nothing was inspected. Kept deliberately small
        because these are unverified and would otherwise hurt tool-retrieval
        precision.
        """
        limit = max(int(limit), 0)
        names = list(self.inspected_tool_names)
        if not names:
            names = list(self.explored_tool_names)
        return [self._tool_card(name, None) for name in names[:limit]]

    @property
    def known_tool_names(self) -> Set[str]:
        """Whitelist used to reject hallucinated tool names in the report."""
        return set(self._tool_info_by_name.keys())

    @property
    def all_tools_info(self) -> List[Dict[str, Any]]:
        return self._all_tools_info

    # ─────────────────────────────────────────────────────────────────────
    # Skill catalog / ref registry (P0)
    # ─────────────────────────────────────────────────────────────────────

    def _load_catalog(self) -> None:
        """Index every addressable skill node once, for refs and suggestions."""
        if self._catalog_loaded:
            return
        self._catalog_loaded = True
        runtime = self.skill_context_runtime
        if runtime is None:
            return

        for attr in ("atomic_retriever", "functional_retriever"):
            retriever = getattr(runtime, attr, None)
            for document in getattr(retriever, "documents", None) or []:
                name = str(getattr(document, "name", "") or "")
                if name and name not in self._atomic_documents:
                    self._atomic_documents[name] = document

        planning = getattr(runtime, "planning_retriever", None)
        for document in getattr(planning, "documents", None) or []:
            task = str(getattr(document, "task", "") or "")
            if task:
                self._planning_by_key.setdefault(plan_key(task), document)

        graph_state = getattr(runtime, "graph_state", None)
        if graph_state is not None:
            self._tool_skill_names |= set(
                getattr(graph_state, "tool_skill_by_tool_name", {}) or {}
            )
        retriever = getattr(runtime, "tool_skill_retriever", None)
        for document in getattr(retriever, "documents", None) or []:
            name = str(getattr(document, "name", "") or "")
            if name:
                self._tool_skill_names.add(name)

    def describe(self) -> Dict[str, Any]:
        """Document counts per layer - logged at session start.

        An empty retriever must be visible rather than silently returning empty
        results (the old ``kind="functional"`` option was a dead end because its
        index held zero documents).
        """
        self._load_catalog()
        return {
            "planning": len(self._planning_by_key),
            "atomic": len(self._atomic_documents),
            "tool_skills": len(self._tool_skill_names),
            "tool_skill_index": getattr(
                self.skill_context_runtime, "tool_skill_retriever", None
            )
            is not None,
            "tools": len(self._tool_info_by_name),
        }

    def _register_ref(self, kind: str, key: str, name: str) -> str:
        ref = make_ref(kind, key)
        self._ref_meta[ref] = {"kind": kind, "key": key, "name": name}
        if ref not in self.explored_skill_refs:
            self.explored_skill_refs.append(ref)
        return ref

    def _known_refs(self) -> List[str]:
        self._load_catalog()
        refs = [make_ref(REF_ATOMIC, name) for name in self._atomic_documents]
        refs += [make_ref(REF_TOOL_SKILL, name) for name in sorted(self._tool_skill_names)]
        refs += [make_ref(REF_PLANNING, key) for key in self._planning_by_key]
        return refs

    def _did_you_mean(self, text: str, limit: int = 3) -> List[str]:
        """Fuzzy suggestions so a bad ref becomes a self-correctable error."""
        probe = str(text or "").strip()
        if not probe:
            return []
        matches = get_close_matches(probe, self._known_refs(), n=limit, cutoff=0.5)
        if matches:
            return matches
        bare = probe.split(":")[-1].strip()
        out: List[str] = []
        for hit in get_close_matches(
            bare,
            list(self._atomic_documents) + sorted(self._tool_skill_names),
            n=limit,
            cutoff=0.6,
        ):
            kind = REF_ATOMIC if hit in self._atomic_documents else REF_TOOL_SKILL
            out.append(make_ref(kind, hit))
        return out

    def _resolve(self, ref: str) -> Dict[str, Any]:
        """Resolve a ref (or, tolerantly, a bare name) to a skill node."""
        self._load_catalog()
        raw = str(ref or "").strip()
        parsed = parse_ref(raw)

        if parsed is not None:
            kind, key = parsed
            node = self._lookup(kind, key)
            if node is not None:
                return {"found": True, "kind": kind, "key": key,
                        "node": node, "resolved_by": "ref"}

        # Tolerant fallback: the model sometimes invents a ref or passes a name.
        bare = raw.split(":", 1)[-1].strip() if parsed is not None else raw
        for kind in (REF_ATOMIC, REF_TOOL_SKILL):
            node = self._lookup(kind, bare)
            if node is not None:
                return {"found": True, "kind": kind, "key": bare,
                        "node": node, "resolved_by": "fallback"}
        # Planning nodes may be referenced by their (truncated) task text.
        if len(bare) >= 12:
            probe = bare[:60]
            for key, document in self._planning_by_key.items():
                task = str(getattr(document, "task", "") or "")
                if task.startswith(probe) or probe in task:
                    return {"found": True, "kind": REF_PLANNING, "key": key,
                            "node": document, "resolved_by": "fallback"}
        return {"found": False, "kind": None, "key": None,
                "node": None, "resolved_by": None}

    def _lookup(self, kind: str, key: str) -> Any:
        if kind == REF_ATOMIC:
            return self._atomic_documents.get(key)
        if kind == REF_PLANNING:
            return self._planning_by_key.get(key)
        if kind == REF_TOOL_SKILL:
            return self._tool_skill_payload(key)
        return None

    def _tool_skill_payload(self, name: str) -> Optional[Dict[str, Any]]:
        """Tool skills live in the skill graph and (P1) in a dedicated index."""
        graph_state = getattr(self.skill_context_runtime, "graph_state", None)
        if graph_state is not None:
            payload = (getattr(graph_state, "tool_skill_by_tool_name", {}) or {}).get(name)
            if isinstance(payload, dict):
                return payload
        retriever = getattr(self.skill_context_runtime, "tool_skill_retriever", None)
        for document in getattr(retriever, "documents", None) or []:
            if str(getattr(document, "name", "")) == name:
                return {
                    "name": name,
                    "document": str(getattr(document, "document", "") or ""),
                    "content": str(getattr(document, "content", "") or ""),
                    "tools": _as_tool_list(getattr(document, "tools", None)) or [name],
                }
        return None

    def _unknown_ref_error(self, ref: str, action: str) -> Dict[str, Any]:
        """Fail loudly: an unresolvable ref must differ from 'node has no data'."""
        return {
            "error": f"unknown ref {str(ref or '')[:80]!r}",
            "node_found": False,
            "did_you_mean": self._did_you_mean(ref),
            "hint": (
                "ref must come from a search_skills / search_tool_skills / "
                f"expand_graph card. Call search_skills first, then {action}(ref=...)."
            ),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Action: search_skills
    # ─────────────────────────────────────────────────────────────────────

    def search_skills(
        self,
        query: str,
        kind: str = "any",
        top_k: int = 5,
        exclude_names: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return {"error": "query must be a non-empty string", "skills": []}
        if self.skill_context_runtime is None:
            return {"error": "skill library is not available", "skills": []}

        kind = (kind or "any").strip().lower()
        if kind == "functional":
            # Kept as an alias only: the functional sub-index holds 0 documents in
            # every library we ship, so advertising it wasted explorer actions.
            kind = "atomic"
        if kind not in {"any", "atomic", "planning"}:
            kind = "any"
        top_k = max(1, min(int(top_k or 1), int(self.config.search_skills_top_k)))
        exclude = {str(n) for n in (exclude_names or set())}

        try:
            embedding = self.skill_context_runtime.indexer.get_query_embedding(query)
        except Exception as exc:  # pragma: no cover - embedding endpoint failure
            logger.warning("Explorer search_skills embedding failed: %r", exc)
            return {"error": f"embedding failed: {exc}", "skills": []}

        cards: List[Dict[str, Any]] = []
        if kind in {"any", "planning"}:
            cards.extend(self._search_planning(embedding, top_k))
        if kind in {"any", "atomic"}:
            cards.extend(
                self._search_skill_docs(
                    self.skill_context_runtime.atomic_retriever, embedding, top_k, "atomic", exclude
                )
            )
        if kind in {"any", "functional"}:
            cards.extend(
                self._search_skill_docs(
                    self.skill_context_runtime.functional_retriever,
                    embedding,
                    top_k,
                    "functional",
                    exclude,
                )
            )

        cards.sort(key=lambda card: card.get("score", 0.0), reverse=True)
        cards = cards[:top_k] if kind != "any" else cards[: top_k * 2]

        for card in cards:
            name = str(card.get("name") or "")
            if name and name not in self.explored_skill_names:
                self.explored_skill_names.append(name)

        return {
            "query": query,
            "kind": kind,
            "skills": cards,
            "hint": (
                "Cards are summaries only. Use the card's `ref` (never its `name`) "
                "with read_skill(ref=...) for the full body, or "
                "expand_graph(ref=..., to=...) to walk the skill graph."
            ),
        }

    def _search_planning(self, embedding: Any, top_k: int) -> List[Dict[str, Any]]:
        retriever = getattr(self.skill_context_runtime, "planning_retriever", None)
        if retriever is None:
            return []
        self._load_catalog()
        cards = []
        for hit in retriever.retrieve(embedding, top_k):
            document = hit.document
            task = str(getattr(document, "task", "") or "")
            key = plan_key(task)
            self._planning_by_key.setdefault(key, document)
            cards.append({
                "ref": self._register_ref(REF_PLANNING, key, _truncate(task, 90)),
                "name": _truncate(task, 90),
                "kind": "planning",
                "summary": _first_sentence(
                    str(getattr(document, "plan", "") or ""), self.config.card_summary_chars
                ),
                "tools": [],
                "score": round(float(hit.score), 4),
            })
        return cards

    def _search_skill_docs(
        self,
        retriever: Any,
        embedding: Any,
        top_k: int,
        kind: str,
        exclude: Set[str],
    ) -> List[Dict[str, Any]]:
        if retriever is None:
            return []
        cards: List[Dict[str, Any]] = []
        for hit in retriever.retrieve(embedding, top_k + len(exclude)):
            document = hit.document
            name = str(getattr(document, "name", "") or "")
            if not name or name in exclude:
                continue
            cards.append(self._card_from_skill_document(document, kind, float(hit.score)))
            if len(cards) >= top_k:
                break
        return cards

    def _card_from_skill_document(
        self, document: Any, kind: str, score: float
    ) -> Dict[str, Any]:
        tools = _as_tool_list(getattr(document, "tools", None))
        name = str(getattr(document, "name", "") or "")
        ref_kind = REF_TOOL_SKILL if kind == REF_TOOL_SKILL else REF_ATOMIC
        card = {
            "ref": self._register_ref(ref_kind, name, name) if name else "",
            "name": name,
            "kind": str(getattr(document, "skill_type", "") or kind),
            "summary": _first_sentence(
                str(getattr(document, "document", "") or ""), self.config.card_summary_chars
            ),
            "tools": tools[:6],
            "tool_count": len(tools),
            "score": round(score, 4),
        }
        return card

    # ─────────────────────────────────────────────────────────────────────
    # Action: read_skill
    # ─────────────────────────────────────────────────────────────────────

    def read_skill(self, ref: str) -> Dict[str, Any]:
        """Read one skill body, addressed by its card ``ref``.

        Accepts a bare name as a tolerated fallback (reported via
        ``resolved_by``) because models occasionally invent refs.
        """
        raw = str(ref or "").strip()
        if not raw:
            return {"error": "ref must be a non-empty string", "node_found": False}
        if self.skill_context_runtime is None:
            return {"error": "skill library is not available", "node_found": False}

        resolved = self._resolve(raw)
        if not resolved["found"]:
            return self._unknown_ref_error(raw, "read_skill")

        kind = resolved["kind"]
        key = resolved["key"]
        node = resolved["node"]
        canonical = make_ref(kind, key)
        if key and key not in self.explored_skill_names:
            self.explored_skill_names.append(key)

        if kind == REF_PLANNING:
            task = str(getattr(node, "task", "") or "")
            body, truncated, omitted = self._compress_body(
                str(getattr(node, "plan", "") or "")
            )
            return {
                "ref": canonical,
                "kind": REF_PLANNING,
                "resolved_by": resolved["resolved_by"],
                "node_found": True,
                "name": _truncate(task, 120),
                "document": _truncate(task, 400),
                "content": body,
                "tools": [],
                "content_truncated": truncated,
                "content_chars_omitted": omitted,
                "hint": "Use expand_graph(ref=..., to='atomic_skills') for its steps.",
            }

        if kind == REF_ATOMIC:
            content = str(getattr(node, "content", "") or "")
            body, truncated, omitted = self._compress_body(content)
            return {
                "ref": canonical,
                "kind": str(getattr(node, "skill_type", "") or REF_ATOMIC),
                "resolved_by": resolved["resolved_by"],
                "node_found": True,
                "name": str(getattr(node, "name", key)),
                "document": _truncate(str(getattr(node, "document", "") or ""), 400),
                "content": body,
                "tools": _as_tool_list(getattr(node, "tools", None)),
                "content_truncated": truncated,
                "content_chars_omitted": omitted,
            }

        # tool skill
        body, truncated, omitted = self._compress_body(str(node.get("content") or ""))
        return {
            "ref": canonical,
            "kind": REF_TOOL_SKILL,
            "resolved_by": resolved["resolved_by"],
            "node_found": True,
            "name": key,
            "document": _truncate(str(node.get("document") or ""), 400),
            "content": body,
            "tools": _as_tool_list(node.get("tools")) or [key],
            "content_truncated": truncated,
            "content_chars_omitted": omitted,
        }

    def _compress_body(self, content: str) -> tuple[str, bool, int]:
        """L2 compression: keep the most action-relevant lines within budget."""
        content = (content or "").strip()
        limit = int(self.config.max_skill_chars)
        if limit <= 0 or len(content) <= limit:
            return content, False, 0

        lines = [line.rstrip() for line in content.splitlines()]
        kept: List[str] = []
        used = 0
        # Pass 1: salient lines (steps / requirements / tool references).
        salient_indices = {
            index
            for index, line in enumerate(lines)
            if any(hint in line.lower() for hint in _SALIENT_HINTS)
        }
        for index, line in enumerate(lines):
            if index not in salient_indices:
                continue
            if used + len(line) + 1 > limit:
                break
            kept.append(line)
            used += len(line) + 1
        # Pass 2: fill remaining budget with the leading lines for context.
        if used < limit:
            for line in lines:
                if line in kept:
                    continue
                if used + len(line) + 1 > limit:
                    break
                kept.append(line)
                used += len(line) + 1

        body = "\n".join(kept).strip() or _truncate(content, limit)
        omitted = max(len(content) - len(body), 0)
        return (
            body + f"\n...[truncated, {omitted} chars omitted]",
            True,
            omitted,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Action: expand_graph
    # ─────────────────────────────────────────────────────────────────────

    def expand_graph(
        self,
        ref: str,
        to: Optional[str] = None,
        limit: int = 8,
        direction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Walk one hop of the skill graph starting from ``ref``.

        Edges (``to``):
          - ``tools``          : atomic skill -> its tool skills
          - ``atomic_skills``  : planning skill -> its atomic skills
          - ``plans``          : atomic skill -> plans that use it

        ``direction`` is the legacy alias kept so an out-of-sync prompt cannot
        break the action. Unresolvable refs and incompatible edges return an
        explicit error (never an empty neighbor list), so the explorer can
        self-correct instead of concluding "this node has no neighbors".
        """
        raw = str(ref or "").strip()
        if not raw:
            return {"error": "ref must be a non-empty string", "node_found": False,
                    "neighbors": []}
        graph_state = getattr(self.skill_context_runtime, "graph_state", None)
        if graph_state is None:
            return {
                "error": "skill graph is not available for this library",
                "node_found": False,
                "neighbors": [],
            }
        if not graph_state.planning_to_atomic and not graph_state.atomic_to_tool:
            # A graph with nodes but no edges would otherwise look like "every
            # node happens to have no neighbors", which the explorer cannot
            # distinguish from a real dead end. Say so explicitly.
            return {
                "error": (
                    "skill graph for this library has no edges; expand_graph "
                    "cannot be used here"
                ),
                "node_found": False,
                "neighbors": [],
                "hint": "Use search_skills / search_tool_skills / search_tools instead.",
            }

        resolved = self._resolve(raw)
        if not resolved["found"]:
            payload = self._unknown_ref_error(raw, "expand_graph")
            payload["neighbors"] = []
            return payload

        kind = resolved["kind"]
        key = resolved["key"]
        canonical = make_ref(kind, key)
        limit = max(1, min(int(limit or 1), int(self.config.expand_graph_limit)))

        edge = normalize_edge(to) or normalize_edge(direction)
        if edge is None:
            # Default edge per layer, so a missing `to` is never an error.
            edge = EDGE_ATOMIC_SKILLS if kind == REF_PLANNING else EDGE_TOOLS

        expected_kind = _EDGE_SOURCE_KIND[edge]
        if kind != expected_kind:
            return {
                "error": (
                    f"edge to={edge!r} starts from a {expected_kind!r} node, "
                    f"but {canonical!r} is {kind!r}"
                ),
                "node_found": True,
                "ref": canonical,
                "neighbors": [],
                "valid_edges": self._valid_edges(kind),
                "hint": (
                    "atomic -> to='tools' | to='plans'; planning -> to='atomic_skills'."
                ),
            }

        neighbors: List[Dict[str, Any]] = []
        if edge == EDGE_TOOLS:
            for tool_name in sorted(graph_state.atomic_to_tool.get(key, set()))[:limit]:
                payload = self._tool_skill_payload(tool_name) or {}
                neighbors.append({
                    "ref": self._register_ref(REF_TOOL_SKILL, tool_name, tool_name),
                    "name": tool_name,
                    "kind": REF_TOOL_SKILL,
                    "summary": _first_sentence(
                        str(payload.get("document") or ""), self.config.card_summary_chars
                    ),
                    "tools": _as_tool_list(payload.get("tools")) or [tool_name],
                    "in_tool_catalog": tool_name in self._tool_info_by_name,
                })
        elif edge == EDGE_ATOMIC_SKILLS:
            task = str(getattr(resolved["node"], "task", "") or "")
            atomic_names = graph_state.planning_to_atomic.get(task)
            if atomic_names is None:
                for graph_key, values in graph_state.planning_to_atomic.items():
                    if task and (task[:60] in graph_key or graph_key[:60] in task):
                        atomic_names = values
                        break
            for atomic_name in sorted(atomic_names or set())[:limit]:
                document = self._atomic_documents.get(atomic_name)
                if document is not None:
                    neighbors.append(
                        self._card_from_skill_document(document, REF_ATOMIC, 0.0)
                    )
                else:
                    neighbors.append({
                        "ref": self._register_ref(REF_ATOMIC, atomic_name, atomic_name),
                        "name": atomic_name,
                        "kind": REF_ATOMIC,
                        "summary": "",
                    })
        else:  # EDGE_PLANS
            for planning_task, atomic_names in graph_state.planning_to_atomic.items():
                if key in atomic_names:
                    plan_hash = plan_key(planning_task)
                    neighbors.append({
                        "ref": self._register_ref(
                            REF_PLANNING, plan_hash, _truncate(planning_task, 90)
                        ),
                        "name": _truncate(planning_task, 90),
                        "kind": REF_PLANNING,
                        "summary": "",
                    })
                if len(neighbors) >= limit:
                    break

        for neighbor in neighbors:
            neighbor_name = str(neighbor.get("name") or "")
            if neighbor_name and neighbor_name not in self.explored_skill_names:
                self.explored_skill_names.append(neighbor_name)

        return {
            "ref": canonical,
            "to": edge,
            "node_found": True,
            "neighbors": neighbors,
            "neighbor_count": len(neighbors),
            "hint": (
                "" if neighbors else
                f"{canonical} has no {edge} edges in this library's skill graph."
            ),
        }

    @staticmethod
    def _valid_edges(kind: str) -> List[str]:
        return [edge for edge, source in _EDGE_SOURCE_KIND.items() if source == kind]

    # ─────────────────────────────────────────────────────────────────────
    # Skill hand-over to the main agent (P2)
    # ─────────────────────────────────────────────────────────────────────

    def resolve_committed_skill(self, ref: str) -> Optional[Dict[str, Any]]:
        """Validate one committed skill ref and render it for the main agent.

        Returns None for anything that cannot be resolved, which is how
        hallucinated refs are rejected (same policy as the tool whitelist).

        NOTE: the returned ``tools`` list is informational only. Committing a
        skill must NEVER auto-expose its tools: the predicted tool set feeds the
        retrieval metric, and a single skill can reference 3-6 tools.
        """
        resolved = self._resolve(ref)
        if not resolved["found"]:
            return None

        kind = resolved["kind"]
        key = resolved["key"]
        node = resolved["node"]

        if kind == REF_PLANNING:
            title = _truncate(str(getattr(node, "task", "") or ""), 110)
            body = str(getattr(node, "plan", "") or "")
            tools: List[str] = []
        elif kind == REF_ATOMIC:
            title = str(getattr(node, "name", key))
            body = str(getattr(node, "content", "") or "")
            tools = _as_tool_list(getattr(node, "tools", None))
        else:
            title = key
            body = str(node.get("content") or "")
            tools = _as_tool_list(node.get("tools")) or [key]

        compressed, truncated, omitted = self._compress_body(body)
        if not compressed.strip():
            # Nothing actionable to hand over.
            return None

        return {
            "ref": make_ref(kind, key),
            "kind": kind,
            "name": title,
            "body": compressed,
            "tools": tools,
            "truncated": truncated,
            "chars_omitted": omitted,
        }

    def render_committed_skills(self, entries: Sequence[Dict[str, Any]], budget: int) -> str:
        """Render committed skills as an actionable block within a char budget.

        The main agent does not "call" skills, so the hand-over is imperative
        guidance, never raw JSON.
        """
        budget = max(int(budget), 0)
        if not entries or budget <= 0:
            return ""
        header = "# Added skills (from capability investigation)"
        lines: List[str] = [header]
        used = len(header) + 1
        for entry in entries:
            tools = ", ".join(str(t) for t in (entry.get("tools") or [])[:6])
            suffix = " (tool skill)" if entry.get("kind") == REF_TOOL_SKILL else ""
            title = f"\n## {entry.get('name', '')}{suffix}"
            if tools:
                title += f"   [tools: {tools}]"
            body = str(entry.get("body") or "").strip()
            block = f"{title}\n{body}\n"
            if used + len(block) > budget:
                remaining = budget - used - len(title) - 2
                if remaining < 120:
                    break
                block = f"{title}\n{body[:remaining].rstrip()}...\n"
            lines.append(block)
            used += len(block)
        if len(lines) == 1:
            return ""
        return "".join([lines[0]] + lines[1:])

    # ─────────────────────────────────────────────────────────────────────
    # Action: search_tool_skills (P1)
    # ─────────────────────────────────────────────────────────────────────

    def search_tool_skills(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search tool skills: how to actually call a tool.

        Tool skills carry the parameter-level knowledge (required fields, enum
        values, preconditions, ordering) that plain tool schemas do not. They
        live in a dedicated index (``index_cache/tool_skills``) that only the
        explorer reads, so the main strategy's retrieval behaviour is untouched.
        """
        query = (query or "").strip()
        if not query:
            return {"error": "query must be a non-empty string", "skills": []}
        if self.skill_context_runtime is None:
            return {"error": "skill library is not available", "skills": []}

        retriever = getattr(self.skill_context_runtime, "tool_skill_retriever", None)
        if retriever is None or not (getattr(retriever, "documents", None) or []):
            return {
                "error": "tool-skill index not built for this library",
                "skills": [],
                "hint": (
                    "Run python -m dtau_bench build-skill-index to create "
                    "index_cache/tool_skills, or use inspect_tool for raw schemas."
                ),
            }

        top_k = max(1, min(int(top_k or 1), int(self.config.search_skills_top_k)))
        try:
            embedding = self.skill_context_runtime.indexer.get_query_embedding(query)
        except Exception as exc:  # pragma: no cover - embedding endpoint failure
            logger.warning("Explorer search_tool_skills embedding failed: %r", exc)
            return {"error": f"embedding failed: {exc}", "skills": []}

        cards: List[Dict[str, Any]] = []
        for hit in retriever.retrieve(embedding, top_k):
            document = hit.document
            name = str(getattr(document, "name", "") or "")
            if not name:
                continue
            card = self._card_from_skill_document(document, REF_TOOL_SKILL, float(hit.score))
            card["kind"] = REF_TOOL_SKILL
            card["in_tool_catalog"] = name in self._tool_info_by_name
            cards.append(card)
            if name not in self.explored_skill_names:
                self.explored_skill_names.append(name)

        return {
            "query": query,
            "skills": cards,
            "hint": (
                "Use read_skill(ref=...) for the full usage notes. `in_tool_catalog"
                "=false` means the tool was renamed or removed in this version."
            ),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Action: search_tools
    # ─────────────────────────────────────────────────────────────────────

    def search_tools(self, query: str, top_k: int = 8) -> Dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return {"error": "query must be a non-empty string", "tools": []}
        top_k = max(1, min(int(top_k or 1), int(self.config.search_tools_top_k)))

        tool_names: List[str] = []
        scores: Dict[str, float] = {}
        if self.tool_rag_runtime is not None:
            try:
                retrieval = self.tool_rag_runtime.retrieve_tools(
                    query_text=query,
                    all_tools_info=self._all_tools_info,
                    top_k=top_k,
                )
                tool_names = [str(n) for n in retrieval.get("tool_names", [])]
                for entry in (retrieval.get("trace") or {}).get("retrieved_tools", []) or []:
                    scores[str(entry.get("name"))] = float(entry.get("score", 0.0))
            except Exception as exc:  # pragma: no cover - retrieval endpoint failure
                logger.warning("Explorer search_tools retrieval failed: %r", exc)
                return {"error": f"tool retrieval failed: {exc}", "tools": []}
        else:
            tool_names = self._keyword_tool_match(query, top_k)

        cards = [self._tool_card(name, scores.get(name)) for name in tool_names]
        for name in tool_names:
            if name and name not in self.explored_tool_names:
                self.explored_tool_names.append(name)

        return {
            "query": query,
            "tools": cards,
            "hint": (
                "These tools are NOT yet available to the main agent. "
                "Call inspect_tool for parameter details, then list the ones you "
                "actually need in finish(commit_tools=[...])."
            ),
        }

    def _keyword_tool_match(self, query: str, top_k: int) -> List[str]:
        tokens = [token for token in re.split(r"\W+", query.lower()) if len(token) > 2]
        scored: List[tuple[int, str]] = []
        for name, item in self._tool_info_by_name.items():
            haystack = (name + " " + self._tool_description(item)).lower()
            hits = sum(1 for token in tokens if token in haystack)
            if hits:
                scored.append((hits, name))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [name for _, name in scored[:top_k]]

    def _tool_card(self, name: str, score: Optional[float]) -> Dict[str, Any]:
        item = self._tool_info_by_name.get(name)
        if item is None:
            return {"name": name, "available": False}
        description = self._tool_description(item)
        attributes = _parse_tool_attributes(description)
        card: Dict[str, Any] = {
            "name": name,
            "domain": self.tool_to_domain.get(name, ""),
            "summary": _first_sentence(
                _strip_attributes(description), self.config.card_summary_chars
            ),
            "cost": attributes.get("cost"),
            "latency": attributes.get("latency"),
            "require_confirmation": self._requires_confirmation(item),
        }
        if score is not None:
            card["score"] = round(score, 4)
        return card

    # ─────────────────────────────────────────────────────────────────────
    # Action: inspect_tool
    # ─────────────────────────────────────────────────────────────────────

    def inspect_tool(self, name: str) -> Dict[str, Any]:
        name = (name or "").strip()
        if not name:
            return {"error": "name must be a non-empty string"}
        item = self._tool_info_by_name.get(name)
        if item is None:
            return {
                "error": f"tool '{name}' does not exist in this version's catalog",
                "hint": "Use search_tools to find the correct name for this version.",
            }

        if name not in self.explored_tool_names:
            self.explored_tool_names.append(name)
        if name not in self.inspected_tool_names:
            self.inspected_tool_names.append(name)

        function = item.get("function") or {}
        description = str(function.get("description") or "")
        attributes = _parse_tool_attributes(description)
        parameters = function.get("parameters") or {}
        properties = parameters.get("properties") or {}
        required = [str(r) for r in (parameters.get("required") or [])]

        def describe(param_name: str, spec: Any) -> Dict[str, Any]:
            spec = spec if isinstance(spec, dict) else {}
            entry: Dict[str, Any] = {
                "name": param_name,
                "type": spec.get("type", "unknown"),
                "description": _truncate(str(spec.get("description") or ""), 160),
            }
            if spec.get("enum"):
                entry["enum"] = spec["enum"][:12]
            return entry

        return {
            "name": name,
            "domain": self.tool_to_domain.get(name, ""),
            "description": _truncate(_strip_attributes(description), 600),
            "required_params": [
                describe(key, properties.get(key)) for key in required
            ],
            "optional_params": [
                describe(key, spec)
                for key, spec in properties.items()
                if key not in set(required)
            ],
            "cost": attributes.get("cost"),
            "latency": attributes.get("latency"),
            "require_confirmation": "confirm" in required,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _tool_name(item: Dict[str, Any]) -> str:
        if not isinstance(item, dict):
            return ""
        function = item.get("function")
        if not isinstance(function, dict):
            return ""
        return str(function.get("name") or "").strip()

    @staticmethod
    def _tool_description(item: Dict[str, Any]) -> str:
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict):
            return ""
        return str(function.get("description") or "")

    @staticmethod
    def _requires_confirmation(item: Dict[str, Any]) -> bool:
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict):
            return False
        parameters = function.get("parameters") or {}
        required = parameters.get("required") or []
        return "confirm" in {str(r) for r in required}
