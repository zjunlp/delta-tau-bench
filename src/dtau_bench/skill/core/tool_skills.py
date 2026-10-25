from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None


TOOL_SKILL_EXTRACTION_PROMPT = """You are constructing a reusable tool skill for an agent skill library.

Your task is to read the tool schema and repeated rollout evidence about a single tool-use failure pattern, then write ONE reusable tool skill that helps future agents avoid or recover from that failure.

# Goal
Produce a tool-centric skill that captures:
1. what the tool is for,
2. the important input/output details,
3. the specific failure pattern seen in the evidence,
4. the preconditions or parameter constraints needed to avoid that failure,
5. a generic recovery pattern.

# Constraints
- Output exactly one JSON object inside <skill>...</skill>.
- The JSON object must contain exactly these keys: `name`, `document`, `content`.
- `name` must be the target tool name, not a new abstract name.
- `document` should describe the tool, key parameters, outputs, and important notes from both schema and evidence.
- `content` should be reusable natural-language usage examples, not Python code.
- If the rollout evidence shows the tool is actually used in a way that differs from the schema, explicitly clarify that usage pattern in the skill instead of assuming the schema is the full story.
- Keep the skill generic and reusable; do not narrate a specific user story.

# Output format
<skill>
{
  "name": "tool_name",
  "document": "...",
  "content": "..."
}
</skill>
"""


TOOL_SKILL_MERGE_PROMPT = """You are merging multiple raw tool-skill facets for a single tool into ONE reusable final tool skill.

# Goal
Given:
- up to 15 raw tool-skill facets extracted from rollout evidence for the same tool

Produce exactly one final merged tool skill that:
1. accurately describes what the tool does,
2. highlights the recurring preconditions, parameter constraints, and failure modes,
3. preserves useful recovery patterns from the raw facets,
4. stays generic and reusable across tasks.

# Constraints
- Output exactly one JSON object inside <skill>...</skill>.
- The JSON object must contain exactly these keys: `name`, `document`, `content`.
- `name` must be the target tool name.
- `document` should be a concise but complete merged description of the tool's usage guidance.
- `content` should contain reusable natural-language usage patterns and cautions, not code.
- Merge overlapping advice; do not hallucinate or add any information not present in the given skills.

# Output format
<skill>
{
  "name": "tool_name",
  "document": "...",
  "content": "..."
}
</skill>
"""


def _extract_error_text(tool_result: Any) -> str:
    if isinstance(tool_result, dict):
        if isinstance(tool_result.get("error"), str) and tool_result.get("error").strip():
            return tool_result["error"].strip()
        if isinstance(tool_result.get("message"), str) and tool_result.get("message").strip():
            return tool_result["message"].strip()
    if isinstance(tool_result, str):
        return tool_result.strip()
    return json.dumps(tool_result, ensure_ascii=False, default=str).strip()


def _extract_tool_name_from_schema(schema: Dict[str, Any]) -> str:
    if not isinstance(schema, dict):
        return ""
    function = schema.get("function")
    if isinstance(function, dict):
        name = str(function.get("name", "")).strip()
        if name:
            return name
    name = str(schema.get("name", "")).strip()
    if name:
        return name
    return ""

def _find_later_success_examples(
    tool_calls: Sequence[Dict[str, Any]],
    tool_name: str,
    current_step_index: int,
    limit: int = 2,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        if str(call.get("tool_name", "")).strip() != tool_name:
            continue
        try:
            step_index = int(call.get("step_index", -1))
        except (TypeError, ValueError):
            step_index = -1
        if step_index <= current_step_index:
            continue
        if bool(call.get("has_error", False)):
            continue
        examples.append(call)
        if len(examples) >= limit:
            break
    return examples


@dataclass
class ToolSkillEvidenceGroup:
    tool_name: str
    evidence_count: int
    distinct_query_count: int
    query_ids: List[str]
    source_domains: List[str]
    schema: Dict[str, Any]
    evidence_items: List[Dict[str, Any]]

    def to_record(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "evidence_count": self.evidence_count,
            "distinct_query_count": self.distinct_query_count,
            "query_ids": list(self.query_ids),
            "source_domains": list(self.source_domains),
            "schema": self.schema,
            "evidence_items": self.evidence_items,
        }


def aggregate_tool_skill_evidence(
    records: Sequence[Dict[str, Any]],
    *,
    global_tool_schemas: Dict[str, Dict[str, Any]],
    min_distinct_tasks: int,
    min_error_events: int,
    max_examples_per_group: int = 6,
) -> List[ToolSkillEvidenceGroup]:
    grouped: Dict[str, Dict[str, Any]] = {}

    for record in records:
        if not isinstance(record, dict):
            continue
        domain = str(record.get("domain", "")).strip()
        if not domain:
            continue
        tool_calls = record.get("tool_calls") or []
        tool_errors = record.get("tool_errors") or []
        if not isinstance(tool_calls, list) or not isinstance(tool_errors, list):
            continue
        passed = record.get("passed")
        query_id = str(record.get("query_id", "")).strip()
        user_task = str(record.get("user_task", "")).strip()

        for event in tool_errors:
            if not isinstance(event, dict):
                continue
            tool_name = str(event.get("tool_name", "")).strip()
            if not tool_name or tool_name not in global_tool_schemas:
                continue
            try:
                current_step_index = int(event.get("step_index", -1))
            except (TypeError, ValueError):
                current_step_index = -1

            later_success_examples = _find_later_success_examples(
                tool_calls,
                tool_name=tool_name,
                current_step_index=current_step_index,
            )

            include_event = (passed is False) or bool(later_success_examples)
            if not include_event:
                continue

            error_text = _extract_error_text(event.get("tool_result"))
            if not error_text:
                continue
            key = tool_name
            bucket = grouped.setdefault(
                key,
                {
                    "tool_name": tool_name,
                    "query_ids": set(),
                    "source_domains": set(),
                    "evidence_items": [],
                    "schema": global_tool_schemas[tool_name],
                },
            )
            bucket["query_ids"].add(query_id)
            if domain:
                bucket["source_domains"].add(domain)

            failed_call = {
                "tool_name": tool_name,
                "arguments": event.get("arguments", {}),
                "tool_result": event.get("tool_result"),
                "error_text": error_text,
            }
            if "cost" in event:
                failed_call["cost"] = event["cost"]
            later_success_payload = []
            for example in later_success_examples:
                success_call = {
                    "tool_name": example.get("tool_name"),
                    "arguments": example.get("arguments", {}),
                    "tool_result": example.get("tool_result"),
                }
                if "cost" in example:
                    success_call["cost"] = example["cost"]
                later_success_payload.append(success_call)

            evidence_item = {
                "query_id": query_id,
                "domain": domain,
                "passed": passed,
                "user_task": user_task,
                "error_step_index": current_step_index,
                "failed_call": failed_call,
                "later_success_examples": later_success_payload,
            }
            if len(bucket["evidence_items"]) < max_examples_per_group:
                bucket["evidence_items"].append(evidence_item)

    groups: List[ToolSkillEvidenceGroup] = []
    for bucket in grouped.values():
        distinct_query_count = len(bucket["query_ids"])
        evidence_count = len(bucket["evidence_items"])
        if distinct_query_count < min_distinct_tasks and evidence_count < min_error_events:
            continue
        groups.append(
            ToolSkillEvidenceGroup(
                tool_name=bucket["tool_name"],
                evidence_count=evidence_count,
                distinct_query_count=distinct_query_count,
                query_ids=sorted(bucket["query_ids"]),
                source_domains=sorted(bucket["source_domains"]),
                schema=bucket["schema"],
                evidence_items=list(bucket["evidence_items"]),
            )
        )

    groups.sort(key=lambda item: (-item.distinct_query_count, -item.evidence_count, item.tool_name))
    return groups


class ToolSkillExtractor:
    def __init__(
        self,
        llm: Any,
        max_retries: int = 3,
        verbose: bool = True,
        cost_aware: bool = False,
    ):
        self.llm = llm
        self.max_retries = max_retries
        self.verbose = verbose
        self.cost_aware = cost_aware

    def _extract_skill(self, text: str) -> Optional[Dict[str, Any]]:
        match = re.search(r"<skill>(.*?)</skill>", text, flags=re.S)
        if not match:
            return None
        try:
            data = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        if not isinstance(data.get("name"), str) or not data["name"].strip():
            return None
        if not isinstance(data.get("document"), str):
            return None
        if not isinstance(data.get("content"), str):
            return None
        return data

    async def extract_group(self, group: ToolSkillEvidenceGroup) -> Optional[Dict[str, Any]]:
        schema_text = json.dumps(group.schema, ensure_ascii=False, indent=2)
        evidence_text = json.dumps(group.evidence_items, ensure_ascii=False, indent=2)
        user_message = (
            f"# Target tool name\n{group.tool_name}\n\n"
            f"# Tool schema\n{schema_text}\n\n"
            f"# Rollout evidence\n{evidence_text}\n"
        )
        response = await self.llm.ainvoke(
            messages=[
                ("system", self._system_prompt()),
                ("human", user_message),
            ],
            regex_extractor=self._extract_skill,
        )
        skill = self._extract_skill(response)
        if skill is None:
            return None
        skill_name = _extract_tool_name_from_schema(group.schema) or group.tool_name
        skill["name"] = skill_name
        skill["tools"] = [skill_name]
        return {
            "option": "add",
            "skill": skill,
            "source_metadata": {
                "tool_name": group.tool_name,
                "evidence_count": group.evidence_count,
                "distinct_query_count": group.distinct_query_count,
                "source_domains": group.source_domains,
            },
            "source_task_ids": group.query_ids,
        }

    def _system_prompt(self) -> str:
        if not self.cost_aware:
            return TOOL_SKILL_EXTRACTION_PROMPT
        from ..cost import COST_AWARE_GUIDANCE

        return TOOL_SKILL_EXTRACTION_PROMPT + COST_AWARE_GUIDANCE

    async def extract_groups(
        self,
        groups: Sequence[ToolSkillEvidenceGroup],
        batch_size: int,
        max_concurrent: int,
        show_progress: bool = False,
    ) -> List[Dict[str, Any]]:
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)
        results: List[Dict[str, Any]] = []

        async def run_one(group: ToolSkillEvidenceGroup) -> Optional[Dict[str, Any]]:
            async with semaphore:
                return await self.extract_group(group)

        tasks = [run_one(group) for group in groups]
        progress = None
        if show_progress and tqdm is not None:
            progress = tqdm(total=len(tasks), desc="ToolSkill Facet Extraction", unit="group")
        for start in range(0, len(tasks), batch_size):
            batch = tasks[start:start + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            for result in batch_results:
                if isinstance(result, Exception):
                    if self.verbose:
                        print(f"ToolSkillExtractor error: {result}")
                    continue
                if result:
                    results.append(result)
            if progress is not None:
                progress.update(len(batch))
        if progress is not None:
            progress.close()
        return results


def prepare_tool_skills_for_output(skills: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for item in skills:
        if "skill" not in item:
            continue
        skill = item["skill"]
        if not isinstance(skill, dict):
            continue
        prepared_item = dict(item)
        prepared_item["embedding_text"] = (
            f"{skill.get('name', '')}\n"
            f"{skill.get('document', '')}\n"
            f"{skill.get('content', '')}"
        )
        prepared.append(prepared_item)
    return prepared


class ToolSkillMerger:
    def __init__(
        self,
        llm: Any,
        verbose: bool = True,
        cost_aware: bool = False,
    ):
        self.llm = llm
        self.verbose = verbose
        self.cost_aware = cost_aware

    def _extract_skill(self, text: str) -> Optional[Dict[str, Any]]:
        match = re.search(r"<skill>(.*?)</skill>", text, flags=re.S)
        if not match:
            return None
        try:
            data = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        if not isinstance(data.get("name"), str) or not data["name"].strip():
            return None
        if not isinstance(data.get("document"), str):
            return None
        if not isinstance(data.get("content"), str):
            return None
        return data

    async def merge_tool(
        self,
        *,
        tool_name: str,
        selected_facets: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        facets_payload = [
            {
                "skill": facet.get("skill"),
                "source_metadata": facet.get("source_metadata", {}),
                "source_task_ids": facet.get("source_task_ids", []),
            }
            for facet in selected_facets
            if isinstance(facet, dict)
        ]
        user_message = (
            f"# Target tool name\n{tool_name}\n\n"
            f"# Raw tool-skill facets to merge\n"
            f"{json.dumps(facets_payload, ensure_ascii=False, indent=2)}\n"
        )
        response = await self.llm.ainvoke(
            messages=[
                ("system", self._system_prompt()),
                ("human", user_message),
            ],
            regex_extractor=self._extract_skill,
        )
        skill = self._extract_skill(response)
        if skill is None:
            return None
        skill["name"] = tool_name
        skill["tools"] = [tool_name]
        return skill

    def _system_prompt(self) -> str:
        if not self.cost_aware:
            return TOOL_SKILL_MERGE_PROMPT
        from ..cost import COST_AWARE_GUIDANCE

        return TOOL_SKILL_MERGE_PROMPT + COST_AWARE_GUIDANCE


def _normalize_single_tool_merge_result(
    merge_result: Dict[str, Any],
    tool_name: str,
    selected_facets: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if "merge_skills" in merge_result:
        payload = merge_result["merge_skills"]
        if isinstance(payload, list):
            if len(payload) != 1:
                raise ValueError(
                    f"Per-tool merge for '{tool_name}' must return exactly one merged skill, "
                    f"got {len(payload)}"
                )
            skill = payload[0]
        else:
            skill = payload
    else:
        skill = merge_result.get("skill", merge_result)

    if not isinstance(skill, dict):
        raise ValueError(f"Merged tool skill for '{tool_name}' must be a dict")

    source_domains = sorted({
        domain
        for facet in selected_facets
        for domain in facet.get("source_metadata", {}).get("source_domains", [])
        if isinstance(domain, str) and domain.strip()
    })
    source_task_ids = sorted({
        str(task_id).strip()
        for facet in selected_facets
        for task_id in facet.get("source_task_ids", [])
        if str(task_id).strip()
    })
    return {
        "option": "add",
        "skill": skill,
        "source_metadata": {
            "tool_name": tool_name,
            "source_domains": source_domains,
            "facet_count": len(selected_facets),
        },
        "source_task_ids": source_task_ids,
        "merge_source_count": len(selected_facets),
    }


def build_tool_merge_candidates(
    raw_facets: Sequence[Dict[str, Any]],
    *,
    max_facets_per_tool: int,
    random_seed: int = 0,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for facet in raw_facets:
        if not isinstance(facet, dict):
            continue
        metadata = facet.get("source_metadata", {})
        if not isinstance(metadata, dict):
            continue
        tool_name = str(metadata.get("tool_name", "")).strip()
        if not tool_name:
            continue
        grouped.setdefault(tool_name, []).append(facet)

    rng = random.Random(random_seed)
    candidates: List[Dict[str, Any]] = []
    for tool_name in sorted(grouped.keys()):
        facets = list(grouped[tool_name])
        selected = facets if len(facets) <= max_facets_per_tool else rng.sample(facets, max_facets_per_tool)
        source_domains = sorted({
            domain
            for facet in facets
            for domain in facet.get("source_metadata", {}).get("source_domains", [])
            if isinstance(domain, str) and domain.strip()
        })
        candidates.append(
            {
                "tool_name": tool_name,
                "total_facet_count": len(facets),
                "selected_facet_count": len(selected),
                "source_domains": source_domains,
                "facets": selected,
            }
        )
    return candidates


async def merge_tool_facets_per_tool(
    raw_facets: Sequence[Dict[str, Any]],
    *,
    llm: Any,
    global_tool_schemas: Dict[str, Dict[str, Any]],
    batch_size: int,
    max_concurrent: int,
    max_facets_per_tool: int,
    random_seed: int = 0,
    quiet: bool,
    show_progress: bool = False,
    cost_aware: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    import asyncio

    merge_candidates = build_tool_merge_candidates(
        raw_facets,
        max_facets_per_tool=max_facets_per_tool,
        random_seed=random_seed,
    )
    merger = ToolSkillMerger(
        llm=llm,
        verbose=not quiet,
        cost_aware=cost_aware,
    )

    merged: List[Dict[str, Any]] = []
    semaphore = asyncio.Semaphore(max_concurrent)
    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(total=len(merge_candidates), desc="ToolSkill Merge", unit="tool")

    async def process_candidate(candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        tool_name = candidate["tool_name"]
        selected_facets = candidate["facets"]
        if tool_name not in global_tool_schemas:
            return None
        async with semaphore:
            skill = await merger.merge_tool(
                tool_name=tool_name,
                selected_facets=selected_facets,
            )
        if not skill:
            return None
        return _normalize_single_tool_merge_result(
            {"skill": skill},
            tool_name=tool_name,
            selected_facets=selected_facets,
        )

    for start in range(0, len(merge_candidates), batch_size):
        batch = merge_candidates[start:start + batch_size]
        batch_tasks = [process_candidate(candidate) for candidate in batch]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        for item in batch_results:
            if isinstance(item, Exception):
                if not quiet:
                    print(f"ToolSkillMerger error: {item}")
            elif item is not None:
                merged.append(item)
        if progress is not None:
            progress.update(len(batch))
    if progress is not None:
        progress.close()
    return merge_candidates, merged
