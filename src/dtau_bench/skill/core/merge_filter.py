from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Set

EmbeddingService = None
CompleteLinkageClustering = None
SkillMerger = None
prepare_skills_for_clustering = None
TwoStageFilterPipeline = None


def _prepare_skills_for_clustering_local(skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        if "return" in skill.get("content", ""):
            skill["content"] = skill["content"].split("return")[0].strip()
        prepared.append(prepared_item)
    return prepared


def group_by_domain(items: Iterable[Dict[str, Any]], metadata_key: str = "source_metadata") -> DefaultDict[str, List[Dict[str, Any]]]:
    grouped: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        metadata = item.get(metadata_key, {})
        if not isinstance(metadata, dict):
            raise ValueError("Expected metadata dict when grouping by domain")
        domain = str(metadata.get("domain", "")).strip()
        if not domain:
            raise ValueError(f"Missing metadata.domain in item: {item}")
        grouped[domain].append(item)
    return grouped


def merge_skill_sources(skills: List[Dict[str, Any]]) -> Dict[str, Any]:
    domains = sorted({
        str(skill.get("source_metadata", {}).get("domain", "")).strip()
        for skill in skills
        if isinstance(skill.get("source_metadata"), dict)
    } - {""})
    if len(domains) != 1:
        raise ValueError(f"Merged cluster must come from exactly one domain, got {domains}")

    task_ids: Set[str] = set()
    for skill in skills:
        single_task_id = str(skill.get("source_task_id", "")).strip()
        if single_task_id:
            task_ids.add(single_task_id)
        multi_task_ids = skill.get("source_task_ids", [])
        if isinstance(multi_task_ids, list):
            for task_id in multi_task_ids:
                task_id_str = str(task_id).strip()
                if task_id_str:
                    task_ids.add(task_id_str)

    metadata = dict(skills[0].get("source_metadata", {}))
    return {
        "domain": domains[0],
        "source_task_ids": sorted(task_ids),
        "source_count": len(skills),
        "metadata": metadata,
    }


def normalize_merged_results(merge_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for result in merge_results:
        if not isinstance(result, dict):
            continue

        if "merge_skills" not in result:
            normalized.append(result)
            continue

        payload = result["merge_skills"]
        merged_skills = payload if isinstance(payload, list) else [payload]
        source_info = merge_skill_sources(result.get("source", []))

        for skill in merged_skills:
            if not isinstance(skill, dict):
                raise ValueError(f"Merged skill payload must be a dict, got {type(skill).__name__}")
            normalized.append(
                {
                    "option": "add",
                    "skill": skill,
                    "source_metadata": source_info["metadata"],
                    "source_task_ids": source_info["source_task_ids"],
                    "merge_source_count": source_info["source_count"],
                }
            )

    return normalized

