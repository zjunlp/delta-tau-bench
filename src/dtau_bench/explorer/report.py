"""Report parsing, tool-name validation and hard truncation.

The explorer's free-form output is never trusted: every tool name it mentions is
checked against the real catalog before the report reaches the main agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

_SECTION_NAMES = ("GOAL", "TOOLS", "WORKFLOW", "PITFALLS", "RULED_OUT")

# Sections are emitted in this order but dropped in reverse priority order when
# the character budget is exceeded (TOOLS > PITFALLS > WORKFLOW > RULED_OUT).
_DISPLAY_ORDER = ("GOAL", "TOOLS", "WORKFLOW", "PITFALLS", "RULED_OUT")
_KEEP_PRIORITY = ("GOAL", "TOOLS", "PITFALLS", "WORKFLOW", "RULED_OUT")

_SECTION_HEADER = re.compile(
    r"^\s*(" + "|".join(_SECTION_NAMES) + r")\s*:\s*(.*)$", re.IGNORECASE
)
_ENTRY_NAME = re.compile(r"^\s*(?:[-*]|\d+[.)])?\s*([A-Za-z_][A-Za-z0-9_]*)\s*:")

_MAX_ENTRIES = {"TOOLS": None, "WORKFLOW": 5, "PITFALLS": 3, "RULED_OUT": 3}


@dataclass
class ReportValidation:
    """Result of validating and truncating an explorer report."""

    report_final: str
    accepted_tool_names: List[str] = field(default_factory=list)
    requested_tool_names: List[str] = field(default_factory=list)
    hallucinated_tool_names: List[str] = field(default_factory=list)
    chars_before_truncate: int = 0
    truncated: bool = False
    dropped_sections: List[str] = field(default_factory=list)


def parse_sections(text: str) -> Dict[str, List[str]]:
    """Split a report into``{SECTION: [lines]}`` (order preserved per section)."""
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for raw_line in (text or "").splitlines():
        match = _SECTION_HEADER.match(raw_line)
        if match:
            current = match.group(1).upper()
            sections.setdefault(current, [])
            inline = match.group(2).strip()
            if inline:
                sections[current].append(inline)
            continue
        if current is None:
            continue
        line = raw_line.strip()
        if line:
            sections[current].append(line)
    return sections


def extract_entry_name(line: str) -> str:
    """Extract the leading identifier of a ``- name: reason`` style entry."""
    match = _ENTRY_NAME.match(line or "")
    return match.group(1) if match else ""


def validate_report(
    *,
    report_raw: str,
    requested_tools: Sequence[str],
    known_tool_names: Set[str],
    max_report_chars: int,
    max_commit_tools: int,
    already_visible: Optional[Iterable[str]] = None,
) -> ReportValidation:
    """Validate tool names, drop hallucinations, then truncate to budget."""
    sections = parse_sections(report_raw)
    already_visible_set = {str(name) for name in (already_visible or [])}

    hallucinated: List[str] = []
    accepted: List[str] = []

    # ── 1. Filter TOOLS / RULED_OUT entries by the real catalog ──
    for section in ("TOOLS", "RULED_OUT"):
        kept_lines: List[str] = []
        for line in sections.get(section, []):
            name = extract_entry_name(line)
            if not name:
                # Free-form line without a tool name: keep as prose.
                kept_lines.append(line)
                continue
            if name not in known_tool_names:
                if name not in hallucinated:
                    hallucinated.append(name)
                continue
            kept_lines.append(line)
            if section == "TOOLS" and name not in accepted:
                accepted.append(name)
        if section in sections:
            sections[section] = kept_lines

    # ── 2. Reconcile with the explicit commit_tools argument ──
    requested_clean: List[str] = []
    for name in requested_tools or []:
        name = str(name).strip()
        if not name or name in requested_clean:
            continue
        requested_clean.append(name)
        if name not in known_tool_names:
            if name not in hallucinated:
                hallucinated.append(name)

    valid_requested = [name for name in requested_clean if name in known_tool_names]
    if valid_requested:
        # Union: commit_tools is authoritative, TOOLS section adds context.
        for name in valid_requested:
            if name not in accepted:
                accepted.append(name)
    accepted = [name for name in accepted if name not in already_visible_set]
    accepted = accepted[: max(int(max_commit_tools), 0)]

    # ── 3. Cap entries per section ──
    for section, cap in _MAX_ENTRIES.items():
        if cap is None or section not in sections:
            continue
        sections[section] = sections[section][:cap]
    if "TOOLS" in sections:
        sections["TOOLS"] = sections["TOOLS"][: max(int(max_commit_tools), 0)]

    # ── 4. Hard truncation by keep-priority ──
    budget = max(int(max_report_chars), 0)
    rendered_before = _render(sections, _DISPLAY_ORDER)
    chars_before = len(rendered_before)

    if budget <= 0 or chars_before <= budget:
        return ReportValidation(
            report_final=rendered_before,
            accepted_tool_names=accepted,
            requested_tool_names=requested_clean,
            hallucinated_tool_names=hallucinated,
            chars_before_truncate=chars_before,
            truncated=False,
        )

    selected: Dict[str, List[str]] = {}
    used = 0
    dropped: List[str] = []
    for section in _KEEP_PRIORITY:
        lines = sections.get(section)
        if not lines:
            continue
        header_cost = len(section) + 2
        if used + header_cost > budget:
            dropped.append(section)
            continue
        kept: List[str] = []
        used += header_cost
        for line in lines:
            cost = len(line) + 3
            if used + cost > budget:
                break
            kept.append(line)
            used += cost
        if kept:
            selected[section] = kept
        else:
            used -= header_cost
            dropped.append(section)

    return ReportValidation(
        report_final=_render(selected, _DISPLAY_ORDER),
        accepted_tool_names=accepted,
        requested_tool_names=requested_clean,
        hallucinated_tool_names=hallucinated,
        chars_before_truncate=chars_before,
        truncated=True,
        dropped_sections=dropped,
    )


def build_fallback_report(
    *,
    goal: str,
    tool_cards: Sequence[Dict[str, object]],
    max_report_chars: int,
) -> Tuple[str, List[str]]:
    """Programmatic report used when the explorer fails to produce one."""
    lines = [f"GOAL: {goal.strip()[:200] or '(not provided)'}", "TOOLS:"]
    names: List[str] = []
    for card in tool_cards:
        name = str(card.get("name") or "").strip()
        if not name:
            continue
        summary = str(card.get("summary") or "")[:120]
        cost = card.get("cost")
        latency = card.get("latency")
        suffix = ""
        if cost is not None or latency is not None:
            suffix = f" | cost={cost} latency={latency}"
        lines.append(f"  - {name}: {summary}{suffix}")
        names.append(name)
    lines.append(
        "PITFALLS:\n  - Explorer could not finish its own report; these tools are "
        "unverified candidates from retrieval only."
    )
    text = "\n".join(lines)
    if max_report_chars > 0 and len(text) > max_report_chars:
        text = text[: max_report_chars - 3].rstrip() + "..."
    return text, names


def _render(sections: Dict[str, List[str]], order: Sequence[str]) -> str:
    blocks: List[str] = []
    for section in order:
        lines = sections.get(section)
        if not lines:
            continue
        if section == "GOAL":
            blocks.append(f"GOAL: {' '.join(lines).strip()}")
            continue
        body = "\n".join(f"  {line.strip()}" for line in lines)
        blocks.append(f"{section}:\n{body}")
    return "\n".join(blocks).strip()
