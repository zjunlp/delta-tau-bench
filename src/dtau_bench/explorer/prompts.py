"""Prompts for the explorer subagent."""

from __future__ import annotations

from typing import List

from dtau_bench.explorer.types import ExploreRequest, ExplorerConfig


REPORT_TEMPLATE = """GOAL: <one sentence>
TOOLS:
  - <tool_name>: <why it is needed> | required: <comma separated required params> | cost=<c> latency=<l>
WORKFLOW:
  1. <one line per step>
PITFALLS:
  - <one line per pitfall>
RULED_OUT:
  - <tool_name>: <why it does not work>"""


def build_system_prompt(config: ExplorerConfig) -> str:
    """System prompt for the explorer subagent."""
    return f"""You are a capability explorer subagent. A main agent is solving a
multi-turn task with tools, ran out of usable tools, and delegated a focused
investigation to you.

You can search a reusable skill library and the full cross-domain tool catalog,
read skill bodies, walk the skill graph, and inspect tool JSON schemas.
You CANNOT execute any real tool and you never talk to the user. Your only
deliverable is a short report plus the list of tools the main agent should be
given.

Rules:
1. Be economical. You have at most {config.max_actions} action calls in total.
   Start broad (search_tools / search_skills), then verify with inspect_tool.
   You MAY request several independent actions in the same turn (e.g. inspect
   three candidate tools at once) — doing so is cheaper and strongly preferred
   over one action per turn.
2. ALWAYS verify a tool with inspect_tool before recommending it. Never invent a
   tool name or a parameter name: only names returned by the backend exist.
3. Tools the main agent can already see are listed in the request. Do not
   recommend them again unless a parameter detail was the blocker.
4. Tool names change across catalog versions. If a skill references a tool that
   no longer exists, search for a semantically equivalent replacement and say so
   explicitly.
5. Prefer tools with lower cost / latency when several options are equivalent.
6. Call finish as soon as you can answer the request — do not keep exploring
   just because you still have budget left.

Addressing skills (important):
7. Every skill card carries a `ref` such as `atomic:verify_engineer_credentials`,
   `tool_skill:record_reading`, or `planning:1a2b3c4d`. read_skill and
   expand_graph accept ONLY that `ref`. Never pass a display name, a tool name,
   or a sentence copied from a card — it will not resolve.
8. If a lookup fails you get `error` plus `did_you_mean`. Use one of those
   suggestions instead of retrying the same string. An empty `neighbors` list
   with `node_found: true` means the node genuinely has no such edges — move on.
9. When you need to know HOW to call a tool (required parameters, valid enum
   values, preconditions, ordering), use search_tool_skills. inspect_tool gives
   you the schema; tool skills give you the learned usage rules.

Reporting:
10. Compress aggressively: the report goes into the main agent's context, so it
    must be short and actionable. Distill skills into imperative steps rather
    than pasting their bodies.
11. If a skill body contains parameter-level or ordering detail that would be
    lost by paraphrasing, put its `ref` in `commit_skills` (max
    {config.max_commit_skills}) — the main agent will receive that procedure
    verbatim. Use it sparingly; it costs the main agent context.
12. `commit_skills` does NOT give the main agent the skill's tools. Any tool you
    want it to use must be inspected and listed in `commit_tools`.

finish(report=...) must follow this exact layout (omit a section if empty):

{REPORT_TEMPLATE}

Hard limits: at most {config.max_commit_tools} tools in TOOLS, at most 5 WORKFLOW
steps, at most 3 PITFALLS, at most 3 RULED_OUT entries, and at most
{config.max_report_chars} characters overall."""


def build_request_prompt(request: ExploreRequest, config: ExplorerConfig) -> str:
    """User message describing the delegated investigation.

    Only session state that the main agent itself can observe is included.
    The simulated user's private``user_system_prompt`` is never passed here.
    """
    blocks: List[str] = [
        "# Investigation request",
        f"GOAL: {request.goal.strip() or '(not provided)'}",
        f"MISSING CAPABILITY: {request.missing_capability.strip() or '(not provided)'}",
    ]
    if request.known_failures.strip():
        blocks.append(f"ALREADY TRIED (reported by main agent): {request.known_failures.strip()}")

    if request.recent_user_messages:
        lines = "\n".join(f"- {text}" for text in request.recent_user_messages)
        blocks.append(
            "# Recent user messages (verbatim constraints — treat as authoritative)\n" + lines
        )

    if request.recent_tool_errors:
        lines = "\n".join(
            f"- {entry.get('tool', '?')}: {entry.get('error', '')}"
            for entry in request.recent_tool_errors
        )
        blocks.append("# Recent failed tool calls\n" + lines)

    if config.include_pseudo_plan and request.pseudo_plan.strip():
        blocks.append("# Pseudo plan the main agent started from\n" + request.pseudo_plan.strip())

    if request.visible_tool_names:
        blocks.append(
            "# Tools the main agent ALREADY has (do not re-commit these)\n"
            + ", ".join(request.visible_tool_names)
        )

    if request.injected_skill_names:
        blocks.append(
            "# Skills already shown to the main agent (already in its context; "
            "read them only if you need a detail)\n"
            + ", ".join(request.injected_skill_names)
        )

    blocks.append(
        "Investigate now. Use the actions, then call finish with a compact report "
        "and the exact tool names to hand over."
    )
    return "\n\n".join(blocks)


def build_budget_warning(actions_left: int) -> str:
    return (
        f"Budget notice: {actions_left} action call(s) left. "
        "Wrap up and prepare to call finish."
    )


FORCE_FINISH_MESSAGE = (
    "Budget exhausted. Call finish now with whatever you have established. "
    "Only include tool names you actually verified."
)
