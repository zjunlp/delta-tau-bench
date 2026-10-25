"""
Gold Replay Executor — executes gold trajectories using pre-generated step_param_codes.

"""

from __future__ import annotations

import json
import logging
import re as re_module
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DatabaseState:
    """
    Captured database state at a point in time.

    Contains entities and relationships tables, each mapping
    table_name -> list of record dicts.
    """

    entities: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    relationships: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.entities and not self.relationships

    @property
    def total_records(self) -> int:
        count = sum(len(v) for v in self.entities.values())
        count += sum(len(v) for v in self.relationships.values())
        return count

    def all_table_names(self) -> Set[str]:
        return set(self.entities.keys()) | set(self.relationships.keys())


@dataclass
class GoldReplayResult:
    """
    Result of gold trajectory replay.

    Attributes:
        success: True if all steps executed without critical error.
        initial_state: DB state before any execution.
        final_state: DB state after execution.
        steps: Per-step execution details.
        errors: List of error messages encountered.
    """

    success: bool = False
    initial_state: Optional[DatabaseState] = None
    final_state: Optional[DatabaseState] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def success_count(self) -> int:
        return sum(1 for s in self.steps if s.get("success", False))


# ─────────────────────────────────────────────────────────────────────────────
# Safe Code Execution
# ─────────────────────────────────────────────────────────────────────────────

# Safe builtins whitelist for param extraction code execution
_SAFE_BUILTINS = {
    # Types
    "str": str, "int": int, "float": float, "bool": bool,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    # Constants
    "True": True, "False": False, "None": None,
    # Common functions
    "len": len, "isinstance": isinstance, "type": type,
    "range": range, "enumerate": enumerate, "zip": zip,
    "min": min, "max": max, "sum": sum, "abs": abs,
    "any": any, "all": all, "next": next, "iter": iter,
    "filter": filter, "map": map,
    # String/collection methods
    "sorted": sorted, "reversed": reversed,
    "getattr": getattr, "hasattr": hasattr, "setattr": setattr,
    "print": print,
}


def _execute_param_code(code: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute pre-generated param extraction code in a sandboxed environment.

    The code is expected to:
    1. Read from `ctx` (entity context + accumulated step results)
    2. Set a `params` variable with the tool call arguments dict

    Args:
        code: Python code string that produces a `params` dict.
        ctx: Execution context containing entity_instances, value_domain_samples,
             and accumulated step_N_result values.

    Returns:
        Extracted parameters dict.

    Raises:
        ValueError: If code fails to produce a valid params dict.
    """
    local_vars: Dict[str, Any] = {"ctx": ctx, "json": json, "re": re_module}

    # Strip import statements (modules are pre-provided)
    clean_code = "\n".join(
        line for line in code.split("\n")
        if not line.strip().startswith("import ")
        and not line.strip().startswith("from ")
    )

    try:
        exec(clean_code, {"__builtins__": _SAFE_BUILTINS}, local_vars)

        if "params" in local_vars:
            params = local_vars["params"]
            if isinstance(params, dict):
                return params
            raise ValueError(f"params is {type(params).__name__}, expected dict")
        raise ValueError("Code did not produce a 'params' variable")

    except Exception as e:
        raise ValueError(f"Param code execution failed: {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# Database State Capture
# ─────────────────────────────────────────────────────────────────────────────


def capture_database_state(server: Any, session_id: str) -> DatabaseState:
    """
    Capture the current database state from an MCP server session.

    Inspects the session's database dict and classifies tables as
    entities vs relationships based on heuristics (entity tables have
    a self-referencing _id field).

    Args:
        server: MCPServerInstance or raw server object with session access.
        session_id: Session identifier.

    Returns:
        DatabaseState with entities and relationships populated.
    """
    try:
        # Get session - support both MCPServerInstance wrapper and raw server
        if hasattr(server, "get_session"):
            session = server.get_session(session_id)
        elif hasattr(server, "_get_session"):
            session = server._get_session(session_id)
        elif hasattr(server, "sessions"):
            session = server.sessions.get(session_id)
        else:
            logger.warning("Server has no session access method")
            return DatabaseState()

        if session is None:
            # Try to create session
            if hasattr(server, "create_session"):
                session = server.create_session(session_id)
            if session is None:
                return DatabaseState()

        # Get database dict from session
        db = (
            getattr(session, "database", None)
            or getattr(session, "databases", None)
            or getattr(session, "db_copy", None)
            or {}
        )

        entities: Dict[str, List[Dict[str, Any]]] = {}
        relationships: Dict[str, List[Dict[str, Any]]] = {}

        for key, value in db.items():
            if not isinstance(value, list):
                continue
            if not value or not isinstance(value[0], dict):
                continue

            # Heuristic: if any field is "{table_name_singular}_id", it's an entity table
            # e.g., table "Customer" has field "customer_id"
            is_entity = any(
                k.endswith("_id") and k.replace("_id", "").lower() == key.lower()
                for k in value[0].keys()
            )
            if is_entity:
                entities[key] = deepcopy(value)
            else:
                relationships[key] = deepcopy(value)

        return DatabaseState(entities=entities, relationships=relationships)

    except Exception as e:
        logger.warning(f"Failed to capture database state: {e}")
        return DatabaseState()


# ─────────────────────────────────────────────────────────────────────────────
# Gold Replay Executor
# ─────────────────────────────────────────────────────────────────────────────


class GoldReplayExecutor:
    """
    Executes gold trajectories using pre-generated step_param_codes.

    No LLM calls needed — codes are provided from task generation phase.

    For tools that require confirmation, ``confirm=True`` is automatically
    injected into the parameters during gold replay. Agent replay should
    **not** use this class — it uses
    ``AgentReplayExecutor`` which leaves confirmation up to the agent.

    Usage:
        from dtau_bench.tools.mcp_wrapper import setup_domain_tools, get_confirmation_tools
        server = setup_domain_tools(domain, "base", data_dir)
        confirm_tools = get_confirmation_tools(domain, data_dir)
        executor = GoldReplayExecutor(server, confirmation_tools=confirm_tools)
        result = executor.execute(gold_trajectory, step_param_codes, entity_context)
    """

    def __init__(
        self,
        server: Any,
        confirmation_tools: Optional[Set[str]] = None,
    ):
        """
        Args:
            server: MCPServerInstance (from mcp_wrapper.setup_domain_tools)
                    or any object with an invoke(session_id, tool_name, **kwargs) method.
            confirmation_tools: Set of tool names that require ``confirm=True``.
                    When provided, the executor automatically injects the flag
                    before invocation so that gold trajectories pass the
                    confirmation gate without the param-code having to include it.
        """
        self.server = server
        self._confirmation_tools: Set[str] = confirmation_tools or set()

    def execute(
        self,
        trajectory: List[str],
        step_param_codes: List[str],
        entity_context: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> GoldReplayResult:
        """
        Execute gold trajectory step by step.

        Args:
            trajectory: List of tool function names in order.
            step_param_codes: List of Python code strings, one per step.
            entity_context: Initial context dict with entity_instances and
                          value_domain_samples.
            session_id: Optional session ID (auto-generated if not provided).

        Returns:
            GoldReplayResult with initial/final DB states and step details.
        """
        session_id = session_id or f"gold_{uuid.uuid4().hex[:8]}"

        if len(step_param_codes) != len(trajectory):
            logger.warning(
                f"Code count mismatch: {len(step_param_codes)} codes for "
                f"{len(trajectory)} trajectory steps"
            )

        # Build execution context from entity_context
        ctx = dict(entity_context)

        # Capture initial state
        initial_state = capture_database_state(self.server, session_id)

        steps: List[Dict[str, Any]] = []
        errors: List[str] = []

        for step_idx, function_name in enumerate(trajectory):
            step_info: Dict[str, Any] = {
                "step_index": step_idx,
                "tool_name": function_name,
                "success": False,
            }

            try:
                # Get param code for this step
                if step_idx >= len(step_param_codes):
                    raise ValueError(f"No param code for step {step_idx}")

                param_code = step_param_codes[step_idx]

                # Execute param code to get arguments
                params = _execute_param_code(param_code, ctx)

                # Inject confirm=True for tools requiring confirmation
                # (gold trajectories are trusted — the confirmation gate
                #  should always be satisfied automatically)
                if self._confirmation_tools and function_name in self._confirmation_tools:
                    params["confirm"] = True

                step_info["arguments"] = params

                # Execute the tool
                result = self._invoke_tool(session_id, function_name, params)
                step_info["result"] = result

                # Check for errors in result
                is_error = isinstance(result, dict) and (
                    result.get("error") or result.get("success") is False
                )

                if is_error:
                    error_msg = result.get("error") or result.get("message", "Unknown")
                    step_info["error"] = str(error_msg)
                    errors.append(f"Step {step_idx} ({function_name}): tool returned error: {error_msg}")
                    logger.warning(f"Step {step_idx} ({function_name}): {error_msg}")
                else:
                    step_info["success"] = True

                # Update context with result for subsequent steps
                ctx[f"step_{step_idx}_result"] = result

            except Exception as e:
                error_msg = f"Step {step_idx} ({function_name}): {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                step_info["error"] = str(e)

            steps.append(step_info)

        # Capture final state
        final_state = capture_database_state(self.server, session_id)

        all_success = len(errors) == 0 and all(s.get("success") for s in steps)

        return GoldReplayResult(
            success=all_success,
            initial_state=initial_state,
            final_state=final_state,
            steps=steps,
            errors=errors,
        )

    def _invoke_tool(
        self, session_id: str, tool_name: str, params: Dict[str, Any]
    ) -> Any:
        """
        Invoke a tool on the MCP server.

        Handles both MCPServerInstance (with invoke method taking session_id)
        and raw server objects.
        """
        # MCPServerInstance has invoke(session_id, tool_name, **kwargs)
        if hasattr(self.server, "invoke"):
            return self.server.invoke(session_id=session_id, tool_name=tool_name, **params)

        # Raw server object (fallback)
        if hasattr(self.server, "server"):
            return self.server.server.invoke(session_id, tool_name, **params)

        raise RuntimeError(f"Server does not support invoke: {type(self.server)}")
