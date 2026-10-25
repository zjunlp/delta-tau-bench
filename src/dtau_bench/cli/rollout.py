#!/usr/bin/env python3
"""
Run one rollout: every task of one split at one catalog version.

Domains are processed in parallel. Each task is saved as a verified independent
record and aggregated into experiments/{run_name}/rollouts/{domain}.jsonl, so an
interrupted run can repair the aggregate before resuming.

Usage:
    python -m dtau_bench rollout \\
      --run-name my_run_v1_test \\
      --strategy all_tools_iterative_rag \\
      --split test --version v1 \\
      --agent-model DeepSeek-V3.2 --user-model DeepSeek-V3.2

    python -m dtau_bench rollout --list --split test --version v1
"""

import argparse
import hashlib
import json
import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────────────────────
# Path setup: add src/ to sys.path so dtau_bench is importable
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Imports (after path setup)
# ─────────────────────────────────────────────────────────────────────────────

from dtau_bench.utils.config import (
    extract_model_request_options,
    load_config,
    load_models_config,
    save_config_snapshot,
    get_data_root,
    get_project_root,
    resolve_path,
)
from dtau_bench.utils.logging import setup_logging, get_logger
from dtau_bench.tools.mcp_wrapper import (
    setup_domain_tools,
    setup_all_tools,
    get_all_domains,
)
from dtau_bench.tools.tool_router import ToolRouter
from dtau_bench.tools.data_loader import (
    load_tasks,
    load_domain_tasks,
    discover_domains,
    TaskData,
)
from dtau_bench.rollout.agent import Agent, AgentConfig
from dtau_bench.rollout.user import SimulatedUser
from dtau_bench.rollout.pipeline import Pipeline, PipelineConfig, PipelineMode, TrajectoryResult
from dtau_bench.rollout.retry import RetryConfig, compute_retry_delay, is_transient_error
from dtau_bench.rollout.record_store import (
    clear_domain_records,
    load_domain_records,
    rebuild_domain_jsonl,
    recover_domain_records,
    remove_record_checkpoints,
    save_record,
)
from dtau_bench.strategy.registry import (
    build_strategy,
    list_strategies,
    is_registered,
    uses_all_tools,
    uses_skill_library,
    uses_explorer,
)


logger = get_logger("scripts.run_rollout")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_AGENT_SYSTEM_PROMPT = """⚠️ **Do not autonomously infer or generate parameters for tool calls. When a tool usage is identified, you must first ask the User to provide or confirm the specific values for the required arguments.**
**You may only determine parameters based on the User's overall goals if (and only if) the User explicitly states they are unsure, asks you to decide, or declines to provide specific details.**

**Efficiency Optimization**: Each tool has an associated cost and latency score shown as [cost: X, latency: Y] at the end of the tool description. Lower scores indicate better efficiency. When multiple tools can accomplish the same task, prefer the one with lower cost and latency scores to optimize overall system performance."""


def select_model_endpoint(model_config: Dict[str, Any], query_id: str) -> str:
    """Pick one configured endpoint per task so multi-turn calls retain affinity."""
    endpoints = model_config.get("base_urls")
    if endpoints is None:
        return str(model_config.get("base_url", "http://localhost:8000/v1"))
    if not isinstance(endpoints, list) or not endpoints or not all(isinstance(url, str) and url for url in endpoints):
        raise ValueError("model base_urls must be a non-empty list of endpoint URLs")
    digest = hashlib.sha256(query_id.encode("utf-8")).digest()
    return endpoints[int.from_bytes(digest[:8], "big") % len(endpoints)]


# ─────────────────────────────────────────────────────────────────────────────
# Per-Domain Processing
# ─────────────────────────────────────────────────────────────────────────────

def process_domain(
    domain: str,
    tasks: List[TaskData],
    version: str,
    run_config: Dict[str, Any],
    output_dir: Path,
    data_dir: Path,
    resume: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Process all tasks for a single domain.
    
    Creates Agent, User, Pipeline, and ToolRouter for this domain,
    then iterates through tasks sequentially writing results to JSONL.
    
    Args:
        domain: Domain name.
        tasks: List of TaskData for this domain.
        version: Tool version string.
        run_config: Merged configuration dict.
        output_dir: Experiment output directory (contains rollouts/).
        data_dir: Root data directory.
        resume: Whether to skip already-completed samples.
        verbose: Print per-sample progress.
    
    Returns:
        Dict with processing stats: {domain, total, completed, skipped, failed, elapsed}.
    """
    start_time = time.time()
    stats = {
        "domain": domain,
        "total": len(tasks),
        "completed": 0,
        "skipped": 0,
        "failed": 0,
        "retried_failed_records": 0,
        "sample_retries": 0,
    }
    all_task_ids = {task.query_id for task in tasks}
    if len(all_task_ids) != len(tasks):
        raise ValueError(f"[{domain}] Duplicate task IDs in rollout input")
    
    # ── Output path ──
    rollout_dir = output_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    output_path = rollout_dir / f"{domain}.jsonl"
    
    # ── Resume or fresh start ──
    if resume:
        recovery = recover_domain_records(
            output_dir,
            domain,
            all_task_ids,
            expected_version=version,
        )
        if recovery["malformed_lines"]:
            logger.warning(
                "[%s] Recovered JSONL after dropping malformed lines: %s",
                domain,
                recovery["malformed_lines"],
            )
        if recovery["restored_from_checkpoints"]:
            logger.warning(
                "[%s] Restored %s record(s) missing from JSONL",
                domain,
                len(recovery["restored_from_checkpoints"]),
            )
        if recovery["checkpoint_conflicts"]:
            logger.warning(
                "[%s] Preferred %s verified checkpoint(s) over conflicting JSONL rows",
                domain,
                len(recovery["checkpoint_conflicts"]),
            )

        retry_failed = bool(run_config.get("retry_failed", False))
        if retry_failed:
            recovered_records = load_domain_records(
                output_dir,
                domain,
                all_task_ids,
                expected_version=version,
            )
            retry_ids = {
                query_id
                for query_id, record in recovered_records.items()
                if not record.get("success", False)
                and is_transient_error(record.get("error"))
            }
            stats["retried_failed_records"] = len(retry_ids)
            if retry_ids:
                remove_record_checkpoints(
                    output_dir,
                    domain,
                    retry_ids,
                    expected_ids=all_task_ids,
                    expected_version=version,
                )
            if verbose and stats["retried_failed_records"]:
                print(
                    f"  [{domain}] Removed {stats['retried_failed_records']} transient "
                    f"failure records for retry"
                )
        completed_ids = set(
            load_domain_records(
                output_dir,
                domain,
                all_task_ids,
                expected_version=version,
            )
        )
        if completed_ids:
            original_count = len(tasks)
            tasks = [t for t in tasks if t.query_id not in completed_ids]
            stats["skipped"] = original_count - len(tasks)
            if verbose:
                print(f"  [{domain}] Resuming: {stats['skipped']} done, {len(tasks)} remaining")
    else:
        # Fresh run: clear both the public JSONL and its durable checkpoints.
        had_output = output_path.exists() or (
            output_dir / "record_checkpoints" / domain
        ).exists()
        clear_domain_records(output_dir, domain)
        if verbose and had_output:
            print(f"  [{domain}] Cleared old output and record checkpoints")
    
    if not tasks:
        final = rebuild_domain_jsonl(
            output_dir,
            domain,
            all_task_ids,
            expected_version=version,
        )
        if final["record_ids"] != all_task_ids:
            raise RuntimeError(
                f"[{domain}] Resume record set mismatch: "
                f"expected {len(all_task_ids)}, got {len(final['record_ids'])}"
            )
        if verbose:
            print(f"  [{domain}] All tasks already completed ✓")
        stats["elapsed"] = time.time() - start_time
        return stats
    
    # ── Extract config values ──
    rollout_cfg = run_config.get("rollout", {})
    parallel_cfg = run_config.get("parallel", {})
    strategy_name = run_config.get("strategy", "single_domain_full")
    
    max_turns = rollout_cfg.get("max_turns", 20)
    max_steps_per_turn = rollout_cfg.get("max_steps_per_turn", 10)
    agent_temperature = rollout_cfg.get("agent_temperature", 0.7)
    user_temperature = rollout_cfg.get("user_temperature", 1.0)
    agent_timeout = rollout_cfg.get("agent_timeout", 180)
    user_timeout = rollout_cfg.get("user_timeout", 120)
    llm_max_retries = rollout_cfg.get("llm_max_retries", 10)
    llm_retry_initial_delay = rollout_cfg.get("llm_retry_initial_delay", 2)
    llm_retry_max_delay = rollout_cfg.get("llm_retry_max_delay", 60)
    llm_retry_total_timeout = rollout_cfg.get("llm_retry_total_timeout", 1200)
    llm_retry_jitter = rollout_cfg.get("llm_retry_jitter", True)
    sample_max_retries = max(1, int(rollout_cfg.get("sample_max_retries", 3)))
    sample_retry_config = RetryConfig(
        max_attempts=sample_max_retries,
        initial_delay=float(rollout_cfg.get("sample_retry_initial_delay", 20)),
        max_delay=float(rollout_cfg.get("sample_retry_max_delay", 300)),
        total_timeout=None,
        jitter=True,
    )
    enable_thinking = run_config.get("enable_thinking", False)
    no_thinking = run_config.get("no_thinking", False)
    
    # Model config
    models_cfg = run_config.get("_models_config", {}).get("models", {})
    agent_model_key = run_config.get("agent_model", "deepseek-v4-pro")
    user_model_key = run_config.get("user_model", "deepseek-flash")
    agent_model_cfg = models_cfg.get(agent_model_key, {})
    user_model_cfg = models_cfg.get(user_model_key, {})
    strategy_request_options = extract_model_request_options(agent_model_cfg)
    agent_request_options = {}
    if agent_model_cfg.get("reasoning_effort") is not None:
        agent_request_options["reasoning_effort"] = agent_model_cfg["reasoning_effort"]
    user_request_options = {
        name: user_model_cfg[name]
        for name in ("reasoning_effort", "max_completion_tokens", "top_p")
        if user_model_cfg.get(name) is not None
    }
    
    # ── Setup tools ──
    all_tools_mode = uses_all_tools(strategy_name)
    
    try:
        if all_tools_mode:
            # All-tools mode: load all domains' tools
            servers, tool_to_domain = setup_all_tools(
                version=version,
                data_dir=data_dir,
                inject_attributes=True,
            )
            tool_router = ToolRouter.from_all_domains(
                servers=servers,
                tool_to_domain=tool_to_domain,
                current_domain=domain,
                enforce_domain=False,  # Strategy handles tool filtering
            )
        else:
            # Single-domain mode: load only current domain's tools
            server = setup_domain_tools(
                domain=domain,
                version=version,
                data_dir=data_dir,
                inject_attributes=True,
            )
            tool_router = ToolRouter.from_single_domain(server)
    except Exception as e:
        logger.error(f"[{domain}] Failed to setup tools: {e}")
        if verbose:
            print(f"  [{domain}] ✗ Tool setup failed: {e}")
        raise RuntimeError(f"[{domain}] Tool setup failed") from e
    
    # ── Policy / system prompt ──
    policy_path = data_dir / "policies" / f"{domain}.md"
    agent_system_prompt = DEFAULT_AGENT_SYSTEM_PROMPT
    if policy_path.exists():
        agent_system_prompt = agent_system_prompt + "\n\n" + policy_path.read_text(encoding="utf-8")
        logger.info(f"[{domain}] Loaded policy file: {policy_path.name}")
    
    # ── Create Agent ──
    agent_model_str = f"openai/{agent_model_cfg.get('model_name', agent_model_key)}"
    agent = Agent(
        model=agent_model_str,
        system_prompt=agent_system_prompt,
        temperature=agent_temperature,
        api_base=agent_model_cfg.get("base_url", "http://localhost:8000/v1"),
        api_key=agent_model_cfg.get("api_key", "EMPTY"),
        max_tokens=agent_model_cfg.get("max_tokens", 4096),
        timeout=agent_timeout,
        max_retries=llm_max_retries,
        retry_initial_delay=llm_retry_initial_delay,
        retry_max_delay=llm_retry_max_delay,
        retry_total_timeout=llm_retry_total_timeout,
        retry_jitter=llm_retry_jitter,
        enable_thinking=enable_thinking and not no_thinking,
        **agent_request_options,
    )
    
    # ── Create SimulatedUser ──
    user_model_str = f"openai/{user_model_cfg.get('model_name', user_model_key)}"
    user = SimulatedUser(
        model=user_model_str,
        temperature=user_temperature,
        api_base=user_model_cfg.get("base_url", "http://localhost:8000/v1"),
        api_key=user_model_cfg.get("api_key", "EMPTY"),
        timeout=user_timeout,
        max_retries=llm_max_retries,
        retry_initial_delay=llm_retry_initial_delay,
        retry_max_delay=llm_retry_max_delay,
        retry_total_timeout=llm_retry_total_timeout,
        retry_jitter=llm_retry_jitter,
        **user_request_options,
    )
    
    # ── Create Strategy ──
    strategy = None
    if is_registered(strategy_name):
        # Build strategy config from run_config (strategy YAML content is merged in)
        strategy_config = {}
        # Pass through retrieval config if present
        if "retrieval" in run_config:
            strategy_config["retrieval"] = dict(run_config["retrieval"])
        # Pass through virtual_tools if present
        if "virtual_tools" in run_config:
            strategy_config["virtual_tools"] = run_config["virtual_tools"]
        # Pass through skill_context config if present
        if "skill_context" in run_config:
            strategy_config["skill_context"] = dict(run_config["skill_context"])
        # Inject rag_index_dir from CLI into retrieval config
        rag_index_dir = run_config.get("rag_index_dir")
        if rag_index_dir:
            strategy_config.setdefault("retrieval", {})["index_dir"] = rag_index_dir
        # Inject embedding config from models.yaml if available (models.yaml takes priority)
        embedding_cfg = run_config.get("_models_config", {}).get("embedding", {}).get("default", {})
        if embedding_cfg and "retrieval" in strategy_config:
            r_cfg = strategy_config["retrieval"]
            if embedding_cfg.get("base_url"):
                r_cfg["embedding_api_base"] = embedding_cfg["base_url"]
            if embedding_cfg.get("backend"):
                backend = embedding_cfg["backend"]
                # Normalize backend name
                if backend == "openai_compatible":
                    backend = "openai_http"
                r_cfg["embedding_backend"] = backend
            if embedding_cfg.get("model"):
                r_cfg["embedding_model"] = embedding_cfg["model"]
        
        # ── Inject skill library paths and config ──
        skill_library_path = run_config.get("skill_library")
        if skill_library_path and uses_skill_library(strategy_name):
            skill_context_cfg = strategy_config.setdefault("skill_context", {})
            # Set skill_library_dir from --skill-library CLI arg
            if "skill_library_dir" not in skill_context_cfg:
                skill_context_cfg["skill_library_dir"] = skill_library_path
            # Set index_dir: default to {skill_library}/index_cache if not specified
            if "index_dir" not in skill_context_cfg:
                default_index = str(Path(skill_library_path) / "index_cache")
                skill_context_cfg["index_dir"] = default_index
            # Inject embedding config from models.yaml into skill_context
            if embedding_cfg:
                if "embedding_api_base" not in skill_context_cfg and embedding_cfg.get("base_url"):
                    skill_context_cfg["embedding_api_base"] = embedding_cfg["base_url"]
                if "embedding_backend" not in skill_context_cfg and embedding_cfg.get("backend"):
                    backend = embedding_cfg["backend"]
                    if backend == "openai_compatible":
                        backend = "openai_http"
                    skill_context_cfg["embedding_backend"] = backend
                if "embedding_model" not in skill_context_cfg and embedding_cfg.get("model"):
                    skill_context_cfg["embedding_model"] = embedding_cfg["model"]
            # Inject LLM config (use agent model for skill LLM calls)
            strategy_config["_llm_config"] = {
                "model_name": agent_model_cfg.get("model_name", agent_model_key),
                "base_url": agent_model_cfg.get("base_url", "http://localhost:8000/v1"),
                "api_key": agent_model_cfg.get("api_key", "EMPTY"),
                "max_tokens": agent_model_cfg.get("max_tokens", 4096),
                "timeout": agent_timeout,
                "request_options": dict(strategy_request_options),
            }
            # Pass embedding config for runtime
            strategy_config["_embedding_config"] = embedding_cfg
            
            # Also inject the embedding config into
            # retrieval sub-config (for the ToolRAGRuntime used in iterative retrieval)
            if "retrieval" in strategy_config and embedding_cfg:
                r_cfg = strategy_config["retrieval"]
                if "embedding_api_base" not in r_cfg and embedding_cfg.get("base_url"):
                    r_cfg["embedding_api_base"] = embedding_cfg["base_url"]
                if "embedding_backend" not in r_cfg and embedding_cfg.get("backend"):
                    backend = embedding_cfg["backend"]
                    if backend == "openai_compatible":
                        backend = "openai_http"
                    r_cfg["embedding_backend"] = backend
                if "embedding_model" not in r_cfg and embedding_cfg.get("model"):
                    r_cfg["embedding_model"] = embedding_cfg["model"]
        
        # ── Inject explorer config (for the explorer-subagent strategy) ──
        if "explorer" in run_config:
            strategy_config["explorer"] = dict(run_config["explorer"])
        if uses_explorer(strategy_name):
            strategy_config["_explorer_llm_config"] = {
                "model_name": agent_model_cfg.get("model_name", agent_model_key),
                "base_url": agent_model_cfg.get("base_url", ""),
                "api_key": agent_model_cfg.get("api_key", "EMPTY"),
                "max_tokens": agent_model_cfg.get("max_tokens", 4096),
                "timeout": agent_timeout,
                "request_options": dict(strategy_request_options),
            }
            # Explorer traces are written under {output_dir}/explore_traces/{domain}.jsonl
            # NOTE: domain/version must come from here, NOT from
            # StrategyRuntimeContext - the pipeline derives those from
            # evaluation_context, which is empty on the test path.
            strategy_config["_run_context"] = {
                "output_dir": str(output_dir),
                "run_name": run_config.get("run_name", ""),
                "domain": domain,
                "version": version,
            }
        
        strategy = build_strategy(strategy_name, config=strategy_config)
        prewarm = getattr(strategy, "prewarm", None)
        if callable(prewarm):
            try:
                prewarm()
            except Exception as exc:
                raise RuntimeError(
                    f"[{domain}] Strategy infrastructure initialization failed"
                ) from exc
        logger.info(f"[{domain}] Strategy: {strategy_name}")
    else:
        logger.warning(f"[{domain}] Strategy '{strategy_name}' not registered, running without strategy")
    
    # ── Create Pipeline ──
    pipeline_config = PipelineConfig(
        max_turns=max_turns,
        max_steps_per_turn=max_steps_per_turn,
        mode=PipelineMode.POSITIVE,
        use_checkpoints=False,  # The CLI owns durable per-task record checkpoints.
        resume=False,
        verbose=verbose,
        current_domain=domain,
        tool_to_domain=tool_router.tool_to_domain if all_tools_mode else None,
    )
    
    pipeline = Pipeline(
        agent=agent,
        user=user,
        tool_router=tool_router,
        config=pipeline_config,
        strategy=strategy,
    )
    
    # ── Run info (for output metadata) ──
    all_tools_count = len(tool_router.get_all_tool_names())
    run_info = {
        "agent_strategy": strategy_name,
        "version": version,
        "split": run_config.get("split", "test"),
        "agent_model": agent_model_cfg.get("model_name", agent_model_key),
        "user_model": user_model_cfg.get("model_name", user_model_key),
        "all_tools_mode": all_tools_mode,
        "all_tools_count": all_tools_count,
        "run_name": run_config.get("run_name", ""),
    }
    
    # Add optional run info fields
    if run_config.get("skill_library"):
        run_info["skill_library_path"] = run_config["skill_library"]
    
    # ── Process tasks in parallel within domain ──
    import threading
    max_query_workers = run_config.get("parallel", {}).get("max_query_workers", 64)
    
    if verbose:
        print(f"  [{domain}] Processing {len(tasks)} tasks ({all_tools_count} tools, "
              f"workers={max_query_workers})...")
    
    # Shared tools_info (read-only, safe to share across threads)
    tools_info = tool_router.get_all_openai_tools(inject_attributes=True)
    
    # File write lock for thread-safe JSONL appending
    write_lock = threading.Lock()
    completed_count = [0]  # Use list for mutability in closure
    
    def _create_thread_pipeline(agent_base_url: str):
        """Create a fresh Pipeline instance for a worker thread."""
        t_agent = Agent(
            model=agent_model_str,
            system_prompt=agent_system_prompt,
            temperature=agent_temperature,
            api_base=agent_base_url,
            api_key=agent_model_cfg.get("api_key", "EMPTY"),
            max_tokens=agent_model_cfg.get("max_tokens", 4096),
            timeout=agent_timeout,
            max_retries=llm_max_retries,
            retry_initial_delay=llm_retry_initial_delay,
            retry_max_delay=llm_retry_max_delay,
            retry_total_timeout=llm_retry_total_timeout,
            retry_jitter=llm_retry_jitter,
            enable_thinking=enable_thinking and not no_thinking,
            **agent_request_options,
        )
        t_user = SimulatedUser(
            model=user_model_str,
            temperature=user_temperature,
            api_base=user_model_cfg.get("base_url", "http://localhost:8000/v1"),
            api_key=user_model_cfg.get("api_key", "EMPTY"),
            timeout=user_timeout,
            max_retries=llm_max_retries,
            retry_initial_delay=llm_retry_initial_delay,
            retry_max_delay=llm_retry_max_delay,
            retry_total_timeout=llm_retry_total_timeout,
            retry_jitter=llm_retry_jitter,
            **user_request_options,
        )
        t_pipeline = Pipeline(
            agent=t_agent,
            user=t_user,
            tool_router=tool_router,
            config=pipeline_config,
            strategy=strategy,
        )
        return t_pipeline
    
    def _process_single_task(task_idx_tuple):
        """Process a single task and return the output entry."""
        i, task = task_idx_tuple
        query_id = task.query_id
        sample = _build_sample_from_task(task, version)
        last_output: Optional[Dict[str, Any]] = None
        last_success = False
        attempts_used = 0
        
        for sample_attempt in range(1, sample_max_retries + 1):
            attempts_used = sample_attempt
            try:
                # Each attempt gets a fresh pipeline and no checkpoint resume. This prevents
                # a transient LLM failure from resuming a partially advanced trajectory.
                agent_base_url = select_model_endpoint(agent_model_cfg, query_id)
                thread_pipeline = _create_thread_pipeline(agent_base_url)
                thread_pipeline.config = replace(
                    thread_pipeline.config,
                    resume=False,
                    use_checkpoints=False,
                )
                thread_pipeline.checkpoint_manager = None

                result = thread_pipeline.process_multi_turn(
                    _build_sample_from_task(task, version),
                    tools_info=tools_info,
                )

                output = result.to_dict()
                output["run"] = {**output.get("run", {}), **run_info}
                output["run"]["agent_base_url"] = agent_base_url
                output["run"]["sample_attempts"] = sample_attempt
                last_output = output
                last_success = result.success

                if result.success or not is_transient_error(result.error):
                    break
                if sample_attempt >= sample_max_retries:
                    break

                delay = compute_retry_delay(sample_retry_config, sample_attempt)
                logger.warning(
                    "[%s] Sample %s transient failure on attempt %s/%s; "
                    "retrying from scratch in %.1fs: %s",
                    domain,
                    query_id,
                    sample_attempt,
                    sample_max_retries,
                    delay,
                    result.error,
                )
                time.sleep(delay)

            except Exception as e:
                logger.error(f"[{domain}] Error processing {query_id}: {e}")
                logger.debug(traceback.format_exc())
                error_entry = {
                    "schema_version": 2,
                    "id": query_id,
                    "success": False,
                    "error": str(e),
                    "task": task.to_dict(),
                    "agent_traj": [],
                    "conversation": {"messages": [], "total_turns": 0},
                    "run": {
                        **run_info,
                        "agent_base_url": select_model_endpoint(agent_model_cfg, query_id),
                        "sample_attempts": sample_attempt,
                    },
                    "artifacts": {},
                }
                last_output = error_entry
                last_success = False

                if not is_transient_error(e) or sample_attempt >= sample_max_retries:
                    break

                delay = compute_retry_delay(sample_retry_config, sample_attempt)
                logger.warning(
                    "[%s] Sample %s exception on attempt %s/%s; "
                    "retrying from scratch in %.1fs: %s",
                    domain,
                    query_id,
                    sample_attempt,
                    sample_max_retries,
                    delay,
                    e,
                )
                time.sleep(delay)

        if attempts_used > 1:
            with write_lock:
                stats["sample_retries"] += attempts_used - 1

        status_type = "success" if last_output is not None else "error"
        if last_output is None:
            last_output = {
                "schema_version": 2,
                "id": query_id,
                "success": False,
                "error": "Unknown rollout failure",
                "task": task.to_dict(),
                "agent_traj": [],
                "conversation": {"messages": [], "total_turns": 0},
                "run": {
                    **run_info,
                    "agent_base_url": select_model_endpoint(agent_model_cfg, query_id),
                },
                "artifacts": {},
            }
        return (status_type, i, query_id, last_output, last_success)
    
    with output_path.open("a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=max_query_workers) as executor:
            futures = {
                executor.submit(_process_single_task, (i, task)): (i, task)
                for i, task in enumerate(tasks)
            }
            
            for future in as_completed(futures):
                status_type, i, query_id, output, success = future.result()
                
                # Publish a verified per-task checkpoint before updating JSONL.
                with write_lock:
                    serialized = save_record(
                        output_dir,
                        domain,
                        query_id,
                        output,
                        expected_version=version,
                    )
                    f.write(serialized)
                    f.flush()
                
                if status_type == "success":
                    stats["completed"] += 1
                else:
                    stats["failed"] += 1
                
                completed_count[0] += 1
                
                if verbose:
                    status = "✓" if success else "✗"
                    if status_type == "success":
                        turns = output.get("conversation", {}).get("total_turns", 0)
                        print(f"    [{completed_count[0]}/{len(tasks)}] {query_id[:16]}... "
                              f"{status} ({turns} turns)")
                    else:
                        print(f"    [{completed_count[0]}/{len(tasks)}] {query_id[:16]}... "
                              f"✗ Error: {output.get('error', 'unknown')}")
    
    # JSONL is a public aggregate; the verified per-task records are authoritative.
    # Rebuilding also repairs an interrupted or partial append before completion.
    final = rebuild_domain_jsonl(
        output_dir,
        domain,
        all_task_ids,
        expected_version=version,
    )
    if final["record_ids"] != all_task_ids:
        missing = sorted(all_task_ids - final["record_ids"])
        raise RuntimeError(
            f"[{domain}] Incomplete rollout record set: "
            f"{len(final['record_ids'])}/{len(all_task_ids)}; missing={missing[:5]}"
        )

    stats["elapsed"] = time.time() - start_time
    return stats


def _build_sample_from_task(task: TaskData, version: str) -> Dict[str, Any]:
    """Convert a TaskData into the sample dict expected by Pipeline.
    
    Pipeline expects:
        - id / query_id
        - messages (initial conversation)
        - user_system_prompt
        - task_info (for strategy to determine gold tools)
    
    Args:
        task: Loaded TaskData object.
        version: Current version string.
    
    Returns:
        Sample dict compatible with Pipeline.process_multi_turn().
    """
    # Get version-specific gold tools
    try:
        version_tools = task.get_version_tools(version)
    except KeyError:
        version_tools = task.gold_trajectory.tools  # Fallback to base
    
    sample = {
        "id": task.query_id,
        "query_id": task.query_id,
        "messages": task.get_initial_messages(),
        "user_system_prompt": task.user_system_prompt,
        "task_info": {
            "domain": task.domain,
            "trajectory": version_tools,
            "trajectory_hash": task.gold_trajectory.trajectory_hash,
            "combo_id": task.combo_id,
            "instruction": task.instruction,
            "reason_for_call": task.reason_for_call,
        },
        # Gold trajectory info (base version — matches step_param_codes)
        "gold_trajectory": {
            "tools": task.gold_trajectory.tools,
            "trajectory_hash": task.gold_trajectory.trajectory_hash,
            "step_param_codes": task.gold_trajectory.step_param_codes,
            "entity_instances": task.gold_trajectory.entity_instances,
            "value_domain_samples": task.gold_trajectory.value_domain_samples,
        },
    }
    return sample


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments following plan §8 interface."""
    parser = argparse.ArgumentParser(
        description="dtau_bench Rollout — Run trajectory generation experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # ── Required / key arguments ──
    parser.add_argument(
        "--run-name", type=str, required=True,
        help="Experiment run name (used for output directory: experiments/{run_name}/).",
    )
    parser.add_argument(
        "--strategy", type=str, default="single_domain_full",
        help="Strategy name (registered in strategy registry). Default: single_domain_full",
    )
    parser.add_argument(
        "--split", type=str, choices=["train", "test"], default="test",
        help="Data split to process (default: test).",
    )
    parser.add_argument(
        "--version", type=str, default="v1",
        help="Tool version to use (e.g., base, v1, v2, ..., v6). Default: v1",
    )
    
    # ── Domain selection ──
    domain_group = parser.add_mutually_exclusive_group()
    domain_group.add_argument(
        "--domains", nargs="+", type=str, default=None,
        help="Specific domains to process. Default: all available domains.",
    )
    domain_group.add_argument(
        "--list", action="store_true",
        help="List available domains and exit.",
    )
    
    # ── Paths ──
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Data root (default: DTAU_BENCH_DATA_ROOT or {project_root}/data).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output base directory (default: {project_root}/experiments/{run_name}).",
    )
    
    # ── Model selection ──
    parser.add_argument(
        "--agent-model", type=str, default="deepseek-v4-pro",
        help="Agent model key from models.yaml (default: deepseek-v4-pro).",
    )
    parser.add_argument(
        "--user-model", type=str, default="deepseek-flash",
        help="Simulated user model key from models.yaml (default: deepseek-flash).",
    )

    # ── Parallelism ──
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="Number of domains to process in parallel (default from config: 8).",
    )
    parser.add_argument(
        "--query-workers", type=int, default=None,
        help="Number of queries to process in parallel per domain (reserved for future).",
    )
    
    # ── Skill library (for skill_context strategies) ──
    parser.add_argument(
        "--skill-library", type=str, default=None,
        help="Path to skill library directory (for skill_context strategies).",
    )

    # ── RAG (for iterative_rag strategies) ──
    parser.add_argument(
        "--rag-index-dir", type=str, default=None,
        help="Path to RAG index directory (for tool retrieval strategies).",
    )
    
    # ── Flags ──
    parser.add_argument("--no-thinking", action="store_true", help="Disable legacy thinking flags; explicit model reasoning_effort takes precedence.")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from previous runs.")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="When resuming, re-run previous transient LLM failures instead of skipping them.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output (DEBUG logging).")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode (WARNING only).")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # ── Logging setup ──
    log_level = "DEBUG" if args.verbose else ("WARNING" if args.quiet else "INFO")
    setup_logging(level=log_level, verbose=args.verbose, quiet=args.quiet)
    
    # ── Resolve paths ──
    project_root = get_project_root()
    data_dir = get_data_root(args.data_dir)
    
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = (project_root / "experiments" / args.run_name).resolve()
    
    # ── List mode ──
    if args.list:
        domains = discover_domains(data_dir, args.split)
        print(f"\nAvailable domains (split={args.split}):")
        print(f"{'─' * 50}")
        for i, d in enumerate(domains, 1):
            print(f"  {i:2d}. {d}")
        print(f"\nTotal: {len(domains)} domains")
        
        # Also list strategies
        strategies = list_strategies()
        print(f"\nRegistered strategies:")
        print(f"{'─' * 50}")
        for s in strategies:
            print(f"  • {s}")
        return
    
    # ── Load models config ──
    models_config = load_models_config()
    
    # ── Load and merge configuration ──
    # Only knobs that legitimately vary per invocation are CLI-settable; the
    # rest live in configs/defaults.yaml.
    cli_overrides: Dict[str, Any] = {}
    if args.max_workers is not None:
        cli_overrides.setdefault("parallel", {})["max_domain_workers"] = args.max_workers
    if args.query_workers is not None:
        cli_overrides.setdefault("parallel", {})["max_query_workers"] = args.query_workers
    
    # Assemble full config: defaults → strategy → CLI overrides
    try:
        run_config = load_config(
            strategy_name=args.strategy,
            cli_overrides=cli_overrides,
        )
    except FileNotFoundError as e:
        logger.warning(f"Config loading issue: {e}")
        # Fall back to defaults only with CLI overrides
        from dtau_bench.utils.config import load_defaults_config, merge_configs
        run_config = merge_configs(load_defaults_config(), cli_overrides)
        run_config["strategy"] = args.strategy
    
    # Inject additional run-level config
    run_config["run_name"] = args.run_name
    run_config["split"] = args.split
    run_config["version"] = args.version
    run_config["agent_model"] = args.agent_model
    run_config["user_model"] = args.user_model
    run_config["no_thinking"] = args.no_thinking
    run_config["enable_thinking"] = not args.no_thinking
    run_config["skill_library"] = args.skill_library
    run_config["rag_index_dir"] = args.rag_index_dir
    run_config["retry_failed"] = args.retry_failed
    run_config["_models_config"] = models_config  # Internal: for Agent/User instantiation
    
    # ── Discover / filter domains ──
    if args.domains:
        domains = args.domains
    else:
        domains = discover_domains(data_dir, args.split)
    
    if not domains:
        print(f"[ERROR] No domains found for split='{args.split}' in {data_dir}")
        sys.exit(1)
    
    # ── Load tasks ──
    print(f"\n{'═' * 70}")
    print(f"  dtau_bench Rollout")
    print(f"{'═' * 70}")
    print(f"  Run Name:   {args.run_name}")
    print(f"  Strategy:   {args.strategy}")
    print(f"  Split:      {args.split}")
    print(f"  Version:    {args.version}")
    print(f"  Domains:    {len(domains)}")
    print(f"  Agent:      {args.agent_model}")
    print(f"  User:       {args.user_model}")
    print(f"  LLM Retry:  {run_config.get('rollout', {}).get('llm_max_retries', 10)}")
    print(f"  Retry Mode: dtau_bench exponential backoff")
    print(f"  Sample Retry: {run_config.get('rollout', {}).get('sample_max_retries', 3)}")
    print(f"  Retry Failed: {args.retry_failed}")
    print(f"  Output:     {output_dir}")
    print(f"  Resume:     {not args.no_resume}")
    print(f"{'═' * 70}\n")
    
    all_tasks = load_tasks(
        split=args.split,
        data_dir=data_dir,
        domains=domains,
        version=args.version,
    )
    
    total_tasks = sum(len(ts) for ts in all_tasks.values())
    if total_tasks == 0:
        print("[WARN] No tasks found matching criteria. Check --split, --version, --domains.")
        sys.exit(0)
    
    print(f"Loaded {total_tasks} tasks across {len(all_tasks)} domains:")
    for d, ts in sorted(all_tasks.items()):
        print(f"  • {d}: {len(ts)} tasks")
    print()
    
    # ── Save config snapshot ──
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "config_snapshot.yaml"
    # Remove internal _models_config (contains api keys) before saving
    config_to_save = {k: v for k, v in run_config.items() if not k.startswith("_")}
    config_to_save["domains"] = domains
    config_to_save["total_tasks"] = total_tasks
    config_to_save["started_at"] = datetime.now().isoformat()
    save_config_snapshot(config_to_save, snapshot_path)
    logger.info(f"Config snapshot saved: {snapshot_path}")
    
    # ── Execute: domain-level parallelism ──
    max_domain_workers = run_config.get("parallel", {}).get("max_domain_workers", 8)
    resume = not args.no_resume
    verbose = not args.quiet
    
    overall_start = time.time()
    all_stats: List[Dict[str, Any]] = []
    
    if max_domain_workers <= 1 or len(all_tasks) == 1:
        # Sequential processing
        for domain, tasks in sorted(all_tasks.items()):
            if not tasks:
                continue
            stats = process_domain(
                domain=domain,
                tasks=tasks,
                version=args.version,
                run_config=run_config,
                output_dir=output_dir,
                data_dir=data_dir,
                resume=resume,
                verbose=verbose,
            )
            all_stats.append(stats)
    else:
        # Parallel domain processing
        with ThreadPoolExecutor(max_workers=max_domain_workers) as executor:
            futures = {
                executor.submit(
                    process_domain,
                    domain=domain,
                    tasks=tasks,
                    version=args.version,
                    run_config=run_config,
                    output_dir=output_dir,
                    data_dir=data_dir,
                    resume=resume,
                    verbose=verbose,
                ): domain
                for domain, tasks in all_tasks.items()
                if tasks
            }
            
            for future in as_completed(futures):
                domain = futures[future]
                try:
                    stats = future.result()
                    all_stats.append(stats)
                except Exception as e:
                    logger.error(f"Domain {domain} failed with exception: {e}")
                    all_stats.append({
                        "domain": domain,
                        "total": len(all_tasks.get(domain, [])),
                        "completed": 0,
                        "skipped": 0,
                        "failed": len(all_tasks.get(domain, [])),
                        "elapsed": 0,
                        "error": str(e),
                    })
    
    # ── Summary ──
    overall_elapsed = time.time() - overall_start
    total_completed = sum(s.get("completed", 0) for s in all_stats)
    total_skipped = sum(s.get("skipped", 0) for s in all_stats)
    total_failed = sum(s.get("failed", 0) for s in all_stats)
    total_sample_retries = sum(s.get("sample_retries", 0) for s in all_stats)
    total_retried_failed_records = sum(s.get("retried_failed_records", 0) for s in all_stats)
    
    print(f"\n{'═' * 70}")
    print(f"  Rollout Complete")
    print(f"{'═' * 70}")
    print(f"  Total tasks:    {total_tasks}")
    print(f"  Completed:      {total_completed}")
    print(f"  Skipped (resume): {total_skipped}")
    print(f"  Retried failed records: {total_retried_failed_records}")
    print(f"  Sample retries: {total_sample_retries}")
    print(f"  Failed:         {total_failed}")
    print(f"  Elapsed:        {overall_elapsed:.1f}s")
    print(f"  Output:         {output_dir / 'rollouts/'}")
    print(f"{'═' * 70}")
    
    # Per-domain summary
    if len(all_stats) > 1:
        print(f"\n  Per-domain breakdown:")
        for s in sorted(all_stats, key=lambda x: x["domain"]):
            status = "✓" if s.get("failed", 0) == 0 else "⚠"
            elapsed = s.get("elapsed", 0)
            print(
                f"    {status} {s['domain']}: "
                f"{s.get('completed', 0)} done, "
                f"{s.get('skipped', 0)} skipped, "
                f"{s.get('failed', 0)} failed "
                f"({elapsed:.1f}s)"
            )
    
    # ── Aggregate token usage from rollout JSONL files ──
    token_agg = {
        "agent_prompt_tokens": 0,
        "agent_completion_tokens": 0,
        "agent_total_tokens": 0,
        "agent_calls": 0,
        "user_prompt_tokens": 0,
        "user_completion_tokens": 0,
        "user_total_tokens": 0,
        "user_calls": 0,
        "total_tokens": 0,
        "samples_with_usage": 0,
    }
    rollout_dir = output_dir / "rollouts"
    if rollout_dir.exists():
        for jsonl_file in sorted(rollout_dir.glob("*.jsonl")):
            try:
                with jsonl_file.open("r", encoding="utf-8") as rf:
                    for line in rf:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        tu = record.get("token_usage")
                        if not tu:
                            continue
                        token_agg["samples_with_usage"] += 1
                        agent = tu.get("agent", {})
                        user = tu.get("user", {})
                        token_agg["agent_prompt_tokens"] += agent.get("prompt_tokens", 0)
                        token_agg["agent_completion_tokens"] += agent.get("completion_tokens", 0)
                        token_agg["agent_total_tokens"] += agent.get("total_tokens", 0)
                        token_agg["agent_calls"] += agent.get("calls", 0)
                        token_agg["user_prompt_tokens"] += user.get("prompt_tokens", 0)
                        token_agg["user_completion_tokens"] += user.get("completion_tokens", 0)
                        token_agg["user_total_tokens"] += user.get("total_tokens", 0)
                        token_agg["user_calls"] += user.get("calls", 0)
                        token_agg["total_tokens"] += tu.get("total_tokens", 0)
            except Exception as e:
                logger.warning(f"Error reading token usage from {jsonl_file}: {e}")

    # ── Save run summary ──
    summary = {
        "run_name": args.run_name,
        "strategy": args.strategy,
        "split": args.split,
        "version": args.version,
        "domains": domains,
        "total_tasks": total_tasks,
        "completed": total_completed,
        "skipped": total_skipped,
        "failed": total_failed,
        "retried_failed_records": total_retried_failed_records,
        "sample_retries": total_sample_retries,
        "elapsed_seconds": round(overall_elapsed, 2),
        "finished_at": datetime.now().isoformat(),
        "per_domain": all_stats,
    }
    if token_agg["samples_with_usage"] > 0:
        summary["token_usage_total"] = token_agg
        logger.info(
            f"Token usage: {token_agg['total_tokens']:,} total tokens "
            f"({token_agg['agent_calls'] + token_agg['user_calls']} API calls, "
            f"{token_agg['samples_with_usage']} samples tracked)"
        )
    summary_path = output_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Run summary saved: {summary_path}")
    
    # Exit with error code if any failures
    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
