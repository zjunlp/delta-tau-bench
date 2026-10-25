"""
Configuration loading and merging utilities.

Provides functions to:
- Load YAML configuration files
- Merge multiple config layers (defaults → strategy → CLI overrides)
- Resolve paths relative to project root
"""

import os
import copy
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import yaml


# OpenAI-compatible request parameters which may live next to endpoint metadata
# in ``configs/models.yaml``.  Keep this list explicit: blindly forwarding a
# model entry would also pass local-only fields such as ``supports_tools`` and,
# more seriously, credentials.
STANDARD_MODEL_REQUEST_OPTION_KEYS = (
    "temperature",
    "top_p",
    "n",
    "stop",
    "max_completion_tokens",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "user",
    "reasoning_effort",
    "verbosity",
    "response_format",
    "seed",
    "logprobs",
    "top_logprobs",
    "parallel_tool_calls",
    "service_tier",
    "safety_identifier",
    "modalities",
    "prediction",
    "audio",
    "web_search_options",
    "stream_options",
    "extra_headers",
)


def extract_model_request_options(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """Return request parameters from one model configuration entry.

    Standard parameters can be written directly in the model entry.  A
    ``request_options`` mapping is also accepted for forward-compatible or
    gateway-specific options.  Direct standard fields win over the nested
    mapping, and ``None`` means "do not send".

    Endpoint metadata and credentials are intentionally never inferred as
    request options.
    """
    nested = entry.get("request_options")
    options = dict(nested) if isinstance(nested, Mapping) else {}
    for name in STANDARD_MODEL_REQUEST_OPTION_KEYS:
        if name in entry:
            value = entry[name]
            if value is None:
                options.pop(name, None)
            else:
                options[name] = copy.deepcopy(value)
    return options


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────

def get_project_root() -> Path:
    """Return the code directory holding ``configs/``.

    Resolution order: ``DTAU_BENCH_ROOT``, then the current working directory.
    All commands are expected to run from the project root.

    Raises:
        FileNotFoundError: If the resolved directory has no ``configs/``.
    """
    env_root = os.environ.get("DTAU_BENCH_ROOT")
    root = Path(env_root).resolve() if env_root else Path.cwd()
    if not (root / "configs").is_dir():
        raise FileNotFoundError(
            f"No 'configs/' directory under {root}. Run from the project root, "
            "or set DTAU_BENCH_ROOT to it."
        )
    return root


def get_data_root(data_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the dataset root without reading model configuration.

    An explicit directory takes precedence over ``DTAU_BENCH_DATA_ROOT``.
    Otherwise the dataset defaults to ``data/`` under the project root.
    Relative directories are resolved against the project root, so the
    dataset layout remains ``tasks/``, ``tools/``, ``database/``, ``servers/``
    and ``meta/`` whether it is bundled with the code or stored separately.

    This function does not require the dataset to exist. Readers and the
    ``validate-data`` command report missing or incomplete datasets.
    """
    configured = data_dir if data_dir is not None else os.environ.get("DTAU_BENCH_DATA_ROOT")
    if configured is None or configured == "":
        return (get_project_root() / "data").resolve()
    path = Path(configured).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (get_project_root() / path).resolve()


def resolve_path(path: Union[str, Path], base: Optional[Path] = None) -> Path:
    """Resolve a path, treating relative paths as relative to base (or project root).
    
    Args:
        path: The path to resolve (absolute paths are returned as-is).
        base: Base directory for relative paths. Defaults to project root.
    
    Returns:
        Resolved absolute Path.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    if base is None:
        base = get_project_root()
    return (base / p).resolve()


# ─────────────────────────────────────────────────────────────────────────────
# YAML loading
# ─────────────────────────────────────────────────────────────────────────────

def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file and return as a dictionary.
    
    Args:
        path: Path to the YAML file.
    
    Returns:
        Parsed dictionary. Empty dict if file is empty.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If YAML root is not a mapping.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping (dict), got {type(data).__name__}: {path}")
    
    return data


def load_strategy_config(strategy_name: str, config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load a strategy configuration by name.
    
    Looks for configs/strategies/{strategy_name}.yaml
    
    Args:
        strategy_name: Name of the strategy (e.g., 'single_domain_full').
        config_dir: Optional override for the strategies config directory.
    
    Returns:
        Strategy configuration dictionary.
    """
    if config_dir is None:
        config_dir = get_project_root() / "configs" / "strategies"
    
    yaml_path = config_dir / f"{strategy_name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Strategy config not found: {yaml_path}\n"
            f"Available strategies: {[p.stem for p in config_dir.glob('*.yaml')]}"
        )
    
    return load_yaml(yaml_path)


def load_models_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load models configuration.
    
    Args:
        config_path: Optional path override. Defaults to configs/models.yaml.
    
    Returns:
        Models configuration dictionary.
    """
    if config_path is None:
        config_path = get_project_root() / "configs" / "models.yaml"
    return _expand_config_environment(load_yaml(config_path))


_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_config_environment(value: Any) -> Any:
    """Expand environment references after YAML parsing, without shell execution.

    ``${NAME:-default}`` uses its default when NAME is absent or empty.
    A missing ``${NAME}`` raises an actionable error without exposing values.
    Parsing first preserves keys, quotes, newlines, and types in credentials.
    """
    if isinstance(value, dict):
        return {key: _expand_config_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_config_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    def substitute(match: re.Match) -> str:
        name, default = match.group(1), match.group(2)
        resolved = os.environ.get(name)
        if resolved:
            return resolved
        if default is not None:
            return default
        raise ValueError(f"Set environment variable {name} required by models.yaml.")

    return _ENV_REFERENCE.sub(substitute, value)


def resolve_model_endpoint(model_key: str) -> Dict[str, Any]:
    """Resolve one ``models:`` entry of configs/models.yaml.

    Args:
        model_key: Key under ``models:``.

    Returns:
        Endpoint fields plus ``request_options``. When the entry lists
        ``base_urls``, the first endpoint is used.

    Raises:
        KeyError: If the key is absent from models.yaml.
    """
    models = load_models_config().get("models") or {}
    entry = models.get(model_key)
    if not entry:
        raise KeyError(
            f"Model key {model_key!r} is not in configs/models.yaml. "
            f"Available: {sorted(models)}"
        )
    base_url = entry.get("base_url") or (entry.get("base_urls") or [""])[0]
    return {
        "model_name": entry.get("model_name", model_key),
        "base_url": base_url,
        "api_key": entry.get("api_key", ""),
        "max_tokens": entry.get("max_tokens", 40960),
        "request_options": extract_model_request_options(entry),
    }


def resolve_embedding_config() -> Dict[str, Any]:
    """Resolve the ``embedding.default`` entry of configs/models.yaml.

    Returns:
        ``{"model", "backend", "base_url", "api_key", "batch_size",
        "request_timeout", "max_retries", "retry_delay"}``.
        The backend is normalised to ``openai_http`` or ``local``.

    Raises:
        KeyError: If ``embedding.default`` is missing.
        ValueError: If the http backend has no base_url.
    """
    entry = (load_models_config().get("embedding") or {}).get("default")
    if not entry:
        raise KeyError(
            "configs/models.yaml has no 'embedding.default' section; both skill "
            "and tool retrieval need it."
        )
    backend = entry.get("backend", "openai_http")
    if backend == "openai_compatible":
        backend = "openai_http"
    base_url = entry.get("base_url", "")
    if backend == "openai_http" and not base_url:
        raise ValueError(
            "embedding.default.base_url is required for the openai_http backend."
        )
    batch_size = entry.get("batch_size", 64)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(
            "embedding.default.batch_size must be a positive integer, "
            f"got {batch_size!r}."
        )
    request_timeout = entry.get("request_timeout", 600.0)
    if (
        isinstance(request_timeout, bool)
        or not isinstance(request_timeout, (int, float))
        or request_timeout <= 0
    ):
        raise ValueError(
            "embedding.default.request_timeout must be a positive number, "
            f"got {request_timeout!r}."
        )
    max_retries = entry.get("max_retries", 2)
    if (
        isinstance(max_retries, bool)
        or not isinstance(max_retries, int)
        or max_retries < 0
    ):
        raise ValueError(
            "embedding.default.max_retries must be a non-negative integer, "
            f"got {max_retries!r}."
        )
    retry_delay = entry.get("retry_delay", 1.0)
    if (
        isinstance(retry_delay, bool)
        or not isinstance(retry_delay, (int, float))
        or retry_delay < 0
    ):
        raise ValueError(
            "embedding.default.retry_delay must be a non-negative number, "
            f"got {retry_delay!r}."
        )
    return {
        "model": entry.get("model", ""),
        "backend": backend,
        "base_url": base_url,
        "api_key": entry.get("api_key", ""),
        "batch_size": batch_size,
        "request_timeout": request_timeout,
        "max_retries": max_retries,
        "retry_delay": retry_delay,
    }


def load_defaults_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load default hyperparameters configuration.
    
    Args:
        config_path: Optional path override. Defaults to configs/defaults.yaml.
    
    Returns:
        Defaults configuration dictionary.
    """
    if config_path is None:
        config_path = get_project_root() / "configs" / "defaults.yaml"
    return load_yaml(config_path)


# ─────────────────────────────────────────────────────────────────────────────
# Config merging
# ─────────────────────────────────────────────────────────────────────────────

def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge multiple config dictionaries (later takes precedence).
    
    Non-dict values are overwritten. Dict values are recursively merged.
    None values in later configs are skipped (don't overwrite).
    
    Args:
        *configs: Variable number of config dicts, in priority order (last wins).
    
    Returns:
        Merged configuration dictionary.
    
    Example:
        >>> merge_configs(
        ...     {"rollout": {"max_turns": 20, "temperature": 0.7}},
        ...     {"rollout": {"max_turns": 30}},
        ... )
        {"rollout": {"max_turns": 30, "temperature": 0.7}}
    """
    result: Dict[str, Any] = {}
    for config in configs:
        if config:
            result = _deep_merge(result, config)
    return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base (override wins for non-dict values)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if value is None:
            continue
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# High-level config assembly
# ─────────────────────────────────────────────────────────────────────────────

def load_config(
    strategy_name: str,
    strategy_config_path: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble full configuration by merging layers.
    
    Merge order (later wins):
        1. defaults.yaml
        2. Strategy config (from strategy name or explicit path)
        3. CLI overrides
    
    Args:
        strategy_name: Name of the strategy.
        strategy_config_path: Optional explicit path to strategy YAML (overrides name-based lookup).
        cli_overrides: Optional dictionary of CLI-specified overrides.
    
    Returns:
        Fully merged configuration dictionary.
    """
    # Layer 1: defaults
    defaults = load_defaults_config()
    
    # Layer 2: strategy config
    if strategy_config_path:
        strategy_cfg = load_yaml(strategy_config_path)
    else:
        strategy_cfg = load_strategy_config(strategy_name)
    
    # Layer 3: CLI overrides (structured as a dict)
    overrides = cli_overrides or {}
    
    # Merge
    merged = merge_configs(defaults, strategy_cfg, overrides)
    
    # Ensure strategy name is set
    merged["strategy"] = strategy_cfg.get("strategy", strategy_name)
    
    return merged


def save_config_snapshot(config: Dict[str, Any], output_path: Union[str, Path]) -> Path:
    """Save a configuration snapshot to YAML for reproducibility.
    
    Args:
        config: The full merged configuration.
        output_path: Path to save the YAML snapshot.
    
    Returns:
        The path where the snapshot was saved.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    return output_path
