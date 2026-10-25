"""
MCP Server wrapper - dynamically loads MCP server classes and exposes tools.

Provides:
- MCPServerInstance: A loaded MCP server with its tools ready for invocation
- load_mcp_server(): Load a single domain's MCP server class
- setup_domain_tools(): Load server + tool definitions for a single domain
- setup_all_tools(): Load all domains' servers and tools (for all-tools mode)
"""

import importlib.util
import json
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from dtau_bench.utils.logging import get_logger

logger = get_logger("tools.mcp_wrapper")


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definition
# ─────────────────────────────────────────────────────────────────────────────

class ToolDefinition:
    """A single tool's definition (schema + metadata).
    
    Attributes:
        name: The function name (e.g., 'create_career_event_record').
        schema: The full OpenAI function calling schema dict.
        node_id: The evolution tree node ID.
        domain: The domain this tool belongs to.
        attributes: Optional attributes (cost, latency, require_confirmation).
    """

    def __init__(
        self,
        name: str,
        schema: Dict[str, Any],
        node_id: str,
        domain: str,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.schema = schema
        self.node_id = node_id
        self.domain = domain
        self.attributes = attributes or {}

    @property
    def cost(self) -> int:
        return self.attributes.get("cost", 1)

    @property
    def latency(self) -> int:
        return self.attributes.get("latency", 1)

    @property
    def require_confirmation(self) -> bool:
        return self.attributes.get("require_confirmation", False)

    def to_openai_tool(self, inject_attributes: bool = True) -> Dict[str, Any]:
        """Convert to OpenAI function calling format.
        
        Args:
            inject_attributes: Whether to inject cost/latency into description.
        
        Returns:
            Dict in OpenAI tool format: {"type": "function", "function": {...}}
        """
        tool = deepcopy(self.schema)
        
        if inject_attributes and self.attributes:
            func = tool["function"]
            desc = func.get("description", "")
            # Inject cost/latency info
            attr_parts = []
            if "cost" in self.attributes:
                attr_parts.append(f"cost={self.attributes['cost']}")
            if "latency" in self.attributes:
                attr_parts.append(f"latency={self.attributes['latency']}")
            if attr_parts:
                desc = f"{desc} [{', '.join(attr_parts)}]"
            func["description"] = desc
            
            # Inject confirm parameter for tools requiring confirmation
            if self.require_confirmation:
                params = func.get("parameters", {})
                props = params.get("properties", {})
                if "confirm" not in props:
                    props["confirm"] = {
                        "type": "boolean",
                        "description": (
                            "Set to true to confirm execution of this action. "
                            "This tool requires explicit confirmation before it modifies data."
                        ),
                    }
                    params["properties"] = props
                    required = params.get("required", [])
                    if "confirm" not in required:
                        required.append("confirm")
                    params["required"] = required
                func["parameters"] = params
            
            tool["function"] = func
        
        return tool

    def __repr__(self) -> str:
        return f"ToolDefinition(name={self.name!r}, domain={self.domain!r})"


# ─────────────────────────────────────────────────────────────────────────────
# MCP Server Instance
# ─────────────────────────────────────────────────────────────────────────────

class MCPServerInstance:
    """A loaded MCP Server instance with its tool definitions.
    
    Wraps a dynamically loaded MCP Server class and provides a clean
    invoke interface with session management.
    
    Attributes:
        domain: The domain name (e.g., 'AthleteContentDashboard').
        server: The instantiated MCP server object.
        tools: Dict mapping tool function names to ToolDefinition objects.
    """

    def __init__(self, domain: str, server: Any, tools: Dict[str, ToolDefinition]):
        self.domain = domain
        self.server = server
        self.tools = tools

    def invoke(self, session_id: str, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Invoke a tool on this server.
        
        Args:
            session_id: Unique session identifier (for database isolation).
            tool_name: The function name to invoke.
            **kwargs: Tool arguments.
        
        Returns:
            Tool execution result dict.
        
        Raises:
            ValueError: If tool_name is not found on this server.
        """
        if tool_name not in self.tools:
            return {"error": f"Unknown tool '{tool_name}' on domain '{self.domain}'"}
        
        try:
            result = self.server.invoke(session_id, tool_name, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Tool invocation failed: {self.domain}.{tool_name}: {e}")
            return {"error": f"Tool execution error: {str(e)}"}

    def get_session(self, session_id: str) -> Any:
        """Get a session context (for database state capture by evaluator).
        
        Args:
            session_id: Session identifier.
        
        Returns:
            Session context object, or None if not found.
        """
        if hasattr(self.server, "_get_session"):
            return self.server._get_session(session_id)
        return self.server.sessions.get(session_id)

    def create_session(self, session_id: str) -> Any:
        """Pre-create a session (ensures database is copied).
        
        Args:
            session_id: Session identifier.
        
        Returns:
            Session context object.
        """
        if hasattr(self.server, "_get_or_create_session"):
            return self.server._get_or_create_session(session_id)
        # Fallback: invoke a harmless operation to trigger session creation
        return None

    def get_tool_names(self) -> List[str]:
        """Get all tool function names available on this server."""
        return list(self.tools.keys())

    def get_openai_tools(self, inject_attributes: bool = True) -> List[Dict[str, Any]]:
        """Get all tool definitions in OpenAI format.
        
        Args:
            inject_attributes: Whether to inject cost/latency into descriptions.
        
        Returns:
            List of OpenAI tool dicts.
        """
        return [t.to_openai_tool(inject_attributes) for t in self.tools.values()]

    def __repr__(self) -> str:
        return f"MCPServerInstance(domain={self.domain!r}, tools={len(self.tools)})"


# ─────────────────────────────────────────────────────────────────────────────
# Server Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_mcp_server_class(domain: str, servers_dir: Path) -> type:
    """Dynamically load an MCP Server class from a Python file.
    
    Args:
        domain: Domain name (e.g., 'AthleteContentDashboard').
        servers_dir: Path to the servers directory (may be versioned, e.g., servers/v1/).
    
    Returns:
        The MCP Server class (not instantiated).
    
    Raises:
        FileNotFoundError: If the server file doesn't exist.
        AttributeError: If the expected class is not found in the module.
    """
    server_file = servers_dir / f"{domain}.py"
    if not server_file.exists():
        raise FileNotFoundError(f"MCP Server file not found: {server_file}")
    
    # Use a unique module name to avoid cache conflicts between versions
    # e.g., "mcp_servers.v1.AgriculturalEngineeringTrainingServer"
    dir_name = servers_dir.name  # "servers" or "v1", "v2", etc.
    if dir_name in ("v1", "v2", "v3", "v4", "v5", "v6"):
        module_name = f"mcp_servers.{dir_name}.{domain}"
    else:
        module_name = f"mcp_servers.{domain}"
    
    spec = importlib.util.spec_from_file_location(module_name, server_file)
    module = importlib.util.module_from_spec(spec)
    
    # Register module so imports within it work
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # Look for class named {Domain}MCP
    class_name = f"{domain}MCP"
    if not hasattr(module, class_name):
        # Fallback: look for any class ending with MCP
        mcp_classes = [
            name for name in dir(module)
            if name.endswith("MCP") and isinstance(getattr(module, name), type)
        ]
        if not mcp_classes:
            raise AttributeError(
                f"No MCP class found in {server_file}. Expected '{class_name}'"
            )
        class_name = mcp_classes[0]
        logger.warning(f"Using fallback MCP class: {class_name} (expected {domain}MCP)")
    
    return getattr(module, class_name)


def load_tool_definitions(
    domain: str,
    version: str,
    tools_dir: Path,
    attributes_dir: Optional[Path] = None,
) -> Dict[str, ToolDefinition]:
    """Load tool definitions from JSON for a specific domain and version.
    
    Args:
        domain: Domain name.
        version: Version string (e.g., 'v1', 'base').
        tools_dir: Path to the tools directory (contains {domain}/{version}_tools.json).
        attributes_dir: Optional path to tool_attributes dir (contains {domain}/{version}.json).
    
    Returns:
        Dict mapping tool function name to ToolDefinition.
    """
    tools_file = tools_dir / domain / f"{version}_tools.json"
    if not tools_file.exists():
        raise FileNotFoundError(f"Tools file not found: {tools_file}")
    
    with tools_file.open("r", encoding="utf-8") as f:
        tools_data = json.load(f)
    
    # Load attributes if available
    attributes = {}
    if attributes_dir:
        attr_file = attributes_dir / domain / f"{version}.json"
        if attr_file.exists():
            with attr_file.open("r", encoding="utf-8") as f:
                attributes = json.load(f)
    
    # Build ToolDefinition objects
    tool_defs = {}
    for item in tools_data:
        node_id = item["node_id"]
        tool_schema = item["tool"]  # {"type": "function", "function": {...}}
        func_name = tool_schema["function"]["name"]
        
        tool_def = ToolDefinition(
            name=func_name,
            schema=tool_schema,
            node_id=node_id,
            domain=domain,
            attributes=attributes.get(func_name, {}),
        )
        tool_defs[func_name] = tool_def
    
    return tool_defs


def _patch_database_path(server_instance: Any, data_dir: Path) -> None:
    """Bind the server database to the explicitly selected data directory.
    
    The MCP server files reference '../database/outputs/' but in dtau_bench
    the actual structure is 'database/' (no 'outputs' layer).
    
    Since the server is already instantiated with _load_database called in __init__,
    reload from the selected root even if a legacy path happened to contain data.
    The dataset and its server modules are never modified.
    
    Args:
        server_instance: The instantiated MCP server.
        data_dir: The dtau_bench data directory root.
    """
    # The correct database root is data_dir/database (entities/ and relationships/ directly under it)
    correct_db_root = data_dir / "database"
    
    if not correct_db_root.exists():
        logger.warning(f"Database directory not found: {correct_db_root}")
        return
    
    if hasattr(server_instance, "base_database"):
        _reload_database(server_instance, correct_db_root)


def _bind_database_loader(server_class: type, data_dir: Path) -> bool:
    """Bind constructor and later reloads to the selected dataset directory."""
    if not callable(getattr(server_class, "_load_database", None)):
        return False
    db_root = data_dir / "database"

    def load_database(instance: Any, domain_name: str) -> None:
        # Dataset loaders update base_database in place, preserve sessions and
        # return None. Keep those semantics when redirecting their file reads.
        _reload_database(
            instance, db_root,
            domain_name=domain_name, preserve_session_state=True,
        )

    server_class._load_database = load_database
    return True


def _reload_database(
    server_instance: Any,
    db_root: Path,
    *,
    domain_name: Optional[str] = None,
    preserve_session_state: bool = False,
) -> None:
    """Reload database from the correct root path.
    
    Args:
        server_instance: The MCP server instance.
        db_root: The correct database root (containing entities/ and relationships/).
    """
    domain_name = server_instance.domain_name if domain_name is None else domain_name
    
    def load_json_file(file_path: str) -> Optional[List]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return [data]
        except (json.JSONDecodeError, IOError):
            pass
        return None
    
    base_database = {}
    
    # Step 1: Load global entities
    entities_dir = db_root / "entities"
    if entities_dir.exists():
        for entity_file in entities_dir.iterdir():
            if not entity_file.suffix == ".json":
                continue
            if entity_file.is_dir():
                continue
            entity_name = entity_file.stem
            data = load_json_file(str(entity_file))
            if data is not None:
                base_database[entity_name] = data
            else:
                base_database[entity_name] = []
    
    # Step 2: Load domain-specific entities (override)
    if domain_name:
        domain_entities_dir = entities_dir / domain_name
        if domain_entities_dir.exists() and domain_entities_dir.is_dir():
            for entity_file in domain_entities_dir.iterdir():
                if not entity_file.suffix == ".json":
                    continue
                entity_name = entity_file.stem
                data = load_json_file(str(entity_file))
                if data is not None:
                    base_database[entity_name] = data
    
    # Step 3: Load relationships
    relationships_dir = db_root / "relationships" / domain_name
    if relationships_dir.exists():
        for rel_file in relationships_dir.iterdir():
            if not rel_file.suffix == ".json":
                continue
            rel_name = rel_file.stem
            data = load_json_file(str(rel_file))
            if data is not None:
                base_database[rel_name] = data
            else:
                base_database[rel_name] = []
    
    if preserve_session_state:
        server_instance.base_database.update(base_database)
    else:
        server_instance.base_database = base_database
        # Clear any existing sessions when replacing a legacy database.
        server_instance.sessions = {}
    
    logger.info(
        f"Reloaded database for {domain_name}: "
        f"{len(base_database)} tables, "
        f"{sum(len(v) for v in base_database.values())} total records"
    )


# ─────────────────────────────────────────────────────────────────────────────
# High-level Setup Functions
# ─────────────────────────────────────────────────────────────────────────────

def setup_domain_tools(
    domain: str,
    version: str,
    data_dir: Path,
    inject_attributes: bool = True,
) -> MCPServerInstance:
    """Load MCP server and tool definitions for a single domain.
    
    This is the main entry point for single-domain rollout mode.
    
    Args:
        domain: Domain name (e.g., 'AthleteContentDashboard').
        version: Tool version (e.g., 'v1').
        data_dir: Root data directory (containing servers/, tools/, database/, meta/).
        inject_attributes: Whether to load tool attributes (cost/latency).
    
    Returns:
        Configured MCPServerInstance ready for tool invocation.
    """
    servers_dir = data_dir / "servers"
    tools_dir = data_dir / "tools"
    attributes_dir = data_dir / "meta" / "tool_attributes" if inject_attributes else None
    
    # Load tool definitions
    tool_defs = load_tool_definitions(domain, version, tools_dir, attributes_dir)
    logger.info(f"Loaded {len(tool_defs)} tool definitions for {domain}/{version}")
    
    # Load and instantiate MCP server – use versioned server for non-base versions
    # e.g., for version="v1", load from servers/v1/{domain}.py
    if version != "base":
        versioned_servers_dir = servers_dir / version
        versioned_file = versioned_servers_dir / f"{domain}.py"
        if not versioned_file.exists():
            raise FileNotFoundError(
                f"Versioned MCP server not found: {versioned_file}. "
                f"Run prepare_data.py to copy versioned servers from "
                f"mcp_servers_versions/{version}/"
            )
        server_class = load_mcp_server_class(domain, versioned_servers_dir)
        logger.info(f"Using versioned server: {versioned_file}")
    else:
        server_class = load_mcp_server_class(domain, servers_dir)
    database_bound = _bind_database_loader(server_class, data_dir)
    server_instance = server_class(domain_name=domain)
    
    # Fix database path if needed
    if not database_bound:
        _patch_database_path(server_instance, data_dir)
    
    return MCPServerInstance(domain=domain, server=server_instance, tools=tool_defs)


def setup_all_tools(
    version: str,
    data_dir: Path,
    domains: Optional[List[str]] = None,
    inject_attributes: bool = True,
) -> Tuple[Dict[str, MCPServerInstance], Dict[str, str]]:
    """Load MCP servers and tools for all domains (for all-tools mode).
    
    Args:
        version: Tool version to load.
        data_dir: Root data directory.
        domains: Optional list of domains to load. If None, discovers all.
        inject_attributes: Whether to load tool attributes.
    
    Returns:
        Tuple of:
        - Dict mapping domain name to MCPServerInstance
        - Dict mapping tool function name to domain name (for routing)
    
    Raises:
        ValueError: If duplicate tool function names found across domains.
    """
    tools_dir = data_dir / "tools"
    
    # Discover domains if not specified
    if domains is None:
        domains = [
            d.name for d in tools_dir.iterdir()
            if d.is_dir() and (d / f"{version}_tools.json").exists()
        ]
        domains.sort()
    
    logger.info(f"Setting up all-tools mode: {len(domains)} domains, version={version}")
    
    servers: Dict[str, MCPServerInstance] = {}
    tool_to_domain: Dict[str, str] = {}
    
    for domain in domains:
        server_instance = setup_domain_tools(domain, version, data_dir, inject_attributes)
        servers[domain] = server_instance
        
        # Build tool→domain mapping, checking for duplicates
        for tool_name in server_instance.get_tool_names():
            if tool_name in tool_to_domain:
                raise ValueError(
                    f"Duplicate tool function name '{tool_name}' found in "
                    f"domains '{tool_to_domain[tool_name]}' and '{domain}'"
                )
            tool_to_domain[tool_name] = domain
    
    total_tools = sum(len(s.tools) for s in servers.values())
    logger.info(f"All-tools setup complete: {total_tools} tools across {len(servers)} domains")
    
    return servers, tool_to_domain


def get_all_domains(data_dir: Path) -> List[str]:
    """Discover all available domain names from the tools directory.
    
    Args:
        data_dir: Root data directory.
    
    Returns:
        Sorted list of domain names.
    """
    tools_dir = data_dir / "tools"
    domains = [d.name for d in tools_dir.iterdir() if d.is_dir()]
    return sorted(domains)


# ─────────────────────────────────────────────────────────────────────────────
# Confirmation Tools Helper
# ─────────────────────────────────────────────────────────────────────────────

def get_confirmation_tools(
    domain: str,
    data_dir: Path,
    version: str = "base",
) -> FrozenSet[str]:
    """Return the set of tool names that require user confirmation.

    Reads from ``meta/tool_attributes/{domain}/{version}.json`` and
    collects every tool whose ``require_confirmation`` attribute is true.

    This is used by :class:`GoldReplayExecutor` to auto-inject
    ``confirm=True`` during gold trajectory replay.

    Args:
        domain: Domain name (e.g. ``'ArtistTeachingStudio'``).
        data_dir: Root data directory (containing ``meta/``).
        version: Attribute version file stem (default ``'base'``).

    Returns:
        A frozen set of tool function names.  Empty if the attribute
        file does not exist or contains no confirmation-required tools.
    """
    attr_file = data_dir / "meta" / "tool_attributes" / domain / f"{version}.json"
    if not attr_file.exists():
        return frozenset()

    try:
        with attr_file.open("r", encoding="utf-8") as f:
            attrs: Dict[str, Dict[str, Any]] = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load tool attributes %s: %s", attr_file, exc)
        return frozenset()

    return frozenset(
        name for name, meta in attrs.items()
        if meta.get("require_confirmation", False)
    )
