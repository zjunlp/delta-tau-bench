"""
Tool call router - dispatches tool calls to the correct MCP server.

Provides:
- ToolRouter: Routes tool calls by function name to the appropriate MCPServerInstance
- Supports single-domain and all-tools modes
- Thread-safe for concurrent tool execution
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from dtau_bench.tools.mcp_wrapper import MCPServerInstance, ToolDefinition
from dtau_bench.utils.logging import get_logger

logger = get_logger("tools.router")


class ToolCallResult:
    """Result of a tool invocation.
    
    Attributes:
        tool_call_id: The original tool call ID from the LLM response.
        tool_name: The function name that was called.
        result: The execution result dict from the MCP server.
        domain: The domain the tool belongs to.
        success: Whether the invocation completed without error.
        error: Error message if invocation failed.
    """

    def __init__(
        self,
        tool_call_id: str,
        tool_name: str,
        result: Dict[str, Any],
        domain: str,
        success: bool = True,
        error: Optional[str] = None,
    ):
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.result = result
        self.domain = domain
        self.success = success
        self.error = error

    def to_message(self) -> Dict[str, Any]:
        """Convert to OpenAI tool message format for conversation history.
        
        Returns:
            Dict with role='tool', content, and tool_call_id.
        """
        if self.success:
            content = json.dumps(self.result, ensure_ascii=False)
        else:
            content = json.dumps({"error": self.error}, ensure_ascii=False)
        
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": content,
        }

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"ToolCallResult({status} {self.tool_name}, id={self.tool_call_id[:8]}...)"


class ToolRouter:
    """Routes tool calls to the appropriate MCP server.
    
    Supports two modes:
    1. Single-domain: All tools belong to one domain/server.
    2. All-tools: Tools from multiple domains, routed by function name.
    
    In all-tools mode, the router additionally validates that calls to tools
    outside the current domain return an error (simulating realistic tool access).
    
    Example:
        >>> router = ToolRouter.from_single_domain(server_instance)
        >>> result = router.route(tool_call_id, "get_athlete_profile", {"athlete_id": "a1"}, session_id)
        
        >>> router = ToolRouter.from_all_domains(servers_dict, tool_to_domain)
        >>> result = router.route(tool_call_id, "get_athlete_profile", {"athlete_id": "a1"}, session_id)
    """

    def __init__(
        self,
        servers: Dict[str, MCPServerInstance],
        tool_to_domain: Dict[str, str],
        current_domain: Optional[str] = None,
        enforce_domain: bool = False,
    ):
        """Initialize ToolRouter.
        
        Args:
            servers: Dict mapping domain name to MCPServerInstance.
            tool_to_domain: Dict mapping tool function name to domain name.
            current_domain: The "correct" domain for the current task (for validation).
            enforce_domain: If True, calls to tools outside current_domain return errors.
        """
        self.servers = servers
        self.tool_to_domain = tool_to_domain
        self.current_domain = current_domain
        self.enforce_domain = enforce_domain
        
        # Build function_name → MCPServerInstance index
        self._tool_index: Dict[str, MCPServerInstance] = {}
        for tool_name, domain in tool_to_domain.items():
            if domain in servers:
                self._tool_index[tool_name] = servers[domain]

    @classmethod
    def from_single_domain(
        cls,
        server: MCPServerInstance,
    ) -> "ToolRouter":
        """Create a router for single-domain mode.
        
        Args:
            server: The single MCPServerInstance.
        
        Returns:
            Configured ToolRouter.
        """
        servers = {server.domain: server}
        tool_to_domain = {name: server.domain for name in server.get_tool_names()}
        return cls(
            servers=servers,
            tool_to_domain=tool_to_domain,
            current_domain=server.domain,
            enforce_domain=False,
        )

    @classmethod
    def from_all_domains(
        cls,
        servers: Dict[str, MCPServerInstance],
        tool_to_domain: Dict[str, str],
        current_domain: Optional[str] = None,
        enforce_domain: bool = True,
    ) -> "ToolRouter":
        """Create a router for all-tools mode.
        
        Args:
            servers: Dict mapping domain to MCPServerInstance.
            tool_to_domain: Dict mapping function name to domain.
            current_domain: The correct domain for the task.
            enforce_domain: Whether to reject calls to wrong-domain tools.
        
        Returns:
            Configured ToolRouter.
        """
        return cls(
            servers=servers,
            tool_to_domain=tool_to_domain,
            current_domain=current_domain,
            enforce_domain=enforce_domain,
        )

    def route(
        self,
        tool_call_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        session_id: str,
    ) -> ToolCallResult:
        """Route and execute a tool call.
        
        Args:
            tool_call_id: The tool call ID from the LLM response.
            tool_name: The function name to call.
            arguments: The parsed arguments dict.
            session_id: Session ID for database isolation.
        
        Returns:
            ToolCallResult with the execution outcome.
        """
        # Check if tool exists
        if tool_name not in self._tool_index:
            return ToolCallResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                result={},
                domain="unknown",
                success=False,
                error=f"Tool '{tool_name}' not found. Available tools: check your tool list.",
            )
        
        # Domain enforcement in all-tools mode
        tool_domain = self.tool_to_domain[tool_name]
        if self.enforce_domain and self.current_domain and tool_domain != self.current_domain:
            return ToolCallResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                result={},
                domain=tool_domain,
                success=False,
                error=(
                    f"Access denied: tool '{tool_name}' belongs to domain '{tool_domain}', "
                    f"but current task domain is '{self.current_domain}'. "
                    f"You can only invoke tools from the current domain."
                ),
            )
        
        # Execute the tool
        server = self._tool_index[tool_name]
        try:
            result = server.invoke(session_id, tool_name, **arguments)
            
            # Check if result contains an error from the server
            if isinstance(result, dict) and "error" in result:
                return ToolCallResult(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    result=result,
                    domain=tool_domain,
                    success=False,
                    error=result["error"],
                )
            
            return ToolCallResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                result=result,
                domain=tool_domain,
                success=True,
            )
        except Exception as e:
            logger.error(f"Tool execution error: {tool_name} - {e}", exc_info=True)
            return ToolCallResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                result={},
                domain=tool_domain,
                success=False,
                error=f"Execution error: {str(e)}",
            )

    def route_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        session_id: str,
    ) -> List[ToolCallResult]:
        """Route multiple tool calls sequentially.
        
        For parallel execution, use route() with ThreadPoolExecutor externally.
        
        Args:
            tool_calls: List of tool call dicts with keys: id, name, arguments.
            session_id: Session ID for database isolation.
        
        Returns:
            List of ToolCallResult objects.
        """
        results = []
        for tc in tool_calls:
            result = self.route(
                tool_call_id=tc["id"],
                tool_name=tc["name"],
                arguments=tc.get("arguments", {}),
                session_id=session_id,
            )
            results.append(result)
        return results

    def get_all_tool_names(self) -> List[str]:
        """Get all available tool function names."""
        return list(self.tool_to_domain.keys())

    def get_all_openai_tools(self, inject_attributes: bool = True) -> List[Dict[str, Any]]:
        """Get all tool definitions in OpenAI format.
        
        Args:
            inject_attributes: Whether to inject cost/latency info.
        
        Returns:
            List of OpenAI tool format dicts.
        """
        tools = []
        for server in self.servers.values():
            tools.extend(server.get_openai_tools(inject_attributes))
        return tools

    def get_domain_tool_names(self, domain: str) -> List[str]:
        """Get tool names for a specific domain."""
        return [name for name, d in self.tool_to_domain.items() if d == domain]

    @property
    def total_tools(self) -> int:
        """Total number of tools across all domains."""
        return len(self.tool_to_domain)

    def __repr__(self) -> str:
        return (
            f"ToolRouter(domains={len(self.servers)}, tools={self.total_tools}, "
            f"current_domain={self.current_domain!r})"
        )
