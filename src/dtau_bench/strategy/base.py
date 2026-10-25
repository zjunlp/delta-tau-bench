"""
Strategy base module - abstract base class and core data structures.

The strategy layer controls what tools and context an Agent sees at each turn.
It is a conversation-scoped controller that:
1. Initializes tool visibility at session start
2. Returns the Agent's view (visible tools + prepended messages) each turn
3. Intercepts virtual tool calls (e.g., retrieve_tools, retrieve_skills)
4. Receives turn-end callbacks for iterative strategies

Lifecycle:
  start_session() → [get_agent_view() → intercept_tool_calls() → on_turn_end()] × N turns
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class StrategyRuntimeContext:
    """Runtime context provided to strategy at session start.
    
    Attributes:
        domain: Current domain being processed.
        version: Tool schema version (e.g., 'v1', 'v2').
        all_tools_mode: Whether running in all-tools mode (cross-domain).
        tool_to_domain: Mapping from tool name to domain (for all-tools mode).
    """
    domain: str
    version: str
    all_tools_mode: bool
    tool_to_domain: Optional[Dict[str, str]] = None


@dataclass
class StrategyView:
    """The view that a strategy provides to the Agent for a given turn.
    
    Attributes:
        visible_tools_info: List of tool definitions in OpenAI format that Agent can use.
        prepended_messages: Messages to prepend before conversation (e.g., skill context).
        metadata: Debug/tracing metadata (not passed to LLM).
    """
    visible_tools_info: Optional[List[Dict[str, Any]]] = None
    prepended_messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategySessionState:
    """Mutable session state maintained across turns within a single conversation.
    
    Strategies should NOT store mutable state in self; instead, all state lives here.
    This makes strategies stateless (only config in self) and enables checkpoint/resume.
    
    Attributes:
        sample_id: The sample identifier.
        query_text: The initial user query (extracted from messages).
        visible_tool_names: List of currently visible tool names.
        visible_tools_info: Full OpenAI-format tool definitions for visible tools.
        prepended_messages: Messages to inject before conversation.
        retrieval_rounds: History of tool retrieval rounds (for RAG strategies).
        debug_history: Debug info per round (for tracing).
        custom_state: Extension point for strategy-specific state.
    """
    sample_id: str
    query_text: str
    visible_tool_names: List[str] = field(default_factory=list)
    visible_tools_info: Optional[List[Dict[str, Any]]] = None
    prepended_messages: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_rounds: List[Dict[str, Any]] = field(default_factory=list)
    debug_history: List[Dict[str, Any]] = field(default_factory=list)
    custom_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyUpdate:
    """Return value from on_turn_end indicating whether state changed.
    
    Attributes:
        state_changed: Whether the strategy's session state was mutated.
        metadata: Additional metadata about the turn-end processing.
    """
    state_changed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyToolInterceptionResult:
    """Result of strategy intercepting tool calls.
    
    When the Agent outputs tool_calls, the strategy can intercept some of them
    (e.g., virtual tools like 'retrieve_tools') and handle them internally.
    
    Attributes:
        handled_tool_responses: Tool response messages for intercepted calls (already handled).
        remaining_tool_calls: Tool calls that should be executed normally via ToolRouter.
        metadata: Debug/tracing metadata about the interception.
    """
    handled_tool_responses: List[Dict[str, Any]] = field(default_factory=list)
    remaining_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Abstract Base Class
# ============================================================================


class AgentStrategy(ABC):
    """Abstract base class for agent strategies.
    
    A strategy controls what tools the Agent sees and what context is injected
    into the conversation. It is conversation-scoped: one instance can serve
    multiple conversations (state is in StrategySessionState, not self).
    
    Subclasses must implement:
        - start_session(): Initialize session state
        - get_agent_view(): Return current turn's view for Agent
    
    Optionally override:
        - on_turn_end(): React to turn completion
        - intercept_tool_calls(): Handle virtual tool calls
        - get_artifacts(): Return strategy-specific output artifacts
    """

    def __init__(self, strategy_name: str, config: Dict[str, Any]):
        """
        Args:
            strategy_name: Unique name identifying this strategy.
            config: Strategy-specific configuration (from YAML).
        """
        self.strategy_name = strategy_name
        self.config = config

    @abstractmethod
    def start_session(
        self,
        sample: Dict[str, Any],
        all_tools_info: List[Dict[str, Any]],
        runtime_context: StrategyRuntimeContext,
    ) -> StrategySessionState:
        """Initialize a new conversation session.
        
        Called once at the beginning of each sample's processing.
        Should determine initial tool visibility and any prepended messages.
        
        Args:
            sample: The sample dict (contains id, messages, instruction, etc.)
            all_tools_info: All available tools in OpenAI format.
            runtime_context: Domain, version, and mode information.
            
        Returns:
            StrategySessionState with initial visibility set.
        """
        raise NotImplementedError

    @abstractmethod
    def get_agent_view(
        self,
        state: StrategySessionState,
        conversation: List[Dict[str, Any]],
    ) -> StrategyView:
        """Get the Agent's view for the current turn.
        
        Called before each Agent LLM call. Returns what tools the Agent can see
        and any messages to prepend to the conversation.
        
        Args:
            state: Current session state.
            conversation: Current conversation messages.
            
        Returns:
            StrategyView with visible tools and prepended messages.
        """
        raise NotImplementedError

    def on_turn_end(
        self,
        state: StrategySessionState,
        conversation: List[Dict[str, Any]],
        turn_artifacts: Dict[str, Any],
    ) -> StrategyUpdate:
        """Callback after a turn ends.
        
        Used by iterative strategies (e.g., iterative RAG) to update state
        based on the Agent's actions in this turn.
        
        Args:
            state: Current session state (may be mutated).
            conversation: Conversation messages after this turn.
            turn_artifacts: Information about what happened this turn.
            
        Returns:
            StrategyUpdate indicating if state changed.
        """
        return StrategyUpdate(state_changed=False, metadata={})

    def intercept_tool_calls(
        self,
        state: StrategySessionState,
        tool_calls: List[Dict[str, Any]],
        conversation: List[Dict[str, Any]],
    ) -> StrategyToolInterceptionResult:
        """Intercept and handle virtual tool calls.
        
        Called after Agent generates tool_calls but before execution.
        Strategies can intercept virtual tools (e.g., 'retrieve_tools') and
        return synthetic responses, passing through real tool calls.
        
        Default implementation: no interception (all calls pass through).
        
        Args:
            state: Current session state (may be mutated if interception occurs).
            tool_calls: List of tool call dicts from Agent's response.
            conversation: Current conversation messages.
            
        Returns:
            StrategyToolInterceptionResult with handled/remaining splits.
        """
        return StrategyToolInterceptionResult(
            handled_tool_responses=[],
            remaining_tool_calls=list(tool_calls),
            metadata={},
        )

    def get_artifacts(self, state: StrategySessionState) -> Dict[str, Any]:
        """Get strategy-produced artifacts for output.
        
        Called at the end of a session to collect strategy-specific data
        that should be included in the TrajectoryResult output (e.g., retrieval
        traces, skill selections, etc.)
        
        Args:
            state: Final session state.
            
        Returns:
            Dict of artifacts to include in output.
        """
        return {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.strategy_name!r})"
