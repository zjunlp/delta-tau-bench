"""
Pipeline for single-turn and multi-turn tool call rollout.

Integrates Agent, SimulatedUser, ToolRouter, Strategy, and CheckpointManager
for fault-tolerant trajectory generation.

Core loop:
1. Agent generates response (with tools)
2. If tool_calls present: strategy intercepts virtual tools, execute remaining
3. Repeat until no tool_calls (inner step loop)
4. SimulatedUser generates response (multi-turn only)
5. Check termination conditions
6. Repeat until conversation ends (outer turn loop)

Outputs TrajectoryResult conforming to the schema defined below.
"""

import copy
import json
import logging
import sys
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

from dtau_bench.rollout.agent import Agent
from dtau_bench.rollout.user import SimulatedUser, ConversationState
from dtau_bench.rollout.checkpoint import (
    CheckpointManager,
    CheckpointData,
    TrajectoryStatus,
    ErrorType,
)
from dtau_bench.tools.tool_router import ToolRouter, ToolCallResult
from dtau_bench.strategy.base import (
    AgentStrategy,
    StrategyRuntimeContext,
    StrategySessionState,
    StrategyView,
    StrategyToolInterceptionResult,
    StrategyUpdate,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def clear_reasoning_content(messages: List[Dict[str, Any]]) -> None:
    """
    Clear reasoning_content from all messages in the list (in-place).
    
    Used at turn boundaries to save context tokens following DeepSeek V3.2 conventions.
    The trajectory should be saved BEFORE calling this function.
    
    We completely remove the key (rather than setting it to None) so that
    downstream API proxies don't interpret it as "thinking mode message with
    missing reasoning_content".
    """
    for message in messages:
        if isinstance(message, dict):
            message.pop("reasoning_content", None)


def copy_messages_without_reasoning(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create a deep copy of messages with reasoning_content cleared."""
    copied = copy.deepcopy(messages)
    clear_reasoning_content(copied)
    return copied


# ============================================================================
# Console Printer for Verbose Output
# ============================================================================

class ConsolePrinter:
    """Pretty prints conversation messages to console during rollout."""

    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
    }

    def __init__(self, enabled: bool = True, use_colors: bool = True, max_content_length: int = 500):
        self.enabled = enabled
        self.use_colors = use_colors and sys.stdout.isatty()
        self.max_content_length = max_content_length

    def _color(self, text: str, color: str) -> str:
        if not self.use_colors:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"

    def _truncate(self, text: str) -> str:
        if self.max_content_length and len(text) > self.max_content_length:
            return text[:self.max_content_length] + "..."
        return text

    def print_turn_header(self, turn: int, step: int, sample_id: str = ""):
        if not self.enabled:
            return
        header = f"Turn {turn + 1} | Step {step + 1}"
        if sample_id:
            header += f" | ID: {sample_id[:12]}..."
        print(self._color(f"\n{'═' * 60}", "bold"))
        print(self._color(f"  {header}", "bold"))
        print(self._color(f"{'═' * 60}", "bold"))

    def print_agent_response(self, message: Dict[str, Any]):
        if not self.enabled:
            return
        print(f"\n🤖 {self._color('Assistant:', 'green')}")
        reasoning = message.get("reasoning_content")
        if reasoning:
            print(self._color("  💭 Thinking:", "magenta"))
            for line in self._truncate(reasoning).split("\n"):
                print(f"    {line}")
        content = message.get("content")
        if content:
            print(self._color("  Content:", "dim"))
            for line in self._truncate(content).split("\n"):
                print(f"    {line}")
        tool_calls = message.get("tool_calls")
        if tool_calls:
            print(self._color(f"  Tool Calls ({len(tool_calls)}):", "yellow"))
            for i, tc in enumerate(tool_calls):
                func = tc.get("function", {})
                name = func.get("name", "unknown")
                args = func.get("arguments", "{}")
                try:
                    if isinstance(args, str):
                        args_dict = json.loads(args)
                    else:
                        args_dict = args
                    args_str = json.dumps(args_dict, ensure_ascii=False, indent=2)
                except Exception:
                    args_str = str(args)
                print(self._color(f"    [{i+1}] {name}", "yellow"))
                args_preview = self._truncate(args_str.replace("\n", " "))
                print(self._color(f"        Args: {args_preview}", "dim"))

    def print_tool_response(self, message: Dict[str, Any]):
        if not self.enabled:
            return
        tool_call_id = message.get("tool_call_id", "unknown")[:12]
        content = message.get("content", "")
        print(f"\n🔧 {self._color(f'Tool Response (ID: {tool_call_id}...):', 'yellow')}")
        try:
            data = json.loads(content)
            formatted = json.dumps(data, ensure_ascii=False, indent=2)
            for line in self._truncate(formatted).split("\n")[:10]:
                print(f"    {line}")
        except Exception:
            print(f"    {self._truncate(content)}")

    def print_user_response(self, message: Dict[str, Any], is_stop: bool = False):
        if not self.enabled:
            return
        content = message.get("content", "")
        status = self._color(" [CONVERSATION ENDED]", "red") if is_stop else ""
        print(f"\n👤 {self._color('User:', 'cyan')}{status}")
        for line in self._truncate(content).split("\n"):
            print(f"    {line}")

    def print_summary(self, result: "TrajectoryResult"):
        if not self.enabled:
            return
        print(self._color(f"\n{'─' * 60}", "dim"))
        status = "✅ Success" if result.success else "❌ Failed"
        status_color = "green" if result.success else "red"
        print(self._color(f"  {status}", status_color))
        if result.error:
            print(self._color(f"  Error: {result.error}", "red"))
        role_counts = {}
        for msg in result.messages:
            role = msg.get("role", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
        counts_str = ", ".join(f"{r}: {c}" for r, c in role_counts.items())
        print(self._color(f"  Messages: {counts_str}", "dim"))
        print(self._color(f"{'─' * 60}\n", "dim"))


# ============================================================================
# Pipeline Configuration and Results
# ============================================================================

class PipelineMode(str, Enum):
    """Mode of operation for the pipeline."""
    POSITIVE = "positive"       # Normal tool calling
    WRONG_TOOL = "wrong_tool"   # Provide wrong tools (agent should refuse)
    NO_TOOL = "no_tool"         # No tools provided (agent should refuse)


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution.

    Attributes:
        max_turns: Max conversation turns (1 = single-turn).
        max_steps_per_turn: Max tool call steps per turn.
        mode: Pipeline mode (positive/wrong_tool/no_tool).
        timeout_per_step: Timeout for each step.
        use_checkpoints: Enable checkpoint saving.
        checkpoint_dir: Directory for checkpoint files.
        resume: Whether to resume from checkpoints.
        verbose: Print messages to console in real-time.
        current_domain: Current domain being evaluated (for all-tools mode).
        tool_to_domain: Mapping from tool name to domain.
    """
    max_turns: int = 20
    max_steps_per_turn: int = 10
    mode: PipelineMode = PipelineMode.POSITIVE
    timeout_per_step: float = 120.0
    use_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"
    resume: bool = True

    # Verbose output
    verbose: bool = False
    verbose_colors: bool = True
    verbose_max_length: int = 500

    # All-tools mode
    current_domain: Optional[str] = None
    tool_to_domain: Optional[Dict[str, str]] = None

    @property
    def is_multi_turn(self) -> bool:
        return self.max_turns > 1


@dataclass
class _TokenUsage:
    """Lightweight per-task token usage accumulator."""
    agent_prompt: int = 0
    agent_completion: int = 0
    agent_calls: int = 0
    user_prompt: int = 0
    user_completion: int = 0
    user_calls: int = 0

    def record_agent(self, usage: Optional[Dict[str, int]]) -> None:
        """Record token usage from an agent LLM call."""
        if not usage:
            return
        self.agent_prompt += usage.get("prompt_tokens", 0)
        self.agent_completion += usage.get("completion_tokens", 0)
        self.agent_calls += 1

    def record_user(self, usage: Optional[Dict[str, int]]) -> None:
        """Record token usage from a simulated-user LLM call."""
        if not usage:
            return
        self.user_prompt += usage.get("prompt_tokens", 0)
        self.user_completion += usage.get("completion_tokens", 0)
        self.user_calls += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dict suitable for JSON output."""
        return {
            "agent": {
                "prompt_tokens": self.agent_prompt,
                "completion_tokens": self.agent_completion,
                "total_tokens": self.agent_prompt + self.agent_completion,
                "calls": self.agent_calls,
            },
            "user": {
                "prompt_tokens": self.user_prompt,
                "completion_tokens": self.user_completion,
                "total_tokens": self.user_prompt + self.user_completion,
                "calls": self.user_calls,
            },
            "total_tokens": (
                self.agent_prompt + self.agent_completion
                + self.user_prompt + self.user_completion
            ),
        }

    @property
    def has_data(self) -> bool:
        return (self.agent_calls + self.user_calls) > 0


@dataclass
class TrajectoryResult:
    """Result of processing a single trajectory.
    
    Conforms to schema_version 2 output format (see the module docstring).
    """
    trajectory_id: str
    success: bool
    task: Dict[str, Any]
    agent_traj: List[Dict[str, Any]]
    conversation: Dict[str, Any]
    run: Dict[str, Any]
    artifacts: Dict[str, Any] = field(default_factory=dict)
    token_usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def messages(self) -> List[Dict[str, Any]]:
        return self.conversation.get("messages", [])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to output JSON format."""
        result = {
            "schema_version": 2,
            "id": self.trajectory_id,
            "success": self.success,
            "error": self.error,
            "task": self.task,
            "agent_traj": self.agent_traj,
            "conversation": self.conversation,
            "run": self.run,
            "artifacts": self.artifacts,
        }
        if self.token_usage:
            result["token_usage"] = self.token_usage
        return result


# ============================================================================
# Pipeline
# ============================================================================

class Pipeline:
    """
    Main pipeline for trajectory generation with fault tolerance.
    
    Supports both single-turn and multi-turn conversations with automatic
    checkpoint management.
    
    The pipeline integrates:
    - Agent (LLM-based assistant)
    - SimulatedUser (multi-turn customer simulator)
    - ToolRouter (routes tool calls to MCP servers)
    - AgentStrategy (controls tool visibility and intercepts, pluggable)
    - CheckpointManager (fault-tolerant persistence)
    
    Example:
        >>> from dtau_bench.rollout.pipeline import Pipeline, PipelineConfig
        >>> from dtau_bench.rollout.agent import Agent
        >>> from dtau_bench.rollout.user import SimulatedUser
        >>> from dtau_bench.tools.tool_router import ToolRouter
        >>> 
        >>> agent = Agent(model="openai/DeepSeek-V3.2", api_base="...", api_key="EMPTY")
        >>> user = SimulatedUser(model="openai/DeepSeek-V3.2", api_base="...", api_key="EMPTY")
        >>> router = ToolRouter.from_single_domain(server)
        >>> 
        >>> pipeline = Pipeline(agent=agent, user=user, tool_router=router,
        ...                     config=PipelineConfig(max_turns=20))
        >>> result = pipeline.process(sample)
    """

    def __init__(
        self,
        agent: Agent,
        tool_router: ToolRouter,
        user: Optional[SimulatedUser] = None,
        config: Optional[PipelineConfig] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        strategy: Optional[AgentStrategy] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            agent: The Agent instance for generating responses.
            tool_router: The ToolRouter for dispatching tool calls.
            user: Optional SimulatedUser for multi-turn conversations.
            config: Pipeline configuration.
            checkpoint_manager: Optional custom checkpoint manager.
            strategy: Optional AgentStrategy instance (controls tool visibility).
        """
        self.agent = agent
        self.tool_router = tool_router
        self.user = user
        self.config = config or PipelineConfig()
        self.strategy = strategy

        # Strategy session state (initialized per sample in process methods)
        self._strategy_state: Optional[StrategySessionState] = None

        # Console printer
        self.printer = ConsolePrinter(
            enabled=self.config.verbose,
            use_colors=self.config.verbose_colors,
            max_content_length=self.config.verbose_max_length,
        )

        # Checkpoint manager
        if self.config.use_checkpoints:
            self.checkpoint_manager = checkpoint_manager or CheckpointManager(
                checkpoint_dir=self.config.checkpoint_dir
            )
        else:
            self.checkpoint_manager = None

    # ──────────────────────────────────────────────────────────────────────
    # Helper methods
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_json_loads(content: Any) -> Optional[Any]:
        if not isinstance(content, str):
            return content
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return None

    @classmethod
    def _extract_tool_error(cls, content: Any) -> tuple:
        """Extract error info from tool result content.
        
        Returns:
            Tuple of (has_error, error_message, parsed_result).
        """
        parsed = cls._safe_json_loads(content)
        if isinstance(parsed, dict):
            error_msg = parsed.get("error") or parsed.get("message")
            if error_msg:
                return True, str(error_msg), parsed
        if isinstance(content, str) and content.startswith("Error:"):
            return True, content, parsed
        return False, None, parsed

    def _execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        session_id: str,
        max_workers: int = 4,
    ) -> List[Dict[str, Any]]:
        """Execute tool calls in parallel via ToolRouter.

        Returns list of tool response messages (role='tool').
        """
        responses = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for tc in tool_calls:
                function = tc.get("function", {})
                tool_name = function.get("name", "")
                arguments_raw = function.get("arguments", "{}")

                # Parse arguments
                if isinstance(arguments_raw, str):
                    try:
                        arguments = json.loads(arguments_raw)
                    except (json.JSONDecodeError, TypeError):
                        arguments = {}
                else:
                    arguments = arguments_raw if isinstance(arguments_raw, dict) else {}

                future = executor.submit(
                    self.tool_router.route,
                    tool_call_id=tc.get("id", "unknown"),
                    tool_name=tool_name,
                    arguments=arguments,
                    session_id=session_id,
                )
                futures[future] = tc

            for future in as_completed(futures):
                tc = futures[future]
                try:
                    result: ToolCallResult = future.result()
                    responses.append(result.to_message())
                except Exception as e:
                    responses.append({
                        "role": "tool",
                        "content": json.dumps({"error": str(e)}, ensure_ascii=False),
                        "tool_call_id": tc.get("id", "unknown"),
                    })

        # Reorder responses to match input order
        id_to_response = {r["tool_call_id"]: r for r in responses}
        ordered = []
        for tc in tool_calls:
            tc_id = tc.get("id", "unknown")
            if tc_id in id_to_response:
                ordered.append(id_to_response[tc_id])
        return ordered

    def _init_strategy_session(
        self,
        sample: Dict[str, Any],
        tools_info: List[Dict[str, Any]],
    ) -> None:
        """Initialize strategy session state for a new sample.
        
        Should be called once at the start of processing each sample.
        If no strategy is set, _strategy_state remains None.
        """
        if self.strategy is None:
            self._strategy_state = None
            return

        # Build runtime context from agent's runtime options
        runtime_options = self.agent.get_runtime_options()
        evaluation_context = runtime_options.get("evaluation_context", {})
        runtime_context = StrategyRuntimeContext(
            domain=evaluation_context.get("domain", sample.get("domain", "")),
            version=evaluation_context.get("version", ""),
            all_tools_mode=evaluation_context.get("all_tools_mode", False),
            tool_to_domain=self.config.tool_to_domain,
        )

        self._strategy_state = self.strategy.start_session(
            sample=sample,
            all_tools_info=tools_info,
            runtime_context=runtime_context,
        )
        logger.debug(
            f"[{sample.get('id', '')}] Strategy session started: "
            f"{self.strategy.strategy_name}, "
            f"visible_tools={len(self._strategy_state.visible_tool_names)}"
        )

    def _get_strategy_view(
        self,
        sample: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tools_info: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Get strategy view (visible tools + prepended messages).
        
        If no strategy is set, returns pass-through (all tools visible, no prepended messages).
        If strategy is set, delegates to strategy.get_agent_view().
        """
        if self.strategy is None or self._strategy_state is None:
            return {
                "visible_tools_info": tools_info,
                "prepended_messages": [],
                "metadata": {},
            }

        view: StrategyView = self.strategy.get_agent_view(
            state=self._strategy_state,
            conversation=messages,
        )
        return {
            "visible_tools_info": view.visible_tools_info if view.visible_tools_info is not None else tools_info,
            "prepended_messages": view.prepended_messages,
            "metadata": view.metadata,
        }

    def _intercept_strategy_tool_calls(
        self,
        sample: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Intercept tool calls via strategy (e.g., for virtual tools like retrieve_tools).
        
        Returns dict with:
        - handled_tool_responses: List of tool messages already handled by strategy
        - remaining_tool_calls: List of tool calls to execute normally
        - metadata: Any metadata from the strategy
        """
        if self.strategy is None or self._strategy_state is None:
            return {
                "handled_tool_responses": [],
                "remaining_tool_calls": list(tool_calls),
                "metadata": {},
            }

        result: StrategyToolInterceptionResult = self.strategy.intercept_tool_calls(
            state=self._strategy_state,
            tool_calls=tool_calls,
            conversation=messages,
        )
        return {
            "handled_tool_responses": result.handled_tool_responses,
            "remaining_tool_calls": result.remaining_tool_calls,
            "metadata": result.metadata,
        }

    def _notify_strategy_turn_end(
        self,
        sample: Dict[str, Any],
        messages: List[Dict[str, Any]],
        turn_artifacts: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Notify strategy that a turn ended. Returns metadata dict."""
        if self.strategy is None or self._strategy_state is None:
            return {}

        update: StrategyUpdate = self.strategy.on_turn_end(
            state=self._strategy_state,
            conversation=messages,
            turn_artifacts=turn_artifacts,
        )
        return update.metadata

    def _get_strategy_artifacts(self) -> Dict[str, Any]:
        """Collect strategy artifacts at the end of a session."""
        if self.strategy is None or self._strategy_state is None:
            return {}
        return self.strategy.get_artifacts(self._strategy_state)

    @staticmethod
    def _extract_task_block(sample: Dict[str, Any], version: str = "") -> Dict[str, Any]:
        """Extract the 'task' block for output from a sample dict."""
        # Support both TaskData.to_dict() format and raw sample format
        gold_traj = sample.get("gold_trajectory", {})
        if isinstance(gold_traj, dict):
            # Preserve the full gold_trajectory dict (including step_param_codes,
            # entity_instances, value_domain_samples) for the evaluator
            gold_trajectory_output = gold_traj
            trajectory_hash = gold_traj.get("trajectory_hash", "") or sample.get("trajectory_hash", "")
        else:
            gold_trajectory_output = gold_traj  # Already a list
            trajectory_hash = sample.get("trajectory_hash", "")

        # Extract task_info fields for domain/instruction
        task_info = sample.get("task_info", {})

        return {
            "instruction": sample.get("instruction", "") or task_info.get("instruction", ""),
            "domains": sample.get("domains", []) or ([task_info.get("domain")] if task_info.get("domain") else []),
            "domain_key": sample.get("domain", "") or task_info.get("domain", ""),
            "gold_trajectory": gold_trajectory_output,
            "trajectory_hash": trajectory_hash,
            "combo_id": sample.get("combo_id", "") or task_info.get("combo_id", ""),
            "query_id": sample.get("id", sample.get("query_id", "")),
            "version": version,
            "split": sample.get("split", ""),
        }

    def _build_run_block(
        self,
        sample: Dict[str, Any],
        run_updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the 'run' block for output."""
        runtime_options = self.agent.get_runtime_options()
        strategy_name = runtime_options.get("agent_strategy", "single_domain_full")
        evaluation_context = runtime_options.get("evaluation_context", {})

        run = {
            "stopped": False,
            "transferred": False,
            "out_of_scope": False,
            "reached_max_turns": False,
            "agent_strategy": strategy_name,
            "version": evaluation_context.get("version", ""),
            "split": evaluation_context.get("split", ""),
            "agent_model": self.agent.config.model,
            "user_model": self.user.model if self.user is not None else None,
            "all_tools_mode": evaluation_context.get("all_tools_mode", False),
            "all_tools_count": self.tool_router.total_tools,
            "run_name": runtime_options.get("run_name", ""),
            "skill_library_path": runtime_options.get("skill_library_path"),
        }
        run.update(run_updates)
        return run

    @staticmethod
    def _build_conversation_block(
        messages: List[Dict[str, Any]],
        total_turns: int,
        has_reasoning: bool,
    ) -> Dict[str, Any]:
        """Build the 'conversation' block for output."""
        assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
        tool_count = sum(1 for m in messages if m.get("role") == "tool")
        return {
            "messages": messages,
            "total_turns": total_turns,
            "has_reasoning": has_reasoning,
            "assistant_message_count": assistant_count,
            "tool_message_count": tool_count,
        }

    def _build_result(
        self,
        sample: Dict[str, Any],
        messages: List[Dict[str, Any]],
        agent_traj: List[Dict[str, Any]],
        total_turns: int,
        has_reasoning: bool,
        run_updates: Dict[str, Any],
        artifacts: Optional[Dict[str, Any]] = None,
        token_tracker: Optional[_TokenUsage] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> TrajectoryResult:
        """Build a TrajectoryResult from components."""
        version = self.agent.get_runtime_options().get("evaluation_context", {}).get("version", "")
        return TrajectoryResult(
            trajectory_id=sample.get("id", sample.get("query_id", "")),
            success=success,
            task=self._extract_task_block(sample, version),
            agent_traj=agent_traj,
            conversation=self._build_conversation_block(messages, total_turns, has_reasoning),
            run=self._build_run_block(sample, run_updates),
            artifacts=artifacts or {},
            token_usage=token_tracker.to_dict() if token_tracker and token_tracker.has_data else None,
            error=error,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Main Entry Points
    # ──────────────────────────────────────────────────────────────────────

    def process(
        self,
        sample: Dict[str, Any],
        tools_info: Optional[List[Dict[str, Any]]] = None,
    ) -> TrajectoryResult:
        """
        Process a sample (auto-detects single vs multi-turn).
        
        Args:
            sample: Sample dict with at minimum:
                - id: Sample identifier
                - messages: Initial conversation messages
                - user_system_prompt: (multi-turn) Scenario instructions for user
            tools_info: Optional tools info override (OpenAI tool format list).
            
        Returns:
            TrajectoryResult.
        """
        if self.config.is_multi_turn:
            return self.process_multi_turn(sample, tools_info)
        else:
            return self.process_single_turn(sample, tools_info)

    def process_single_turn(
        self,
        sample: Dict[str, Any],
        tools_info: Optional[List[Dict[str, Any]]] = None,
    ) -> TrajectoryResult:
        """Process a single-turn trajectory."""
        sample_id = sample.get("id", sample.get("query_id", ""))
        messages = copy.deepcopy(sample.get("messages", []))
        tools_info = tools_info or self.tool_router.get_all_openai_tools()

        # Handle modes
        if self.config.mode == PipelineMode.NO_TOOL:
            tools_info = None
        elif self.config.mode == PipelineMode.WRONG_TOOL:
            tools_info = sample.get("wrong_tools", tools_info)

        # Initialize strategy session
        self._init_strategy_session(sample, tools_info or [])

        try:
            if self.checkpoint_manager and self.config.use_checkpoints:
                with self.checkpoint_manager.checkpoint_context(
                    sample_id, initial_messages=messages
                ) as cp:
                    result = self._run_single_turn_loop(
                        sample, sample_id, cp.messages, tools_info, cp
                    )
                    if result.success:
                        self.checkpoint_manager.mark_completed(sample_id)
                    return result
            else:
                return self._run_single_turn_loop(
                    sample, sample_id, messages, tools_info, None
                )
        except Exception as e:
            logger.error(f"Single-turn error for {sample_id}: {e}")
            return self._build_result(
                sample=sample, messages=messages, agent_traj=[],
                total_turns=1, has_reasoning=False, run_updates={},
                success=False, error=str(e),
            )

    def process_multi_turn(
        self,
        sample: Dict[str, Any],
        tools_info: Optional[List[Dict[str, Any]]] = None,
    ) -> TrajectoryResult:
        """Process a multi-turn trajectory."""
        if self.user is None:
            raise ValueError("SimulatedUser required for multi-turn processing")

        sample_id = sample.get("id", sample.get("query_id", ""))
        messages = copy.deepcopy(sample.get("messages", []))
        tools_info = tools_info or self.tool_router.get_all_openai_tools()
        user_system_prompt = sample.get("user_system_prompt", "")

        # Initialize strategy session
        self._init_strategy_session(sample, tools_info)

        # Register user scenario
        self.user.set_scenario(sample_id, user_system_prompt)

        # Check for checkpoint
        start_turn = 0
        start_step = 0
        if self.config.resume and self.checkpoint_manager and self.checkpoint_manager.can_resume(sample_id):
            checkpoint = self.checkpoint_manager.load(sample_id)
            if checkpoint:
                messages = checkpoint.messages
                start_turn = checkpoint.current_turn
                start_step = checkpoint.current_step
                logger.info(f"Resuming {sample_id} from turn {start_turn}, step {start_step}")

        try:
            if self.checkpoint_manager and self.config.use_checkpoints:
                with self.checkpoint_manager.checkpoint_context(
                    sample_id, initial_messages=messages
                ) as cp:
                    cp.user_system_prompt = user_system_prompt
                    result = self._run_multi_turn_loop(
                        sample, sample_id, cp.messages, tools_info,
                        cp, start_turn, start_step,
                    )
                    if result.success:
                        self.checkpoint_manager.mark_completed(sample_id)
                    return result
            else:
                return self._run_multi_turn_loop(
                    sample, sample_id, messages, tools_info,
                    None, start_turn, start_step,
                )
        except Exception as e:
            import traceback
            logger.error(f"Multi-turn error for {sample_id}: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return self._build_result(
                sample=sample, messages=messages, agent_traj=[],
                total_turns=0, has_reasoning=False, run_updates={},
                success=False, error=str(e),
            )

    # ──────────────────────────────────────────────────────────────────────
    # Core Loops
    # ──────────────────────────────────────────────────────────────────────

    def _run_single_turn_loop(
        self,
        sample: Dict[str, Any],
        sample_id: str,
        messages: List[Dict[str, Any]],
        tools_info: Optional[List[Dict[str, Any]]],
        checkpoint: Optional[CheckpointData],
    ) -> TrajectoryResult:
        """Inner loop for single-turn processing."""
        start_step = checkpoint.current_step if checkpoint else 0
        artifacts: Dict[str, Any] = {}
        agent_traj: List[Dict[str, Any]] = []
        assistant_message_index = -1
        token_tracker = _TokenUsage()

        for step in range(start_step, self.config.max_steps_per_turn):
            if checkpoint:
                checkpoint.current_step = step

            self.printer.print_turn_header(0, step, sample_id)

            # Get strategy view
            strategy_view = self._get_strategy_view(sample, messages, tools_info or [])
            visible_tools_info = strategy_view["visible_tools_info"]
            agent_messages = list(strategy_view["prepended_messages"]) + messages

            # Agent generates response
            agent_msg = self.agent.generate(agent_messages, visible_tools_info)
            token_tracker.record_agent(agent_msg.pop("_usage", None))
            assistant_message_index += 1
            messages.append(agent_msg)
            self.printer.print_agent_response(agent_msg)

            if checkpoint:
                checkpoint.messages = messages

            # Check for tool calls
            tool_calls = agent_msg.get("tool_calls")
            if not tool_calls:
                break

            # Strategy intercept + execute
            interception = self._intercept_strategy_tool_calls(sample, messages, tool_calls)
            handled_responses = list(interception["handled_tool_responses"])
            remaining_tool_calls = list(interception["remaining_tool_calls"])
            executed_responses = (
                self._execute_tool_calls(remaining_tool_calls, sample_id)
                if remaining_tool_calls else []
            )
            tool_responses = handled_responses + executed_responses
            messages.extend(tool_responses)

            # Build agent_traj entries
            response_by_id = {r.get("tool_call_id"): r for r in tool_responses}
            remaining_ids = {tc.get("id") for tc in remaining_tool_calls}
            for tc in tool_calls:
                if tc.get("id") not in remaining_ids:
                    continue
                function = tc.get("function", {})
                raw_args = function.get("arguments", {})
                parsed_args = self._safe_json_loads(raw_args)
                if not isinstance(parsed_args, dict):
                    parsed_args = {}
                response = response_by_id.get(tc.get("id"), {})
                content = response.get("content")
                has_error, error_message, parsed_result = self._extract_tool_error(content)
                agent_traj.append({
                    "step_index": len(agent_traj),
                    "turn_index": 0,
                    "assistant_message_index": assistant_message_index,
                    "tool_call_id": tc.get("id"),
                    "tool_name": function.get("name", ""),
                    "arguments": parsed_args,
                    "tool_result": parsed_result,
                    "tool_result_raw": content,
                    "has_error": has_error,
                    "error_message": error_message,
                })

            for resp in tool_responses:
                self.printer.print_tool_response(resp)

            if checkpoint:
                checkpoint.messages = messages

        # Collect strategy artifacts
        strategy_artifacts = self._get_strategy_artifacts()
        if strategy_artifacts:
            artifacts.update(strategy_artifacts)

        result = self._build_result(
            sample=sample, messages=messages, agent_traj=agent_traj,
            total_turns=1,
            has_reasoning=any(m.get("reasoning_content") for m in messages if isinstance(m, dict)),
            run_updates={}, artifacts=artifacts,
            token_tracker=token_tracker, success=True,
        )
        self.printer.print_summary(result)
        return result

    def _run_multi_turn_loop(
        self,
        sample: Dict[str, Any],
        sample_id: str,
        messages: List[Dict[str, Any]],
        tools_info: Optional[List[Dict[str, Any]]],
        checkpoint: Optional[CheckpointData],
        start_turn: int,
        start_step: int,
    ) -> TrajectoryResult:
        """
        Inner loop for multi-turn processing.
        
        Handles DeepSeek V3.2 reasoning_content:
        - Preserved in trajectory for saving
        - Cleared at turn boundaries to save context tokens
        - Uses separate trajectory_messages only when reasoning_content is detected
        """
        has_reasoning = False
        trajectory_messages = None  # Lazy init only if needed
        artifacts: Dict[str, Any] = {}
        agent_traj: List[Dict[str, Any]] = []
        assistant_message_index = -1
        token_tracker = _TokenUsage()

        for turn in range(start_turn, self.config.max_turns):
            step_start = start_step if turn == start_turn else 0

            if checkpoint:
                checkpoint.current_turn = turn

            for step in range(step_start, self.config.max_steps_per_turn):
                if checkpoint:
                    checkpoint.current_step = step

                self.printer.print_turn_header(turn, step, sample_id)

                # Get strategy view
                strategy_view = self._get_strategy_view(sample, messages, tools_info or [])
                visible_tools_info = strategy_view["visible_tools_info"]
                agent_messages = list(strategy_view["prepended_messages"]) + messages

                # Agent generates response
                agent_msg = self.agent.generate(agent_messages, visible_tools_info)
                token_tracker.record_agent(agent_msg.pop("_usage", None))
                assistant_message_index += 1

                # Check for reasoning_content (DeepSeek V3.2)
                if agent_msg.get("reasoning_content"):
                    if not has_reasoning:
                        has_reasoning = True
                        trajectory_messages = copy.deepcopy(messages)
                        logger.debug(f"[{sample_id}] Detected reasoning_content")

                messages.append(agent_msg)
                if trajectory_messages is not None:
                    trajectory_messages.append(copy.deepcopy(agent_msg))

                self.printer.print_agent_response(agent_msg)

                if checkpoint:
                    checkpoint.messages = (
                        copy.deepcopy(trajectory_messages) if trajectory_messages else messages
                    )

                # Check for tool calls
                tool_calls = agent_msg.get("tool_calls")
                if not tool_calls:
                    break

                # Strategy intercept + execute
                interception = self._intercept_strategy_tool_calls(sample, messages, tool_calls)
                handled_responses = list(interception["handled_tool_responses"])
                remaining_tool_calls = list(interception["remaining_tool_calls"])
                executed_responses = (
                    self._execute_tool_calls(remaining_tool_calls, sample_id)
                    if remaining_tool_calls else []
                )
                tool_responses = handled_responses + executed_responses
                messages.extend(tool_responses)

                # Build agent_traj entries
                response_by_id = {r.get("tool_call_id"): r for r in tool_responses}
                remaining_ids = {tc.get("id") for tc in remaining_tool_calls}
                for tc in tool_calls:
                    if tc.get("id") not in remaining_ids:
                        continue
                    function = tc.get("function", {})
                    raw_args = function.get("arguments", {})
                    parsed_args = self._safe_json_loads(raw_args)
                    if not isinstance(parsed_args, dict):
                        parsed_args = {}
                    response = response_by_id.get(tc.get("id"), {})
                    content = response.get("content")
                    has_error, error_message, parsed_result = self._extract_tool_error(content)
                    agent_traj.append({
                        "step_index": len(agent_traj),
                        "turn_index": turn,
                        "assistant_message_index": assistant_message_index,
                        "tool_call_id": tc.get("id"),
                        "tool_name": function.get("name", ""),
                        "arguments": parsed_args,
                        "tool_result": parsed_result,
                        "tool_result_raw": content,
                        "has_error": has_error,
                        "error_message": error_message,
                    })

                if trajectory_messages is not None:
                    trajectory_messages.extend(copy.deepcopy(tool_responses))

                for resp in tool_responses:
                    self.printer.print_tool_response(resp)

                if checkpoint:
                    checkpoint.messages = (
                        copy.deepcopy(trajectory_messages) if trajectory_messages else messages
                    )

            # ─── Turn ended: User response ───
            message_count_before_user = len(messages)
            trajectory = {"id": sample_id, "messages": messages}
            state = self.user.generate(trajectory)
            token_tracker.record_user(state.usage)
            messages = state.messages
            new_user_message_added = (
                len(messages) > message_count_before_user
                and messages[-1].get("role") == "user"
            )

            # Sync trajectory_messages
            if trajectory_messages is not None and messages and messages[-1].get("role") == "user":
                trajectory_messages.append(copy.deepcopy(messages[-1]))

            # Verbose output
            if messages and messages[-1].get("role") == "user":
                self.printer.print_user_response(
                    messages[-1], is_stop=not state.should_continue
                )

            # Empty, non-terminal user output does not create a new user turn.
            # Keep the current turn's authentic reasoning for the next API call.
            if has_reasoning and new_user_message_added:
                clear_reasoning_content(messages)
                logger.debug(f"[{sample_id}] Turn {turn + 1} ended, cleared reasoning")

            if checkpoint:
                checkpoint.messages = (
                    copy.deepcopy(trajectory_messages) if trajectory_messages else messages
                )

            # Notify strategy
            strategy_metadata = self._notify_strategy_turn_end(
                sample=sample,
                messages=messages,
                turn_artifacts={"turn_index": turn, "should_end": not state.should_continue},
            )

            # Check termination
            should_end = not state.should_continue

            if should_end:
                # Collect strategy artifacts
                strategy_artifacts = self._get_strategy_artifacts()
                if strategy_artifacts:
                    artifacts.update(strategy_artifacts)

                final_messages = trajectory_messages if trajectory_messages else messages
                result = self._build_result(
                    sample=sample,
                    messages=final_messages,
                    agent_traj=agent_traj,
                    total_turns=turn + 1,
                    has_reasoning=has_reasoning,
                    run_updates={
                        "stopped": state.is_stopped,
                        "transferred": state.is_transferred,
                        "out_of_scope": state.is_out_of_scope,
                    },
                    artifacts=artifacts,
                    token_tracker=token_tracker,
                    success=True,
                )
                self.printer.print_summary(result)
                return result

        # Reached max turns - collect strategy artifacts
        strategy_artifacts = self._get_strategy_artifacts()
        if strategy_artifacts:
            artifacts.update(strategy_artifacts)

        final_messages = trajectory_messages if trajectory_messages else messages
        result = self._build_result(
            sample=sample,
            messages=final_messages,
            agent_traj=agent_traj,
            total_turns=self.config.max_turns,
            has_reasoning=has_reasoning,
            run_updates={"reached_max_turns": True},
            artifacts=artifacts,
            token_tracker=token_tracker,
            success=True,
        )
        self.printer.print_summary(result)
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Batch Processing
    # ──────────────────────────────────────────────────────────────────────

    def process_batch(
        self,
        samples: List[Dict[str, Any]],
        max_workers: int = 4,
        tools_info: Optional[List[Dict[str, Any]]] = None,
        progress_callback=None,
    ) -> List[TrajectoryResult]:
        """
        Process multiple samples in parallel.
        
        Args:
            samples: List of sample dictionaries.
            max_workers: Number of parallel workers.
            tools_info: Optional shared tools info.
            progress_callback: Optional callback(completed, total).
            
        Returns:
            List of TrajectoryResult objects.
        """
        tools_info = tools_info or self.tool_router.get_all_openai_tools()
        results = []
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process, sample, tools_info): sample
                for sample in samples
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    sample = futures[future]
                    results.append(self._build_result(
                        sample=sample,
                        messages=copy.deepcopy(sample.get("messages", [])),
                        agent_traj=[], total_turns=0, has_reasoning=False,
                        run_updates={}, success=False, error=str(e),
                    ))

                completed += 1
                if progress_callback:
                    progress_callback(completed, len(samples))

        return results

    def filter_completed(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out completed samples (for resume mode).
        
        Args:
            samples: List of all samples.
            
        Returns:
            List of samples that haven't been completed yet.
        """
        if not self.checkpoint_manager:
            return samples
        completed = set()
        for cp in self.checkpoint_manager.list_checkpoints(TrajectoryStatus.COMPLETED):
            completed.add(cp.trajectory_id)
        return [s for s in samples if s.get("id", s.get("query_id", "")) not in completed]
