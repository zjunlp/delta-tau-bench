"""
Agent class for rollout - generates assistant responses via LLM.

model access across providers (OpenAI-compatible endpoints, DeepSeek, etc.).

"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import litellm
from litellm import completion

from dtau_bench.rollout.reasoning import build_reasoning_control_extra_body
from dtau_bench.rollout.retry import RetryConfig, call_with_retry

# Suppress LiteLLM info logs
litellm.set_verbose = False


@dataclass
class AgentConfig:
    """Configuration for an Agent instance.
    
    Attributes:
        model: LiteLLM model identifier (e.g., "openai/DeepSeek-V3.2").
        system_prompt: Optional system prompt for the agent.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        api_key: API key for the endpoint.
        api_base: Custom API base URL.
        timeout: Request timeout in seconds.
        max_retries: Number of retries on failure.
        enable_thinking: Enable reasoning mode.
        runtime_options: Strategy hints and evaluation context (not forwarded to LLM).
        extra_params: Additional parameters passed to LiteLLM.
    """
    model: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout: int = 180
    max_retries: int = 3
    retry_initial_delay: float = 2.0
    retry_max_delay: float = 60.0
    retry_total_timeout: Optional[float] = 1200.0
    retry_jitter: bool = True
    enable_thinking: bool = False
    runtime_options: Dict[str, Any] = field(default_factory=dict)
    extra_params: Dict[str, Any] = field(default_factory=dict)


class Agent:
    """
    Agent for generating assistant responses in rollout.
    
    Wraps LiteLLM completion API with:
    - System prompt injection
    - Tool calling support
    - reasoning_content extraction
    - Strategy-managed tools override (_tools_info)
    
    Example:
        >>> from dtau_bench.rollout.agent import Agent
        >>> agent = Agent(
        ...     model="openai/DeepSeek-V3.2",
        ...     api_base="http://localhost:8000/v1",
        ...     api_key="EMPTY",
        ...     system_prompt="You are a helpful assistant."
        ... )
        >>> response = agent.generate(messages, tools)
    """

    def __init__(
        self,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: int = 180,
        max_retries: int = 3,
        retry_initial_delay: float = 2.0,
        retry_max_delay: float = 60.0,
        retry_total_timeout: Optional[float] = 1200.0,
        retry_jitter: bool = True,
        enable_thinking: bool = False,
        runtime_options: Optional[Dict[str, Any]] = None,
        **extra_params
    ):
        """Initialize an Agent.
        
        Args:
            model: LiteLLM model string (e.g., "openai/DeepSeek-V3.2").
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            api_key: API key (if None, uses env var).
            api_base: Custom API base URL.
            timeout: Request timeout in seconds.
            max_retries: Number of retries on failure.
            retry_initial_delay: Initial delay before retrying a transient failure.
            retry_max_delay: Maximum delay between transient-failure retries.
            retry_total_timeout: Maximum wall time spent retrying one LLM call.
            retry_jitter: Randomize retry sleep to avoid synchronized retry storms.
            enable_thinking: Enable reasoning mode.
            runtime_options: Agent runtime strategy hints.
            **extra_params: Additional parameters passed to LiteLLM.
        """
        self.config = AgentConfig(
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            retry_initial_delay=retry_initial_delay,
            retry_max_delay=retry_max_delay,
            retry_total_timeout=retry_total_timeout,
            retry_jitter=retry_jitter,
            enable_thinking=enable_thinking,
            runtime_options=runtime_options or {},
            extra_params=extra_params,
        )
        self.system_prompt = system_prompt

        # Tools info override (set by strategy layer for RAG/skill context modes)
        self._tools_info: Optional[List[Dict[str, Any]]] = None

    def set_system_prompt(self, system_prompt: str):
        """Update the system prompt."""
        self.system_prompt = system_prompt
        self.config.system_prompt = system_prompt

    def set_tools_info(self, tools_info: Optional[List[Dict[str, Any]]]):
        """Set tools info override (used by RAG strategy to limit visible tools).
        
        When set, generate() uses this instead of the tools argument.
        """
        self._tools_info = tools_info

    def clear_tools_info(self):
        """Clear the stored tools info, reverting to default behavior."""
        self._tools_info = None

    def get_runtime_options(self) -> Dict[str, Any]:
        """Get a shallow copy of runtime strategy options."""
        return dict(self.config.runtime_options)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            messages: Conversation history (list of message dicts).
            tools: Optional tool definitions in OpenAI format.
            tool_choice: Tool choice directive ("auto", "none", "required").
            
        Returns:
            Dict containing the assistant response:
            - role: "assistant"
            - content: Text content (may be None if tool_calls present)
            - tool_calls: List of tool call objects (if any)
            - reasoning_content: Thinking content (DeepSeek V3.2+, if present)
            
        Raises:
            litellm.exceptions.APIError: On API errors after retries exhausted.
        """
        # Prepare messages with system prompt
        full_messages = []
        if self.system_prompt:
            full_messages.append({"role": "system", "content": self.system_prompt})
        full_messages.extend(messages)

        # Build completion kwargs
        kwargs = {
            "model": self.config.model,
            "messages": full_messages,
            "temperature": self.config.temperature,
            "timeout": self.config.timeout,
            # dtau_bench owns retry/backoff so one logical attempt maps to one API call.
            "num_retries": 0,
            **self.config.extra_params,
        }

        if self.config.max_tokens:
            kwargs["max_tokens"] = self.config.max_tokens

        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base

        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key

        # Use stored tools_info if set (RAG mode), otherwise use the tools argument
        effective_tools = self._tools_info if self._tools_info is not None else tools
        if effective_tools:
            kwargs["tools"] = effective_tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        # A standard reasoning_effort setting takes precedence over legacy
        # provider-specific thinking flags; avoid sending both dialects.
        reasoning_extra_body = None
        if kwargs.get("reasoning_effort") is None:
            reasoning_extra_body = build_reasoning_control_extra_body(
                self.config.enable_thinking
            )
        else:
            # Custom gateways can support newer models than LiteLLM's registry.
            # Preserve the explicitly configured effort instead of dropping it.
            kwargs["allowed_openai_params"] = list(dict.fromkeys([
                *(kwargs.get("allowed_openai_params") or []), "reasoning_effort"
            ]))
        if reasoning_extra_body:
            kwargs["extra_body"] = {
                **kwargs.get("extra_body", {}),
                **reasoning_extra_body,
            }

        # Call LiteLLM with dtau_bench-managed retry/backoff.
        response = call_with_retry(
            lambda: completion(**kwargs),
            config=RetryConfig(
                max_attempts=self.config.max_retries,
                initial_delay=self.config.retry_initial_delay,
                max_delay=self.config.retry_max_delay,
                total_timeout=self.config.retry_total_timeout,
                jitter=self.config.retry_jitter,
            ),
            operation=f"agent LLM call ({self.config.model})",
        )

        # Extract and normalize the response
        choice = response.choices[0]
        message = choice.message

        result = {
            "role": "assistant",
            "content": message.content,
            "tool_calls": None,
        }

        # Extract token usage metadata (underscore prefix = not part of message)
        if hasattr(response, "usage") and response.usage:
            result["_usage"] = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
            }

        # Preserve returned reasoning strings, including an explicit empty string.
        if isinstance(getattr(message, "reasoning_content", None), str):
            result["reasoning_content"] = message.reasoning_content

        # Handle tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        return result

    def __repr__(self) -> str:
        return f"Agent(model={self.config.model!r})"
