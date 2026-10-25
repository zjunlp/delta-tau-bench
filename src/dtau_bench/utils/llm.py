"""
LLM call wrapper using LiteLLM for unified model access.

Provides:
- LLMConfig: Configuration dataclass for model endpoints
- LLMClient: Async/sync client with retry, timeout, and structured output support
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import litellm
from litellm import completion, acompletion

from dtau_bench.utils.config import (
    STANDARD_MODEL_REQUEST_OPTION_KEYS,
    extract_model_request_options,
)

logger = logging.getLogger(__name__)

# Suppress verbose litellm logging
litellm.suppress_debug_info = True


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """Configuration for an LLM endpoint.
    
    Attributes:
        model_name: The model identifier (e.g., "DeepSeek-V3.2").
        base_url: API base URL.
        api_key: API key (use "EMPTY" for local deployments).
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        timeout: Request timeout in seconds.
        max_retries: Number of retries on failure.
        retry_delay: Delay between retries in seconds.
        supports_tools: Whether the model supports tool/function calling.
        supports_thinking: Whether the model supports thinking/reasoning mode.
    """
    model_name: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 180
    max_retries: int = 3
    retry_delay: float = 1.0
    supports_tools: bool = True
    supports_thinking: bool = False
    extra_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, model_key: str, models_config: Dict[str, Any], **overrides) -> "LLMConfig":
        """Create LLMConfig from models.yaml configuration.
        
        Args:
            model_key: Key in the models config (e.g., 'deepseek-v3').
            models_config: The loaded models.yaml dict.
            **overrides: Additional overrides (e.g., temperature).
        
        Returns:
            Configured LLMConfig instance.
        
        Raises:
            KeyError: If model_key not found in config.
        """
        models = models_config.get("models", {})
        if model_key not in models:
            available = list(models.keys())
            raise KeyError(f"Model '{model_key}' not found. Available: {available}")
        
        cfg = models[model_key]
        params = {
            "model_name": cfg["model_name"],
            "base_url": cfg.get("base_url", "http://localhost:8000/v1"),
            "api_key": cfg.get("api_key", "EMPTY"),
            "max_tokens": cfg.get("max_tokens", 4096),
            "supports_tools": cfg.get("supports_tools", True),
            "supports_thinking": cfg.get("supports_thinking", False),
            "extra_params": extract_model_request_options(cfg),
        }
        params.update(overrides)
        return cls(**params)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Client
# ─────────────────────────────────────────────────────────────────────────────

class LLMClient:
    """Unified LLM client with retry and structured output support.
    
    Uses LiteLLM as the backend to support various providers through
    a consistent OpenAI-compatible interface.
    
    Example:
        >>> config = LLMConfig(model_name="DeepSeek-V3.2", base_url="http://localhost:8000/v1")
        >>> client = LLMClient(config)
        >>> response = client.chat(messages=[{"role": "user", "content": "Hello"}])
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        # LiteLLM model string for custom endpoints: "openai/{model_name}"
        self._model_str = f"openai/{config.model_name}"

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> Any:
        """Synchronous chat completion with retry logic.
        
        Args:
            messages: Conversation messages in OpenAI format.
            tools: Optional list of tool definitions (OpenAI function calling format).
            tool_choice: Tool choice mode ('auto', 'none', 'required', or specific tool).
            temperature: Override config temperature.
            max_tokens: Override config max_tokens.
            stop: Stop sequences.
            **kwargs: Additional parameters passed to litellm.completion.
        
        Returns:
            LiteLLM response object (OpenAI-compatible).
        
        Raises:
            Exception: After all retries exhausted.
        """
        params = self._build_params(messages, tools, tool_choice, temperature, max_tokens, stop, **kwargs)
        
        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = completion(**params)
                return response
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * attempt
                    logger.warning(
                        f"LLM call failed (attempt {attempt}/{self.config.max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"LLM call failed after {self.config.max_retries} attempts: {e}")
        
        raise last_error

    async def achat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> Any:
        """Async chat completion with retry logic.
        
        Same interface as chat() but async. Use within async contexts
        for concurrent rollout processing.
        """
        import asyncio
        
        params = self._build_params(messages, tools, tool_choice, temperature, max_tokens, stop, **kwargs)
        
        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = await acompletion(**params)
                return response
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * attempt
                    logger.warning(
                        f"Async LLM call failed (attempt {attempt}/{self.config.max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Async LLM call failed after {self.config.max_retries} attempts: {e}")
        
        raise last_error

    def _build_params(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Union[str, Dict]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stop: Optional[List[str]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Build parameters dict for litellm call."""
        params: Dict[str, Any] = {
            "model": self._model_str,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "timeout": self.config.timeout,
            "api_base": self.config.base_url,
            "api_key": self.config.api_key,
        }

        # Reasoning models commonly use max_completion_tokens instead of the
        # legacy max_tokens.  Never send both: some OpenAI-compatible gateways
        # reject the request even though LiteLLM accepts both arguments.
        configured_extra = dict(self.config.extra_params)
        call_extra = dict(kwargs)
        if (
            "max_completion_tokens" not in configured_extra
            and "max_completion_tokens" not in call_extra
        ):
            params["max_tokens"] = (
                max_tokens if max_tokens is not None else self.config.max_tokens
            )
        
        if tools and self.config.supports_tools:
            params["tools"] = tools
            if tool_choice:
                params["tool_choice"] = tool_choice
        
        if stop:
            params["stop"] = stop
        
        # Extra model-specific params
        params.update(configured_extra)
        params.update(call_extra)

        # With an ``openai/{custom-model}`` route LiteLLM cannot infer the
        # gateway's model capabilities and may otherwise drop newer OpenAI
        # parameters.  Explicitly allow every configured standard parameter.
        configured_allowed = params.get("allowed_openai_params") or []
        if isinstance(configured_allowed, str):
            configured_allowed = [configured_allowed]
        passthrough = [
            name
            for name in STANDARD_MODEL_REQUEST_OPTION_KEYS
            if name in params and params[name] is not None
        ]
        if passthrough or configured_allowed:
            params["allowed_openai_params"] = list(
                dict.fromkeys([*configured_allowed, *passthrough])
            )
        
        return params

    def extract_content(self, response: Any) -> str:
        """Extract text content from LLM response.
        
        Args:
            response: LiteLLM response object.
        
        Returns:
            The assistant's text content (empty string if None).
        """
        message = self._extract_message(response)
        content = self._value(message, "content")
        return content if isinstance(content, str) else ""

    def extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from LLM response.
        
        Args:
            response: LiteLLM response object.
        
        Returns:
            List of tool call dicts with keys: id, function.name, function.arguments.
            Empty list if no tool calls.
        """
        message = self._extract_message(response)
        tool_calls = self._value(message, "tool_calls")
        if not tool_calls:
            return []
        
        result = []
        for tc in tool_calls:
            function = self._value(tc, "function")
            call = {
                "id": self._value(tc, "id"),
                "type": "function",
                "function": {
                    "name": self._value(function, "name"),
                    "arguments": self._value(function, "arguments"),
                },
            }
            result.append(call)
        return result

    @staticmethod
    def _value(payload: Any, name: str) -> Any:
        if isinstance(payload, dict):
            return payload.get(name)
        return getattr(payload, name, None)

    @classmethod
    def _extract_message(cls, response: Any) -> Any:
        choices = cls._value(response, "choices")
        if not isinstance(choices, (list, tuple)) or not choices:
            return None
        return cls._value(choices[0], "message")

    def extract_tool_calls_parsed(self, response: Any) -> List[Dict[str, Any]]:
        """Extract tool calls with parsed arguments (dict instead of JSON string).
        
        Returns:
            List of dicts with keys: id, name, arguments (as dict).
        """
        raw_calls = self.extract_tool_calls(response)
        parsed = []
        for call in raw_calls:
            try:
                args = json.loads(call["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {}
            parsed.append({
                "id": call["id"],
                "name": call["function"]["name"],
                "arguments": args,
            })
        return parsed

    @property
    def model_name(self) -> str:
        """The model name."""
        return self.config.model_name
    
    def __repr__(self) -> str:
        return f"LLMClient(model={self.config.model_name}, base_url={self.config.base_url})"
