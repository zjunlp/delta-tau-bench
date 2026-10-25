"""Regression tests for Steward model protocol and infrastructure setup."""

from __future__ import annotations

import copy
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from litellm import ModelResponse

from dtau_bench.explorer.agent import ExplorerAgent
from dtau_bench.explorer.prompts import FORCE_FINISH_MESSAGE
from dtau_bench.explorer.types import ExploreRequest, ExplorerConfig
from dtau_bench.skill.core.llm.client import LLM
from dtau_bench.strategy.base import StrategyRuntimeContext
from dtau_bench.strategy.steward import StewardStrategy
from dtau_bench.strategy.train_steward import TrainStewardStrategy
from dtau_bench.utils.config import extract_model_request_options
from dtau_bench.utils.llm import LLMClient, LLMConfig


_MISSING = object()


class _MinimalExplorerBackend:
    known_tool_names = set()
    explored_tool_names = []
    explored_skill_names = []

    def resolve_committed_skill(self, ref):
        return None


class _ScriptedExplorerClient(LLMClient):
    def __init__(self, responses):
        super().__init__(LLMConfig(model_name="test-model", max_retries=1))
        self.responses = list(responses)
        self.calls = []
        self.requests = []

    def chat(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        self.calls.append(copy.deepcopy(kwargs["messages"]))
        return self.responses.pop(0)


def _response_payload(action_name, arguments, reasoning=_MISSING):
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": f"call-{action_name}",
            "type": "function",
            "function": {
                "name": action_name,
                "arguments": json.dumps(arguments),
            },
        }],
    }
    if reasoning is not _MISSING:
        message["reasoning_content"] = reasoning
    return {
        "id": f"response-{action_name}",
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


class ExplorerReasoningReplayTests(unittest.TestCase):
    def test_second_turn_replays_reasoning_for_objects_and_dicts(self) -> None:
        cases = (
            ("text", "thinking", True),
            ("empty", "", True),
            ("none", None, False),
            ("missing", _MISSING, False),
        )
        for shape in ("object", "dict"):
            for label, value, expected_present in cases:
                with self.subTest(shape=shape, value=label):
                    first = _response_payload("note", {"text": "checked"}, value)
                    second = _response_payload(
                        "finish", {"report": "done", "commit_tools": []}
                    )
                    if shape == "object":
                        first = ModelResponse(**first)
                        second = ModelResponse(**second)

                    client = _ScriptedExplorerClient([first, second])
                    agent = ExplorerAgent(
                        backend=_MinimalExplorerBackend(),
                        llm_client=client,
                        config=ExplorerConfig(max_actions=2, max_llm_turns=2),
                    )
                    result = agent.run(
                        ExploreRequest(goal="g", missing_capability="m")
                    )

                    self.assertEqual("finish", result.stop_reason)
                    assistant = next(
                        item
                        for item in client.calls[1]
                        if item.get("role") == "assistant"
                    )
                    self.assertEqual(
                        expected_present, "reasoning_content" in assistant
                    )
                    if expected_present:
                        self.assertEqual(value, assistant["reasoning_content"])


class ExplorerForceFinishTests(unittest.TestCase):
    def test_force_finish_uses_only_finish_tool_without_tool_choice(self) -> None:
        no_action = {
            "id": "response-no-action",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        forced_finish = _response_payload(
            "finish",
            {
                "report": "forced report",
                "commit_tools": [],
                "confidence": "medium",
            },
        )
        client = _ScriptedExplorerClient([no_action, forced_finish])
        agent = ExplorerAgent(
            backend=_MinimalExplorerBackend(),
            llm_client=client,
            config=ExplorerConfig(max_actions=1, max_llm_turns=1),
        )

        result = agent.run(ExploreRequest(goal="g", missing_capability="m"))

        self.assertEqual("forced_finish:max_llm_turns", result.stop_reason)
        self.assertEqual("forced report", result.report_raw)
        self.assertEqual(2, len(client.requests))
        force_request = client.requests[1]
        self.assertNotIn("tool_choice", force_request)
        self.assertEqual(
            ["finish"],
            [tool["function"]["name"] for tool in force_request["tools"]],
        )
        self.assertEqual(
            FORCE_FINISH_MESSAGE,
            force_request["messages"][-1]["content"],
        )


class ModelRequestOptionTests(unittest.TestCase):
    def test_extracts_standard_options_without_endpoint_metadata(self) -> None:
        options = extract_model_request_options({
            "model_name": "custom-model",
            "base_url": "https://example.invalid/v1",
            "api_key": "do-not-forward",
            "supports_tools": True,
            "request_options": {"gateway_extension": "enabled", "top_p": 0.1},
            "top_p": 0.8,
            "reasoning_effort": "high",
            "max_completion_tokens": 2048,
        })
        self.assertEqual({
            "gateway_extension": "enabled",
            "top_p": 0.8,
            "reasoning_effort": "high",
            "max_completion_tokens": 2048,
        }, options)
        self.assertNotIn("api_key", options)
        self.assertNotIn("base_url", options)

    def test_litellm_client_forces_custom_gateway_passthrough(self) -> None:
        client = LLMClient(LLMConfig(
            model_name="future-model",
            max_tokens=4096,
            extra_params={
                "reasoning_effort": "high",
                "top_p": 0.7,
                "max_completion_tokens": 2048,
            },
        ))
        params = client._build_params(
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            tool_choice=None,
            temperature=None,
            max_tokens=None,
            stop=None,
        )
        self.assertEqual("high", params["reasoning_effort"])
        self.assertEqual(2048, params["max_completion_tokens"])
        self.assertNotIn("max_tokens", params)
        self.assertTrue({
            "reasoning_effort", "top_p", "max_completion_tokens"
        }.issubset(params["allowed_openai_params"]))

    def test_langchain_client_keeps_custom_gateway_options(self) -> None:
        with patch("dtau_bench.skill.core.llm.client.ChatOpenAI") as chat_cls:
            LLM(
                model="future-model",
                api_key="test-placeholder",
                base_url="https://example.invalid/v1",
                max_tokens=4096,
                request_options={
                    "reasoning_effort": "high",
                    "top_p": 0.6,
                    "max_completion_tokens": 1536,
                },
            )

        kwargs = chat_cls.call_args.kwargs
        self.assertEqual("high", kwargs["reasoning_effort"])
        self.assertEqual(0.6, kwargs["top_p"])
        self.assertEqual(1536, kwargs["max_completion_tokens"])
        self.assertNotIn("max_tokens", kwargs)
        self.assertEqual({}, kwargs["disabled_params"])

    def test_skill_and_explorer_clients_receive_model_options(self) -> None:
        request_options = {"reasoning_effort": "high", "top_p": 0.5}
        train = TrainStewardStrategy(config={
            "_llm_config": {"request_options": request_options}
        })
        self.assertEqual(request_options, train._build_llm_config().extra_params)

        steward = StewardStrategy(config={
            "explorer": {"enabled": True},
            "_explorer_llm_config": {"request_options": request_options},
        })
        self.assertEqual(
            request_options,
            steward._get_explorer_llm_client().config.extra_params,
        )


class StewardFailClosedTests(unittest.TestCase):
    _context = StrategyRuntimeContext(
        domain="test", version="v1", all_tools_mode=False
    )

    def test_train_steward_does_not_hide_retrieval_failure(self) -> None:
        runtime = SimpleNamespace(
            build_context_for_query=lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("retrieval unavailable")
            )
        )
        strategy = TrainStewardStrategy(config={})
        strategy.set_runtime(runtime)
        with self.assertRaisesRegex(RuntimeError, "retrieval unavailable"):
            strategy.start_session(
                {"id": "sample", "messages": [{"role": "user", "content": "x"}]},
                [],
                self._context,
            )

    def test_steward_requires_tool_index(self) -> None:
        strategy = StewardStrategy(config={"retrieval": {}})
        with self.assertRaisesRegex(ValueError, "retrieval.index_dir"):
            strategy._get_tool_rag_runtime()

    def test_shared_skill_runtime_initializes_once_under_concurrency(self) -> None:
        strategy = StewardStrategy(config={
            "skill_context": {
                "skill_library_dir": "/synthetic/library",
                "index_dir": "/synthetic/index",
            }
        })
        runtime = object()
        with patch(
            "dtau_bench.strategy.steward_base.SkillContextRuntime.from_config",
            return_value=runtime,
        ) as factory:
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(
                    lambda _: strategy._get_skill_context_runtime(), range(32)
                ))

        self.assertTrue(all(item is runtime for item in results))
        factory.assert_called_once()


if __name__ == "__main__":
    unittest.main()
