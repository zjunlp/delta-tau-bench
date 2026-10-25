"""Offline regression tests for rollout message protocol handling."""

from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from dtau_bench.rollout.agent import Agent
from dtau_bench.rollout.pipeline import Pipeline, PipelineConfig
from dtau_bench.rollout.user import ConversationState


class AgentReasoningContentTests(unittest.TestCase):
    def test_generate_preserves_only_reasoning_strings_returned_by_provider(self) -> None:
        cases = (
            ("non-empty", "thinking", True),
            ("empty", "", True),
            ("none", None, False),
        )

        for label, provider_value, expected_present in cases:
            with self.subTest(label=label):
                message = SimpleNamespace(
                    content="answer",
                    tool_calls=None,
                    reasoning_content=provider_value,
                )
                response = SimpleNamespace(
                    choices=[SimpleNamespace(message=message)],
                    usage=None,
                )
                agent = Agent(model="test-model", max_retries=1)

                with patch("dtau_bench.rollout.agent.completion", return_value=response):
                    result = agent.generate([{"role": "user", "content": "hello"}])

                self.assertEqual(expected_present, "reasoning_content" in result)
                if expected_present:
                    self.assertEqual(provider_value, result["reasoning_content"])

    def test_generate_does_not_invent_missing_reasoning_content(self) -> None:
        message = SimpleNamespace(content="answer", tool_calls=None)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=None,
        )
        agent = Agent(model="test-model", max_retries=1)

        with patch("dtau_bench.rollout.agent.completion", return_value=response):
            result = agent.generate([{"role": "user", "content": "hello"}])

        self.assertNotIn("reasoning_content", result)


class _RecordingAgent:
    def __init__(self) -> None:
        self.calls = []
        self.config = SimpleNamespace(model="fake-agent")

    def get_runtime_options(self):
        return {}

    def generate(self, messages, tools):
        self.calls.append(copy.deepcopy(messages))
        call_number = len(self.calls)
        response = {
            "role": "assistant",
            "content": f"assistant response {call_number}",
            "tool_calls": None,
        }
        if call_number == 1:
            response["reasoning_content"] = "private reasoning from turn one"
        return response


class _FakeRouter:
    total_tools = 0

    def get_all_openai_tools(self):
        return []


class _ScriptedUser:
    model = "fake-user"

    def __init__(self, add_user_message: bool) -> None:
        self.add_user_message = add_user_message
        self.calls = 0

    def set_scenario(self, sample_id, instructions):
        pass

    def generate(self, trajectory):
        self.calls += 1
        messages = copy.deepcopy(trajectory["messages"])
        if self.calls == 1:
            if self.add_user_message:
                messages.append({"role": "user", "content": "one more detail"})
            return ConversationState(messages=messages)
        return ConversationState(messages=messages, is_stopped=True)


class MultiTurnReasoningProtocolTests(unittest.TestCase):
    def _run_pipeline(self, *, add_user_message: bool):
        agent = _RecordingAgent()
        pipeline = Pipeline(
            agent=agent,
            user=_ScriptedUser(add_user_message),
            tool_router=_FakeRouter(),
            config=PipelineConfig(
                max_turns=2,
                max_steps_per_turn=1,
                use_checkpoints=False,
                resume=False,
            ),
        )
        result = pipeline.process(
            {
                "id": "protocol-case",
                "messages": [{"role": "user", "content": "initial request"}],
                "user_system_prompt": "test scenario",
            }
        )
        self.assertTrue(result.success, result.error)
        self.assertEqual(2, len(agent.calls))
        return agent, result

    @staticmethod
    def _first_assistant(messages):
        return next(message for message in messages if message.get("role") == "assistant")

    def test_empty_simulated_user_turn_keeps_reasoning_for_next_agent_call(self) -> None:
        agent, _ = self._run_pipeline(add_user_message=False)

        prior_assistant = self._first_assistant(agent.calls[1])
        self.assertEqual(
            "private reasoning from turn one",
            prior_assistant.get("reasoning_content"),
        )

    def test_real_user_turn_clears_api_history_but_preserves_saved_trajectory(self) -> None:
        agent, result = self._run_pipeline(add_user_message=True)

        api_history_assistant = self._first_assistant(agent.calls[1])
        self.assertNotIn("reasoning_content", api_history_assistant)

        saved_assistant = self._first_assistant(result.messages)
        self.assertEqual(
            "private reasoning from turn one",
            saved_assistant.get("reasoning_content"),
        )


if __name__ == "__main__":
    unittest.main()
