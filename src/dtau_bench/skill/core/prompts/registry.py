"""Prompt registry for benchmark-specific prompts."""

from typing import Dict, Optional
from .plan_prompts import PLAN_EXTRACTION_PROMPTS, PLAN_COMBINE_PROMPTS
from .skill_prompts import SKILL_EXTRACTION_PROMPTS
from .filter_prompts import GENERAL_FILTER_PROMPTS, TOOL_FILTER_PROMPTS
from .merge_prompts import SKILL_MERGE_PROMPTS


class PromptRegistry:
    """
    Registry for benchmark-specific prompts.

    Supports fallback to default prompts when benchmark-specific ones are not available.
    """

    _prompts: Dict[str, Dict[str, str]] = {
        "plan_extraction": PLAN_EXTRACTION_PROMPTS,
        "plan_combine": PLAN_COMBINE_PROMPTS,
        "skill_extraction": SKILL_EXTRACTION_PROMPTS,
        "tool_summary": {
            "default": """You are an AI assistant specialized in analyzing agent trajectories.
Your task is to summarize a single interaction: based on the environment feedback from the current step, extract and summarize the key information in no more than 50 words.

# Inputs Description
1. The AI assistant's reasoning and action
2. The resulting environment feedback after the action

# Summary Guidelines
1. Summarize what the environment feedback conveys in light of the AI assistant's intent.
2. Preserve details that are tightly relevant to the intent verbatim when possible; compress other redundant information.
3. Summarize only factual content from the environment feedback—do not invent anything.
4. Write the summary in the tone of the environment feedback.

# Output Format
<feedback>
Your summary of the environment feedback
</feedback>
"""
        },
        "general_filter": GENERAL_FILTER_PROMPTS,
        "tool_filter": TOOL_FILTER_PROMPTS,
        "skill_merge": SKILL_MERGE_PROMPTS,
    }

    @classmethod
    def get(cls, prompt_type: str, benchmark: str = "default") -> str:
        """
        Get prompt for a specific type and benchmark.

        Args:
            prompt_type: Type of prompt (e.g., "plan_extraction", "skill_extraction")
            benchmark: Prompt variant name

        Returns:
            The prompt string, falling back to default if benchmark-specific not found
        """
        prompts = cls._prompts.get(prompt_type, {})
        return prompts.get(benchmark, prompts.get("default", ""))

    @classmethod
    def register(cls, prompt_type: str, benchmark: str, prompt: str) -> None:
        """
        Register a new prompt.

        Args:
            prompt_type: Type of prompt
            benchmark: Benchmark name
            prompt: The prompt string
        """
        if prompt_type not in cls._prompts:
            cls._prompts[prompt_type] = {}
        cls._prompts[prompt_type][benchmark] = prompt

    @classmethod
    def list_types(cls) -> list:
        """List all available prompt types."""
        return list(cls._prompts.keys())

    @classmethod
    def list_benchmarks(cls, prompt_type: str) -> list:
        """List all benchmarks for a prompt type."""
        return list(cls._prompts.get(prompt_type, {}).keys())
