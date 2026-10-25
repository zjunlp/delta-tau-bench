"""Plan extraction from trajectories."""

import re
import logging
from typing import Dict, Optional, List, Any
from collections import defaultdict

from .base import BasePlanExtractor
from ..prompts.registry import PromptRegistry

logger = logging.getLogger(__name__)


class PlanExtractor(BasePlanExtractor):
    """
    Extract reusable plans from successful trajectories.

    Plans are step-by-step instructions for completing tasks,
    derived from analyzing agent interaction histories.
    """

    def __init__(
        self,
        llm,
        benchmark: str = "atomic",
        max_retries: int = 5,
        verbose: bool = True,
        prompt_suffix: str = "",
    ):
        super().__init__(llm, benchmark, verbose=verbose)
        self.max_retries = max_retries
        self.prompt_suffix = prompt_suffix

    def get_prompt(self) -> str:
        """Get plan extraction prompt for benchmark."""
        return PromptRegistry.get("plan_extraction", self.benchmark) + self.prompt_suffix

    def _extract_plan_from_response(self, text: str) -> Optional[str]:
        """Extract plan from LLM response using <plan> tags."""
        match = re.search(r"<plan>(.*?)</plan>", text, flags=re.S)
        if match:
            return match.group(1).strip()
        if self.verbose:
            logger.warning("No <plan> block found in response")
        return None

    async def extract(
        self,
        item: Dict,
        **kwargs
    ) -> Optional[Dict]:
        """
        Extract plan from a trajectory item.

        Args:
            item: Dictionary with 'user_task' and 'task_history' keys

        Returns:
            Item with added 'plan' key, or None if extraction failed
        """
        user_task = item.get("user_task", "")
        trajectory = item.get("trajectory", [])

        messages = [
            ("system", self.get_prompt()),
            ("human", f"user task: {user_task}\n\nan agent's interaction history: {trajectory}")
        ]
        if isinstance(item.get("cost_summary"), dict):
            messages[1] = (
                "human",
                messages[1][1]
                + f"\n\nobserved action cost summary: {item['cost_summary']}"
                + f"\n\nobserved action costs: {item.get('observed_action_costs', [])}",
            )

        if self.verbose:
            logger.info(f"Extracting plan from task: {user_task[:50]}...")

        try:
            response = await self.llm.ainvoke(
                messages=messages,
                regex_extractor=self._extract_plan_from_response,
                **kwargs
            )

            plan = self._extract_plan_from_response(response)

            if plan:
                result = item.copy()
                result["plan"] = plan
                if self.verbose:
                    logger.info("Plan extraction successful")
                return result
            else:
                logger.error("Failed to extract plan from response")
                return None

        except Exception as e:
            logger.error(f"Error during plan extraction: {e}")
            return None

    async def extract_and_group(
        self,
        items: List[Dict],
        filter_threshold: float = 0.999,
        batch_size: int = 10,
        max_concurrent: int = 5,
        **kwargs
    ) -> Dict[str, List[Dict]]:
        """
        Extract plans and group by task.

        Args:
            items: List of trajectory items
            filter_threshold: Minimum reward threshold
            batch_size: Batch size for processing
            max_concurrent: Max concurrent batches

        Returns:
            Dictionary mapping user_task to list of extracted plans
        """
        # Filter by reward
        filtered = [
            item for item in items
            if item.get("reward", item.get("after_score", 0)) > filter_threshold
        ]

        logger.info(
            f"Filtered {len(filtered)}/{len(items)} items "
            f"(threshold: {filter_threshold})"
        )

        if not filtered:
            logger.warning("No items passed filter threshold")
            return {}

        # Extract plans
        results = await self.extract_batch(
            filtered,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            **kwargs
        )

        # Group by task
        grouped = defaultdict(list)
        for result in results:
            if result and "plan" in result:
                grouped[result["user_task"]].append(result)

        return dict(grouped)


class PlanCombiner:
    """
    Combine multiple plans for the same task into an optimal plan.

    Uses LLM to merge plans while preserving critical workflow branches.
    """

    def __init__(
        self,
        llm,
        benchmark: str = "atomic",
        max_retries: int = 5,
        verbose: bool = True,
        prompt_suffix: str = "",
    ):
        self.llm = llm
        self.benchmark = benchmark
        self.max_retries = max_retries
        self.verbose = verbose
        self.prompt_suffix = prompt_suffix

    def _extract_plan_from_response(self, text: str) -> Optional[str]:
        """Extract plan from LLM response."""
        match = re.search(r"<plan>(.*?)</plan>", text, flags=re.S)
        if match:
            return match.group(1).strip()
        if self.verbose:
            logger.warning("No <plan> block found in response")
        return None

    async def combine(
        self,
        user_task: str,
        plans_list: List[str],
        **kwargs
    ) -> Optional[Dict]:
        """
        Combine multiple plans into one optimal plan.

        Args:
            user_task: The user's task description
            plans_list: List of plan strings to combine

        Returns:
            Dictionary with 'task' and 'plan' keys
        """
        if len(plans_list) == 1:
            return {"task": user_task, "plan": plans_list[0]}

        # Format plans for prompt
        plans_text = ""
        for idx, plan in enumerate(plans_list):
            plans_text += f"plan {idx+1}:\n {plan}\n\n"

        prompt = PromptRegistry.get("plan_combine", self.benchmark) + self.prompt_suffix
        messages = [
            ("system", prompt),
            ("human", f"user task: {user_task}\n\nOther planning experts' plans: {plans_text}")
        ]

        if self.verbose:
            logger.info(f"Combining {len(plans_list)} plans for task: {user_task[:50]}...")

        try:
            response = await self.llm.ainvoke(
                messages=messages,
                regex_extractor=self._extract_plan_from_response,
                **kwargs
            )

            plan = self._extract_plan_from_response(response)

            if plan:
                if self.verbose:
                    logger.info("Plan combination successful")
                return {"task": user_task, "plan": plan}
            else:
                logger.error("Failed to extract combined plan")
                return None

        except Exception as e:
            logger.error(f"Error during plan combination: {e}")
            return None

    async def combine_grouped_plans(
        self,
        grouped_plans: Dict[str, List[Dict]],
        concurrent: bool = False,
        **kwargs
    ) -> Dict[str, str]:
        """
        Combine plans for all tasks.

        Args:
            grouped_plans: Dictionary mapping task to list of plan results
            concurrent: Whether to process tasks concurrently

        Returns:
            Dictionary mapping task to combined plan
        """
        plan_library = {}

        if concurrent:
            tasks = []
            task_keys = []
            for task, items in grouped_plans.items():
                plans = [item.get("plan", item) if isinstance(item, dict) else item
                         for item in items]
                tasks.append(self.combine(user_task=task, plans_list=plans, **kwargs))
                task_keys.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for task, result in zip(task_keys, results):
                if isinstance(result, Exception):
                    logger.error(f"Error combining plans for {task}: {result}")
                elif result:
                    plan_library[result["task"]] = result["plan"]
        else:
            if self.verbose:
                logger.info("Combining plans sequentially...")

            for task, items in grouped_plans.items():
                plans = [item.get("plan", item) if isinstance(item, dict) else item
                         for item in items]

                if len(plans) > 1:
                    result = await self.combine(
                        user_task=task,
                        plans_list=plans,
                        **kwargs
                    )
                    if result:
                        plan_library[result["task"]] = result["plan"]
                else:
                    plan_library[task] = plans[0]

            if self.verbose:
                logger.info("Plan combination complete")

        return plan_library


import asyncio  # Import at end to avoid circular issues
