"""Base classes for filtering."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import asyncio
import logging
from tqdm import tqdm

from ..prompts.registry import PromptRegistry

logger = logging.getLogger(__name__)


class BaseFilter(ABC):
    """Base class for skill filtering."""

    def __init__(
        self,
        llm,
        benchmark: str = "atomic",
        verbose: bool = True
    ):
        self.llm = llm
        self.benchmark = benchmark
        self.verbose = verbose

    @abstractmethod
    async def filter(self, skill: Dict) -> bool:
        """
        Filter a single skill.

        Args:
            skill: Skill dictionary to filter

        Returns:
            True if skill passes filter, False otherwise
        """
        pass

    async def filter_batch(
        self,
        skills: List[Dict],
        batch_size: int = 10,
        max_concurrent: int = 5,
        show_progress: bool = True,
        **kwargs
    ) -> List[Dict]:
        """
        Filter a batch of skills.

        Args:
            skills: List of skill dictionaries
            batch_size: Batch size for processing
            max_concurrent: Maximum concurrent batches
            show_progress: Whether to show progress bar

        Returns:
            List of skills with filter_result field added
        """
        logger.info(f"Filtering {len(skills)} skills")

        # Create batches
        batches = []
        for i in range(0, len(skills), batch_size):
            batches.append(skills[i:i + batch_size])

        logger.info(
            f"Processing in {len(batches)} batches "
            f"(batch_size: {batch_size}, max_concurrent: {max_concurrent})"
        )

        # Progress bar
        pbar = None
        if show_progress:
            pbar = tqdm(
                total=len(skills),
                desc=f"Filtering ({self.__class__.__name__})",
                unit="skill",
                ncols=100
            )

        final_results = []

        try:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def process_batch(batch):
                async with semaphore:
                    results = []
                    tasks = [self._filter_single(skill, **kwargs) for skill in batch]
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                    for i, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            logger.error(f"Error filtering skill: {result}")
                            skill_copy = batch[i].copy()
                            skill_copy["filter_result"] = False
                            results.append(skill_copy)
                        else:
                            results.append(result)

                        if pbar:
                            pbar.update(1)

                    return results

            batch_tasks = [process_batch(batch) for batch in batches]
            all_results = await asyncio.gather(*batch_tasks)

            for batch_results in all_results:
                final_results.extend(batch_results)

        finally:
            if pbar:
                pbar.close()

        # Statistics
        passed = sum(1 for r in final_results if r.get("filter_result", False))
        logger.info(
            f"Filter complete: {passed}/{len(skills)} passed "
            f"({passed/len(skills)*100:.1f}%)"
        )

        return final_results

    async def _filter_single(self, skill: Dict, **kwargs) -> Dict:
        """Filter a single skill with error handling."""
        skill_copy = skill.copy()
        try:
            result = await self.filter(skill, **kwargs)
            skill_copy["filter_result"] = result
        except Exception as e:
            logger.error(f"Error in filter: {e}")
            skill_copy["filter_result"] = False
        return skill_copy


class GeneralFilter(BaseFilter):
    """
    General quality filter (Stage 1).

    Checks for:
    - Domain specificity (uses proper APIs)
    - No over-encapsulation
    - No Python library imports
    - Parameter reusability
    - No functional style (no return statements)
    """

    def get_prompt(self) -> str:
        """Get filter prompt based on skill type."""
        return PromptRegistry.get("general_filter", self.benchmark)

    async def filter(self, skill: Dict, **kwargs) -> bool:
        """Check if skill passes general quality filter."""
        skill_data = skill.get("skill", skill)
        content = skill_data.get("content", "")

        messages = [
            ("system", self.get_prompt()),
            ("human", f"# Here is the function: {content}")
        ]

        response = await self.llm.ainvoke(messages=messages, **kwargs)

        if self.verbose:
            logger.debug(f"Filter response: {response}")

        return "good" in response.lower()


class ToolSchemaFilter(BaseFilter):
    """
    Tool-schema validation filter (Stage 2).

    Validates that skill's tool usage matches the tool specifications.
    """

    def __init__(
        self,
        llm,
        benchmark: str = "atomic",
        tool_schemas: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ):
        super().__init__(llm, benchmark, verbose)
        self.tool_schemas = tool_schemas or {}

    def load_schemas(self, schemas: Dict[str, Any]) -> None:
        """Load tool schemas."""
        self.tool_schemas = schemas

    def get_tool_schema(self, tool_name: str) -> Optional[Dict]:
        """Get schema for a specific tool."""
        return self.tool_schemas.get(tool_name)

    def get_prompt(self) -> str:
        """Get tool filter prompt."""
        return PromptRegistry.get("tool_filter", self.benchmark)

    async def filter(self, skill: Dict, **kwargs) -> bool:
        """Check if skill's tool usage is correct."""
        skill_data = skill.get("skill", skill)
        content = skill_data.get("content", "")
        tools = skill_data.get("tools", [])

        # Get tool documentation
        tool_docs = []
        for tool_name in tools:
            schema = self.get_tool_schema(tool_name)
            if schema:
                tool_docs.append(schema)

        if not tool_docs:
            # No schemas available, skip validation
            return True

        import json
        tool_doc_str = json.dumps(tool_docs, ensure_ascii=False, indent=2)

        messages = [
            ("system", self.get_prompt()),
            ("human", (
                f"# **Tool invocation content**:\n{content}\n\n"
                f"# **Tool specifications**:{tool_doc_str}\n"
            ))
        ]

        response = await self.llm.ainvoke(messages=messages, **kwargs)

        if self.verbose:
            logger.debug(f"Tool filter response: {response}")

        # Check for 'correct' in the answer tag
        answer_part = response.split("<answer>")[-1].lower() if "<answer>" in response else response.lower()
        return "correct" in answer_part
