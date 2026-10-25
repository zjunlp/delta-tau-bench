"""Base classes for extraction."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import asyncio
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """Base class for all extractors."""

    def __init__(
        self,
        llm,
        benchmark: str = "atomic",
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ):
        """
        Initialize base extractor.

        Args:
            llm: LLM instance for extraction
            benchmark: Prompt variant name
            config: Optional configuration dictionary
            verbose: Whether to output verbose logs
        """
        self.llm = llm
        self.benchmark = benchmark
        self.config = config or {}
        self.verbose = verbose

    @abstractmethod
    async def extract(self, item: Dict) -> Optional[Dict]:
        """
        Extract from a single item.

        Args:
            item: Input item dictionary

        Returns:
            Extracted result or None if failed
        """
        pass

    async def _process_single_item(
        self,
        item: Dict,
        index: int,
        **kwargs
    ) -> tuple:
        """Process a single item with error handling."""
        try:
            result = await self.extract(item, **kwargs)
            return (index, result)
        except Exception as e:
            logger.error(f"Error processing item {index}: {e}")
            return (index, None)

    async def _process_batch(
        self,
        batch: List[tuple],
        pbar: Optional[tqdm],
        **kwargs
    ) -> List[tuple]:
        """Process a batch of items."""
        tasks = [
            self._process_single_item(item, idx, **kwargs)
            for idx, item in batch
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                original_idx = batch[i][0]
                logger.error(f"Error processing item {original_idx}: {result}")
                processed_results.append((original_idx, None))
            else:
                processed_results.append(result)

            if pbar:
                pbar.update(1)

        return processed_results

    async def extract_batch(
        self,
        items: List[Dict],
        batch_size: int = 10,
        max_concurrent: int = 5,
        show_progress: bool = True,
        **kwargs
    ) -> List[Optional[Dict]]:
        """
        Batch extraction with concurrency control.

        Args:
            items: List of items to process
            batch_size: Size of each batch
            max_concurrent: Maximum concurrent batches
            show_progress: Whether to show progress bar
            **kwargs: Additional arguments passed to extract

        Returns:
            List of extraction results (None for failed items)
        """
        # Index items
        indexed_items = list(enumerate(items))

        # Create batches
        batches = []
        for i in range(0, len(indexed_items), batch_size):
            batch = indexed_items[i:i + batch_size]
            batches.append(batch)

        logger.info(
            f"Processing {len(items)} items in {len(batches)} batches "
            f"(batch_size: {batch_size}, max_concurrent: {max_concurrent})"
        )

        # Progress bar
        pbar = None
        if show_progress:
            pbar = tqdm(
                total=len(items),
                desc=f"Extracting ({self.__class__.__name__})",
                unit="item",
                ncols=100
            )

        # Initialize results
        final_results = [None] * len(items)

        try:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def process_batch_with_semaphore(batch):
                async with semaphore:
                    return await self._process_batch(batch, pbar, **kwargs)

            batch_tasks = [
                process_batch_with_semaphore(batch)
                for batch in batches
            ]

            all_batch_results = await asyncio.gather(*batch_tasks)

            # Collect results
            for batch_results in all_batch_results:
                for original_idx, result in batch_results:
                    final_results[original_idx] = result

        finally:
            if pbar:
                pbar.close()

        # Log statistics
        success_count = sum(1 for r in final_results if r is not None)
        logger.info(
            f"Extraction complete: {success_count}/{len(items)} successful "
            f"({success_count/len(items)*100:.1f}%)"
        )

        return final_results


class BasePlanExtractor(BaseExtractor):
    """Base class for plan extraction."""

    @abstractmethod
    def get_prompt(self) -> str:
        """Get the extraction prompt for this benchmark."""
        pass

    @abstractmethod
    def _extract_plan_from_response(self, text: str) -> Optional[str]:
        """Extract plan from LLM response."""
        pass


class BaseSkillExtractor(BaseExtractor):
    """Base class for skill extraction."""

    @abstractmethod
    def get_skill_type(self) -> str:
        """Return 'functional' or 'atomic'."""
        pass

    @abstractmethod
    def get_prompt(self) -> str:
        """Get the extraction prompt."""
        pass

    @abstractmethod
    def _extract_skills_from_response(self, text: str) -> Optional[List[Dict]]:
        """Extract skills from LLM response."""
        pass
