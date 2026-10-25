"""Async utilities for batch processing."""

import asyncio
from typing import List, Callable, Any, Optional, TypeVar
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AsyncBatchProcessor:
    """
    Generic async batch processor with concurrency control.

    Processes items in batches with progress tracking and error handling.
    """

    def __init__(
        self,
        batch_size: int = 10,
        max_concurrent: int = 5,
        show_progress: bool = True,
        desc: str = "Processing"
    ):
        """
        Initialize batch processor.

        Args:
            batch_size: Number of items per batch
            max_concurrent: Maximum concurrent batches
            show_progress: Whether to show progress bar
            desc: Progress bar description
        """
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.show_progress = show_progress
        self.desc = desc

    async def process(
        self,
        items: List[T],
        process_func: Callable[[T], Any],
        **kwargs
    ) -> List[Any]:
        """
        Process items in batches.

        Args:
            items: List of items to process
            process_func: Async function to process each item
            **kwargs: Additional arguments for process_func

        Returns:
            List of results
        """
        # Create batches
        batches = []
        for i in range(0, len(items), self.batch_size):
            batches.append(items[i:i + self.batch_size])

        logger.info(
            f"Processing {len(items)} items in {len(batches)} batches "
            f"(batch_size: {self.batch_size}, max_concurrent: {self.max_concurrent})"
        )

        # Progress bar
        pbar = None
        if self.show_progress:
            pbar = tqdm(
                total=len(items),
                desc=self.desc,
                unit="item",
                ncols=100
            )

        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_batch(batch):
            async with semaphore:
                batch_results = []
                tasks = [process_func(item, **kwargs) for item in batch]
                task_results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in task_results:
                    if isinstance(result, Exception):
                        logger.error(f"Error processing item: {result}")
                        batch_results.append(None)
                    else:
                        batch_results.append(result)

                    if pbar:
                        pbar.update(1)

                return batch_results

        try:
            batch_tasks = [process_batch(batch) for batch in batches]
            all_results = await asyncio.gather(*batch_tasks)

            for batch_results in all_results:
                results.extend(batch_results)

        finally:
            if pbar:
                pbar.close()

        success_count = sum(1 for r in results if r is not None)
        logger.info(
            f"Processing complete: {success_count}/{len(items)} successful "
            f"({success_count/len(items)*100:.1f}%)"
        )

        return results
