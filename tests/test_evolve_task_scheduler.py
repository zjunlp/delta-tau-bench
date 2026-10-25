import asyncio
import unittest

from dtau_bench.skill.evolve.proposals import _run_batched_tasks


class EvolveTaskSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_concurrency_is_not_capped_by_batch_size(self):
        active = 0
        peak = 0
        four_started = asyncio.Event()
        release_one = asyncio.Event()
        release_rest = asyncio.Event()
        fifth_started = asyncio.Event()

        async def job(index):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 4:
                four_started.set()
            if index == 4:
                fifth_started.set()
            try:
                if index == 0:
                    await release_one.wait()
                else:
                    await release_rest.wait()
                return index
            finally:
                active -= 1

        scheduler = asyncio.create_task(_run_batched_tasks(
            [job(index) for index in range(8)],
            batch_size=2,
            max_concurrent=4,
            show_progress=False,
            progress_desc="test",
            progress_unit="task",
        ))

        await asyncio.wait_for(four_started.wait(), timeout=1)
        self.assertEqual(peak, 4)
        release_one.set()
        await asyncio.wait_for(fifth_started.wait(), timeout=1)
        self.assertEqual(active, 4)
        release_rest.set()

        self.assertEqual(await asyncio.wait_for(scheduler, timeout=1), list(range(8)))

    async def test_results_keep_input_order_and_exceptions_are_returned(self):
        async def job(index):
            await asyncio.sleep(0)
            if index == 1:
                raise RuntimeError("expected")
            return index

        results = await _run_batched_tasks(
            [job(index) for index in range(3)],
            batch_size=1,
            max_concurrent=3,
            show_progress=False,
            progress_desc="test",
            progress_unit="task",
        )

        self.assertEqual(results[0], 0)
        self.assertIsInstance(results[1], RuntimeError)
        self.assertEqual(results[2], 2)


if __name__ == "__main__":
    unittest.main()
