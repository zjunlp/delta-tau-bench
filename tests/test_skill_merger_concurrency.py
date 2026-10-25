import asyncio
import unittest

from dtau_bench.skill.core.clustering.merger import SkillMerger


class SkillMergerConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_merge_clusters_limits_concurrency_and_preserves_order(self):
        merger = SkillMerger(
            llm=object(),
            verbose=False,
            max_concurrent=2,
        )
        active = 0
        peak_active = 0
        started = []
        first_pair_started = asyncio.Event()
        release_first_pair = asyncio.Event()

        async def controlled_merge(cluster, **kwargs):
            nonlocal active, peak_active
            cluster_id = cluster[0]["id"]
            active += 1
            peak_active = max(peak_active, active)
            started.append(cluster_id)
            if len(started) == 2:
                first_pair_started.set()
            if cluster_id < 2:
                await release_first_pair.wait()
            active -= 1
            return {"id": cluster_id}

        merger.merge = controlled_merge
        clusters = [[{"id": cluster_id}] for cluster_id in range(5)]
        merge_task = asyncio.create_task(merger.merge_clusters(clusters))

        await asyncio.wait_for(first_pair_started.wait(), timeout=1)
        await asyncio.sleep(0)
        self.assertEqual([0, 1], started)
        self.assertEqual(2, active)

        release_first_pair.set()
        results = await asyncio.wait_for(merge_task, timeout=1)

        self.assertLessEqual(peak_active, 2)
        self.assertEqual([0, 1, 2, 3, 4], [result["id"] for result in results])

    async def test_merge_clusters_keeps_log_and_skip_exception_behavior(self):
        merger = SkillMerger(
            llm=object(),
            verbose=False,
            max_concurrent=1,
        )

        async def sometimes_fails(cluster, **kwargs):
            cluster_id = cluster[0]["id"]
            if cluster_id == 1:
                raise RuntimeError("expected test failure")
            return {"id": cluster_id}

        merger.merge = sometimes_fails
        clusters = [[{"id": cluster_id}] for cluster_id in range(3)]

        with self.assertLogs(
            "dtau_bench.skill.core.clustering.merger", level="ERROR"
        ):
            results = await merger.merge_clusters(clusters)

        self.assertEqual([0, 2], [result["id"] for result in results])

    def test_rejects_non_positive_concurrency(self):
        with self.assertRaisesRegex(ValueError, "max_concurrent"):
            SkillMerger(llm=object(), max_concurrent=0)


if __name__ == "__main__":
    unittest.main()
