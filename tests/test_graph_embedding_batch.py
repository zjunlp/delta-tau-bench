"""Resource-control tests for skill-graph embedding."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dtau_bench.skill.bootstrap import graph
from dtau_bench.skill.bootstrap.pipeline import BootstrapConfig, BootstrapPipeline
from dtau_bench.skill.evolve.partition import run_rebuild_and_partition
from dtau_bench.skill.evolve.proposals import load_evolution_artifacts


class GraphEmbeddingBatchTests(unittest.TestCase):
    def test_embed_texts_splits_requests_and_preserves_order(self) -> None:
        calls: list[list[str]] = []

        def fake_embed(_indexer, texts, is_query=False):
            self.assertFalse(is_query)
            calls.append(list(texts))
            return [[float(text.removeprefix("text-"))] for text in texts]

        texts = [f"text-{index}" for index in range(5)]
        with patch(
            "dtau_bench.rag.indexer.ToolIndexer._get_embeddings_batch",
            new=fake_embed,
        ):
            embeddings = graph._embed_texts(
                texts,
                embedding_api_base="https://embedding.invalid/v1",
                embedding_api_key="",
                embedding_model="fixture",
                batch_size=2,
            )

        self.assertEqual([["text-0", "text-1"], ["text-2", "text-3"], ["text-4"]], calls)
        self.assertEqual([[0.0], [1.0], [2.0], [3.0], [4.0]], embeddings)

    def test_bootstrap_forwards_merge_embedding_batch_size_to_graph(self) -> None:
        captured: list[int] = []

        def fake_build_skill_graph(**kwargs):
            captured.append(kwargs["embedding_batch_size"])
            output_dir = kwargs["output_dir"]
            (output_dir / "skill_graph.json").write_text(
                json.dumps({"nodes": [{"node_id": "fixture"}], "edges": []}),
                encoding="utf-8",
            )
            (output_dir / "skill_graph_summary.json").write_text(
                "{}", encoding="utf-8"
            )
            return {"total_nodes": 1, "total_edges": 0}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = BootstrapConfig(
                output_dir=root / "skills",
                rollout_dir=root / "rollouts",
                query_dir=root / "queries",
                tool_schemas_dir=root / "tools",
                tool_schema_version="v1",
                model="fixture",
                base_url="https://llm.invalid/v1",
                api_key="",
                embedding_api_base="https://embedding.invalid/v1",
                embedding_model="fixture",
                merge_embedding_batch_size=7,
            )
            pipeline = BootstrapPipeline(config)
            with patch.object(
                pipeline,
                "_stage_should_run",
                side_effect=[False, False, False, True],
            ), patch(
                "dtau_bench.skill.bootstrap.graph.build_skill_graph",
                side_effect=fake_build_skill_graph,
            ):
                pipeline.run()

        self.assertEqual([7], captured)

    def test_evolve_forwards_merge_embedding_batch_size_to_graph(self) -> None:
        captured: list[int] = []
        config = SimpleNamespace(
            tool_schemas_dir=Path("/nonexistent/tools"),
            tool_schema_version="v2",
            prev_library=None,
            embedding_api_base="https://embedding.invalid/v1",
            embedding_api_key="",
            embedding_model="fixture",
            plan_atomic_coverage_threshold=0.5,
            merge_embedding_batch_size=9,
        )

        with tempfile.TemporaryDirectory() as tmp, patch(
            "dtau_bench.skill.bootstrap.graph.build_skill_graph",
            side_effect=lambda **kwargs: captured.append(
                kwargs["embedding_batch_size"]
            ),
        ):
            asyncio.run(
                run_rebuild_and_partition(
                    output_dir=Path(tmp),
                    artifacts=load_evolution_artifacts(None),
                    config=config,
                )
            )

        self.assertEqual([9], captured)


if __name__ == "__main__":
    unittest.main()
