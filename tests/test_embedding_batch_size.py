import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from dtau_bench.cli import chain
from dtau_bench.cli.build_skill_index import parse_args as parse_skill_args
from dtau_bench.cli.build_tool_index import parse_args as parse_tool_args
from dtau_bench.rag.config import RAGConfig
from dtau_bench.rag.indexer import ToolDocument, ToolIndexer
from dtau_bench.skill.indexer import SkillIndexer
from dtau_bench.skill.models import SkillDocument
from dtau_bench.utils.config import resolve_embedding_config


class EmbeddingConfigBatchSizeTests(unittest.TestCase):
    @patch("dtau_bench.utils.config.load_models_config")
    def test_resolve_embedding_config_includes_batch_size(self, load_models):
        load_models.return_value = {
            "embedding": {
                "default": {
                    "model": "embedding-model",
                    "backend": "local",
                    "batch_size": 8,
                }
            }
        }

        config = resolve_embedding_config()

        self.assertEqual(8, config["batch_size"])

    @patch("dtau_bench.utils.config.load_models_config")
    def test_resolve_embedding_config_defaults_to_historical_size(self, load_models):
        load_models.return_value = {
            "embedding": {
                "default": {"model": "embedding-model", "backend": "local"}
            }
        }

        self.assertEqual(64, resolve_embedding_config()["batch_size"])

    @patch("dtau_bench.utils.config.load_models_config")
    def test_resolve_embedding_config_rejects_invalid_size(self, load_models):
        load_models.return_value = {
            "embedding": {
                "default": {
                    "model": "embedding-model",
                    "backend": "local",
                    "batch_size": 0,
                }
            }
        }

        with self.assertRaisesRegex(ValueError, "batch_size"):
            resolve_embedding_config()


class EmbeddingBatchCliTests(unittest.TestCase):
    def test_tool_cli_accepts_batch_size_override(self):
        with patch.object(
            sys,
            "argv",
            ["build-tool-index", "--version", "v1", "--batch-size", "8"],
        ):
            self.assertEqual(8, parse_tool_args().batch_size)

    def test_skill_cli_accepts_batch_size_override(self):
        with patch.object(
            sys,
            "argv",
            [
                "build-skill-index",
                "--skill-library-dir",
                "skills",
                "--batch-size",
                "8",
            ],
        ):
            self.assertEqual(8, parse_skill_args().batch_size)

    def test_cli_rejects_non_positive_batch_size(self):
        with patch.object(
            sys,
            "argv",
            ["build-tool-index", "--version", "v1", "--batch-size", "0"],
        ), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                parse_tool_args()
        self.assertEqual(2, raised.exception.code)


class IndexEmbeddingBatchSizeTests(unittest.TestCase):
    @staticmethod
    def _config(api_key="secret-key-that-must-not-be-persisted"):
        return RAGConfig(
            embedding_model="Qwen3-Embedding-0.6B",
            embedding_backend="local",
            embedding_api_key=api_key,
            query_instruction="query",
        )

    def test_tool_index_batches_and_records_non_secret_config(self):
        indexer = ToolIndexer(
            config=self._config(),
            tools_dir=Path("unused"),
            embedding_batch_size=2,
        )
        documents = [
            ToolDocument(
                tool_name=f"tool-{i}",
                function_name=f"tool-{i}",
                domain="demo",
                description="",
                parameters_text="",
                full_text=f"tool {i}",
            )
            for i in range(5)
        ]
        batch_lengths = []
        indexer._load_documents_from_tools_dir = lambda: documents

        def embed(texts, is_query=False):
            batch_lengths.append(len(texts))
            return [[float(i), 0.0] for i, _ in enumerate(texts)]

        indexer._get_embeddings_batch = embed

        index = indexer.build_index()

        self.assertEqual([2, 2, 1], batch_lengths)
        self.assertEqual(2, index["metadata"]["embedding_batch_size"])
        self.assertNotIn("api_key", index["metadata"])
        self.assertNotIn("secret-key", repr(index["metadata"]))

    def test_skill_index_batches_and_records_size(self):
        indexer = SkillIndexer(
            config=self._config(),
            embedding_batch_size=2,
        )
        documents = [
            SkillDocument(
                skill_type="atomic",
                name=f"skill-{i}",
                document="",
                content="",
                tools=[],
                embedding_text=f"skill {i}",
            )
            for i in range(5)
        ]
        batch_lengths = []

        def embed(texts, is_query=False):
            batch_lengths.append(len(texts))
            return [[float(i), 0.0] for i, _ in enumerate(texts)]

        indexer._tool_indexer._get_embeddings_batch = embed

        index = indexer._build_index(documents, "skills", "/skills")

        self.assertEqual([2, 2, 1], batch_lengths)
        self.assertEqual(2, index["metadata"]["embedding_batch_size"])

    def test_validation_detects_explicit_batch_size_change(self):
        indexer = ToolIndexer(
            config=self._config(),
            embedding_batch_size=8,
        )
        index = {
            "metadata": {
                "embedding_model": "Qwen3-Embedding-0.6B",
                "embedding_backend": "local",
                "query_instruction": "query",
                "query_format_version": "instruct_query_v1",
                "embedding_batch_size": 64,
            }
        }

        self.assertFalse(indexer.validate_index(index))

    def test_validation_treats_missing_batch_metadata_as_historical_default(self):
        indexer = ToolIndexer(
            config=self._config(),
            embedding_batch_size=64,
        )
        index = {
            "metadata": {
                "embedding_model": "Qwen3-Embedding-0.6B",
                "embedding_backend": "local",
                "query_instruction": "query",
                "query_format_version": "instruct_query_v1",
            }
        }

        self.assertTrue(indexer.validate_index(index))


class ChainEmbeddingBatchMetadataTests(unittest.TestCase):
    def test_chain_validator_requires_configured_batch_size_without_api_key(self):
        library_dir = Path("/tmp/test-skill-library")
        embedding = {
            "model": "Qwen3-Embedding-0.6B",
            "backend": "local",
            "base_url": "",
            "api_key": "secret-key-that-must-not-be-persisted",
            "batch_size": 8,
        }
        with patch(
            "dtau_bench.skill.loader.SkillLibraryLoader.load",
            return_value=SimpleNamespace(tool_skills=[]),
        ), patch(
            "dtau_bench.utils.config.resolve_embedding_config",
            return_value=embedding,
        ), patch.object(chain, "_validate_embedding_index_dir") as validate_dir:
            chain._validate_skill_index_artifacts(library_dir)

        self.assertEqual(2, validate_dir.call_count)
        for call in validate_dir.call_args_list:
            metadata = call.kwargs["expected_metadata"]
            self.assertEqual(8, metadata["embedding_batch_size"])
            self.assertNotIn("api_key", metadata)
            self.assertNotIn("secret-key", repr(metadata))

    def test_chain_validator_uses_historical_default_for_old_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp)
            np.save(index_dir / "embeddings.npy", np.array([[1.0, 0.0]]))
            (index_dir / "documents.json").write_text("[{}]", encoding="utf-8")
            (index_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "kind": "skills",
                        "total_documents": 1,
                        "embedding_dim": 2,
                    }
                ),
                encoding="utf-8",
            )

            chain._validate_embedding_index_dir(
                index_dir,
                kind="skills",
                expected_metadata={"embedding_batch_size": 64},
            )
            with self.assertRaisesRegex(
                chain.ArtifactValidationError,
                "embedding_batch_size=64; expected 8",
            ):
                chain._validate_embedding_index_dir(
                    index_dir,
                    kind="skills",
                    expected_metadata={"embedding_batch_size": 8},
                )


if __name__ == "__main__":
    unittest.main()
