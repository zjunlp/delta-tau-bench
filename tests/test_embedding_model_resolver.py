import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dtau_bench.rag.config import RAGConfig
from dtau_bench.rag.indexer import ToolDocument, ToolIndexer
from dtau_bench.rag.retriever import ToolRetriever
from dtau_bench.skill.core.clustering.embedding import EmbeddingService
from dtau_bench.skill.indexer import SkillIndexer
from dtau_bench.skill.models import SkillDocument
from dtau_bench.utils.embedding_http import (
    EmbeddingModelResolutionError,
    HTTPEmbeddingModelResolver,
)


class _JSONResponse:
    def __init__(self, payload, status=200):
        self._body = json.dumps(payload).encode("utf-8")
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _EmbeddingGateway:
    def __init__(self, advertised_model, embedding=None):
        self.advertised_model = advertised_model
        self.embedding = embedding or [1.0, 0.0]
        self.model_requests = []
        self.embedding_payloads = []

    def __call__(self, req, timeout=None):
        if req.full_url.endswith("/models"):
            self.model_requests.append(req)
            return _JSONResponse({"data": [{"id": self.advertised_model}]})
        if req.full_url.endswith("/embeddings"):
            payload = json.loads(req.data.decode("utf-8"))
            self.embedding_payloads.append(payload)
            return _JSONResponse(
                {
                    "data": [
                        {"embedding": self.embedding}
                        for _ in payload.get("input", [])
                    ]
                }
            )
        raise AssertionError(f"Unexpected URL: {req.full_url}")


def _http_config():
    return RAGConfig(
        embedding_model="Qwen3-Embedding-0.6B",
        embedding_backend="openai_http",
        embedding_api_base="http://embedding.example/v1",
        embedding_api_key="secret",
        query_instruction="retrieve tools",
    )


def _tool_document():
    return ToolDocument(
        tool_name="demo",
        function_name="demo",
        domain="demo",
        description="",
        parameters_text="",
        full_text="demo tool",
    )


class HTTPEmbeddingModelResolverTests(unittest.TestCase):
    def _resolve_from(self, configured_model, advertised_ids):
        resolver = HTTPEmbeddingModelResolver(
            base_url="http://embedding.example/v1/",
            configured_model=configured_model,
            api_key="secret",
        )
        response = _JSONResponse(
            {"data": [{"id": model_id} for model_id in advertised_ids]}
        )
        with patch(
            "dtau_bench.utils.embedding_http.request.urlopen",
            return_value=response,
        ) as urlopen:
            result = resolver.resolve()
            cached_result = resolver.resolve()
        return result, cached_result, urlopen

    def test_exact_match_wins_and_is_cached_per_instance(self):
        result, cached_result, urlopen = self._resolve_from(
            "Qwen3-Embedding-0.6B",
            [
                "/models/Qwen3-Embedding-0.6B",
                "Qwen3-Embedding-0.6B",
            ],
        )

        self.assertEqual("Qwen3-Embedding-0.6B", result)
        self.assertEqual(result, cached_result)
        self.assertEqual(1, urlopen.call_count)
        req = urlopen.call_args.args[0]
        self.assertEqual("http://embedding.example/v1/models", req.full_url)
        self.assertEqual("GET", req.get_method())
        self.assertEqual("Bearer secret", req.get_header("Authorization"))

    def test_unique_basename_match_returns_advertised_id(self):
        advertised = "/srv/models/Qwen3-Embedding-0.6B"
        result, _, _ = self._resolve_from(
            "Qwen3-Embedding-0.6B",
            [advertised, "/srv/models/another-model"],
        )

        self.assertEqual(advertised, result)

    def test_no_match_fails_closed(self):
        with self.assertRaisesRegex(
            EmbeddingModelResolutionError,
            "no exact or basename match",
        ):
            self._resolve_from("wanted", ["other"])

    def test_multiple_basename_matches_fail_closed(self):
        with self.assertRaisesRegex(
            EmbeddingModelResolutionError,
            "multiple basename matches",
        ):
            self._resolve_from(
                "Qwen3-Embedding-0.6B",
                [
                    "/models/a/Qwen3-Embedding-0.6B",
                    "/models/b/Qwen3-Embedding-0.6B",
                ],
            )


class EmbeddingClientModelResolutionTests(unittest.TestCase):
    advertised_model = "/srv/models/Qwen3-Embedding-0.6B"

    def test_tool_indexer_uses_resolved_id_but_metadata_keeps_configured_name(self):
        gateway = _EmbeddingGateway(self.advertised_model)
        indexer = ToolIndexer(
            config=_http_config(),
            tools_dir=Path("unused"),
            embedding_batch_size=1,
        )
        indexer._load_documents_from_tools_dir = lambda: [_tool_document()]

        with patch(
            "dtau_bench.utils.embedding_http.request.urlopen",
            side_effect=gateway,
        ):
            index = indexer.build_index()
            indexer._embed_via_http(["second request"])

        self.assertEqual(1, len(gateway.model_requests))
        self.assertEqual(
            [self.advertised_model, self.advertised_model],
            [payload["model"] for payload in gateway.embedding_payloads],
        )
        self.assertEqual(
            "Qwen3-Embedding-0.6B",
            index["metadata"]["embedding_model"],
        )
        self.assertNotIn(self.advertised_model, repr(index["metadata"]))

    def test_tool_retriever_uses_resolved_id_and_caches_lookup(self):
        gateway = _EmbeddingGateway(self.advertised_model)
        retriever = ToolRetriever(
            index={
                "documents": [_tool_document()],
                "embeddings": np.array([[1.0, 0.0]], dtype=np.float32),
            },
            config=_http_config(),
        )

        with patch(
            "dtau_bench.utils.embedding_http.request.urlopen",
            side_effect=gateway,
        ):
            first = retriever._embed_query_via_http("first")
            second = retriever._embed_query_via_http("second")

        self.assertEqual(1, len(gateway.model_requests))
        self.assertEqual((2,), first.shape)
        self.assertEqual((2,), second.shape)
        self.assertTrue(
            all(
                payload["model"] == self.advertised_model
                for payload in gateway.embedding_payloads
            )
        )
        self.assertEqual("Qwen3-Embedding-0.6B", retriever.config.embedding_model)

    def test_skill_indexer_delegates_resolution_and_keeps_short_name_metadata(self):
        gateway = _EmbeddingGateway(self.advertised_model)
        indexer = SkillIndexer(
            config=_http_config(),
            embedding_batch_size=1,
        )
        document = SkillDocument(
            skill_type="atomic",
            name="demo",
            document="",
            content="",
            tools=["demo"],
            embedding_text="demo skill",
        )

        with patch(
            "dtau_bench.utils.embedding_http.request.urlopen",
            side_effect=gateway,
        ):
            index = indexer._build_index([document], "skills", "/skills")

        self.assertEqual(1, len(gateway.model_requests))
        self.assertEqual(self.advertised_model, gateway.embedding_payloads[0]["model"])
        self.assertEqual(
            "Qwen3-Embedding-0.6B",
            index["metadata"]["embedding_model"],
        )
        self.assertNotIn(self.advertised_model, repr(index["metadata"]))

    def test_clustering_embedding_service_uses_resolved_id_and_caches_lookup(self):
        gateway = _EmbeddingGateway(self.advertised_model)
        service = EmbeddingService(
            model="Qwen3-Embedding-0.6B",
            base_url="http://embedding.example/v1",
            batch_size=1,
            api_key="secret",
        )

        with patch(
            "dtau_bench.utils.embedding_http.request.urlopen",
            side_effect=gateway,
        ):
            first = service._embed_sync(["first"])
            second = service._embed_sync(["second"])

        self.assertEqual(1, len(gateway.model_requests))
        self.assertEqual((1, 2), first.shape)
        self.assertEqual((1, 2), second.shape)
        self.assertTrue(
            all(
                payload["model"] == self.advertised_model
                for payload in gateway.embedding_payloads
            )
        )
        self.assertEqual("Qwen3-Embedding-0.6B", service.model)


if __name__ == "__main__":
    unittest.main()
