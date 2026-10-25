import io
import json
import unittest
from unittest.mock import patch
from urllib import error

from dtau_bench.rag.config import RAGConfig
from dtau_bench.rag.indexer import ToolIndexer
from dtau_bench.utils.config import resolve_embedding_config


class _JSONResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def _indexer(*, timeout=600.0, max_retries=2, retry_delay=0.0):
    config = RAGConfig(
        embedding_model="embedding-model",
        embedding_backend="openai_http",
        embedding_api_base="http://embedding.example/v1",
        embedding_http_timeout=timeout,
        embedding_http_max_retries=max_retries,
        embedding_http_retry_delay=retry_delay,
    )
    indexer = ToolIndexer(config=config, embedding_batch_size=1)
    indexer._http_model_resolver._resolved_model = "embedding-model"
    return indexer


class EmbeddingHTTPRetryTests(unittest.TestCase):
    def test_timeout_is_retried_with_configured_timeout(self):
        response = _JSONResponse({"data": [{"embedding": [1.0, 0.0]}]})
        indexer = _indexer(timeout=321.0, max_retries=2)

        with patch(
            "dtau_bench.rag.indexer.request.urlopen",
            side_effect=[TimeoutError("slow batch"), response],
        ) as urlopen, patch("dtau_bench.rag.indexer.time.sleep") as sleep:
            embeddings = indexer._embed_via_http(["text"])

        self.assertEqual([[1.0, 0.0]], embeddings)
        self.assertEqual(2, urlopen.call_count)
        self.assertEqual([321.0, 321.0], [call.kwargs["timeout"] for call in urlopen.call_args_list])
        sleep.assert_not_called()

    def test_timeout_exhaustion_has_bounded_attempts(self):
        indexer = _indexer(timeout=5.0, max_retries=1)

        with patch(
            "dtau_bench.rag.indexer.request.urlopen",
            side_effect=TimeoutError("still slow"),
        ) as urlopen:
            with self.assertRaisesRegex(RuntimeError, "timed out.*5s"):
                indexer._embed_via_http(["text"])

        self.assertEqual(2, urlopen.call_count)

    def test_runtime_query_keeps_single_bounded_attempt(self):
        indexer = _indexer(timeout=600.0, max_retries=2)

        with patch(
            "dtau_bench.rag.indexer.request.urlopen",
            side_effect=TimeoutError("slow query"),
        ) as urlopen:
            with self.assertRaisesRegex(RuntimeError, "timed out.*120s"):
                indexer._get_embeddings_batch(["query"], is_query=True)

        self.assertEqual(1, urlopen.call_count)
        self.assertEqual(120.0, urlopen.call_args.kwargs["timeout"])

    def test_non_retryable_http_error_fails_immediately(self):
        indexer = _indexer(max_retries=2)
        bad_request = error.HTTPError(
            "http://embedding.example/v1/embeddings",
            400,
            "Bad Request",
            hdrs=None,
            fp=io.BytesIO(b"invalid input"),
        )

        with patch(
            "dtau_bench.rag.indexer.request.urlopen",
            side_effect=bad_request,
        ) as urlopen:
            with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
                indexer._embed_via_http(["text"])

        self.assertEqual(1, urlopen.call_count)


class EmbeddingHTTPConfigTests(unittest.TestCase):
    @patch("dtau_bench.utils.config.load_models_config")
    def test_resolve_embedding_http_controls(self, load_models_config):
        load_models_config.return_value = {
            "embedding": {
                "default": {
                    "model": "embedding-model",
                    "backend": "openai_compatible",
                    "base_url": "http://embedding.example/v1",
                    "request_timeout": 450,
                    "max_retries": 4,
                    "retry_delay": 2.5,
                }
            }
        }

        resolved = resolve_embedding_config()

        self.assertEqual(450, resolved["request_timeout"])
        self.assertEqual(4, resolved["max_retries"])
        self.assertEqual(2.5, resolved["retry_delay"])

    def test_rag_config_round_trip_preserves_http_controls(self):
        config = _indexer(
            timeout=450.0,
            max_retries=4,
            retry_delay=2.5,
        ).config

        restored = RAGConfig.from_dict(config.to_dict())

        self.assertEqual(450.0, restored.embedding_http_timeout)
        self.assertEqual(4, restored.embedding_http_max_retries)
        self.assertEqual(2.5, restored.embedding_http_retry_delay)


if __name__ == "__main__":
    unittest.main()
