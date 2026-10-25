import unittest

from dtau_bench.skill.core.clustering.embedding import EmbeddingService


class EmbeddingServiceEndpointTests(unittest.TestCase):
    def test_base_url_with_v1_does_not_duplicate_version_path(self):
        service = EmbeddingService(
            base_url="http://127.0.0.1:8009/v1",
        )

        self.assertEqual(
            "http://127.0.0.1:8009/v1/embeddings",
            service.endpoint,
        )

    def test_base_url_with_v1_and_trailing_slash_is_normalized(self):
        service = EmbeddingService(
            base_url="http://127.0.0.1:8009/v1/",
        )

        self.assertEqual(
            "http://127.0.0.1:8009/v1/embeddings",
            service.endpoint,
        )

    def test_unversioned_base_url_gets_v1_path(self):
        service = EmbeddingService(
            base_url="http://127.0.0.1:8009",
        )

        self.assertEqual(
            "http://127.0.0.1:8009/v1/embeddings",
            service.endpoint,
        )


if __name__ == "__main__":
    unittest.main()
