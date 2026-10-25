"""Global configuration settings for skill context."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import os


@dataclass
class EmbeddingConfig:
    """Configuration for embedding service."""
    model: str = "Qwen3-Embedding-8B"
    index_type: str = "FAISS-HNSW"
    similarity_threshold: float = 0.45
    batch_size: int = 32


@dataclass
class ClusteringConfig:
    """Configuration for DBSCAN clustering."""
    eps: float = 0.10  # 1 - cosine_similarity_threshold (0.90)
    min_samples: int = 1
    metric: str = "cosine"


@dataclass
class ExtractionConfig:
    """Configuration for skill extraction."""
    batch_size: int = 10
    max_concurrent: int = 5
    max_retries: int = 5
    filter_threshold: float = 0.999


@dataclass
class Settings:
    """Global settings for skill context."""

    # Benchmark settings
    benchmark: str = "atomic"
    supported_benchmarks: tuple = ("atomic", "functional")

    # LLM settings
    default_model: str = "gpt-4.1-2025-04-14"

    # Embedding settings
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    # Clustering settings
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)

    # Extraction settings
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)

    # Paths
    output_dir: str = "./output"
    cache_dir: str = "./.cache"

    # Logging
    log_level: str = "INFO"
    verbose: bool = True

    def __post_init__(self):
        """Validate settings after initialization."""
        if self.benchmark not in self.supported_benchmarks:
            raise ValueError(
                f"Unsupported benchmark: {self.benchmark}. "
                f"Must be one of: {self.supported_benchmarks}"
            )

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""
        return cls(
            benchmark=os.getenv("SKILL_BENCHMARK", "atomic"),
            default_model=os.getenv("SKILL_MODEL", "gpt-4.1-2025-04-14"),
            output_dir=os.getenv("SKILL_OUTPUT_DIR", "./output"),
            cache_dir=os.getenv("SKILL_CACHE_DIR", "./.cache"),
            log_level=os.getenv("SKILL_LOG_LEVEL", "INFO"),
            verbose=os.getenv("SKILL_VERBOSE", "true").lower() == "true",
        )


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def set_settings(settings: Settings) -> None:
    """Set the global settings instance."""
    global _settings
    _settings = settings
