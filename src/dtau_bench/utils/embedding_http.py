"""Shared helpers for OpenAI-compatible HTTP embedding services."""

from __future__ import annotations

import json
import threading
from typing import List, Optional
from urllib import error, request


class EmbeddingModelResolutionError(RuntimeError):
    """Raised when a configured embedding model cannot be resolved safely."""


def build_openai_api_url(base_url: str, resource: str) -> str:
    """Build a versioned OpenAI-compatible API URL from a configured base."""
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("Embedding HTTP service requires a base_url")
    base = base_url.strip().rstrip("/")
    versioned_base = base if base.endswith("/v1") else f"{base}/v1"
    return f"{versioned_base}/{resource.lstrip('/')}"


def _model_basename(model_id: str) -> str:
    """Return the final path component used for a conservative alias match."""
    return model_id.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]


class HTTPEmbeddingModelResolver:
    """Resolve a configured model name to the ID advertised by the service.

    Resolution is deliberately conservative: an exact ID match wins; otherwise
    the configured name's basename must match exactly one advertised ID's
    basename. The successful result is cached on this resolver instance.
    """

    def __init__(
        self,
        base_url: str,
        configured_model: str,
        api_key: Optional[str] = None,
        timeout: float = 30,
    ) -> None:
        if not isinstance(configured_model, str) or not configured_model.strip():
            raise ValueError("Embedding HTTP service requires a configured model")
        self.configured_model = configured_model.strip()
        self.models_endpoint = build_openai_api_url(base_url, "models")
        self.api_key = api_key or ""
        self.timeout = timeout
        self._resolved_model: Optional[str] = None
        self._lock = threading.Lock()

    def resolve(self) -> str:
        """Return the advertised model ID, querying the service at most once."""
        if self._resolved_model is not None:
            return self._resolved_model

        with self._lock:
            if self._resolved_model is not None:
                return self._resolved_model
            model_ids = self._fetch_model_ids()
            self._resolved_model = self._select_model_id(model_ids)
            return self._resolved_model

    def _fetch_model_ids(self) -> List[str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(self.models_endpoint, headers=headers, method="GET")
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise EmbeddingModelResolutionError(
                f"Embedding models request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise EmbeddingModelResolutionError(
                f"Failed to reach embedding models endpoint {self.models_endpoint}: {exc}"
            ) from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EmbeddingModelResolutionError(
                "Embedding models endpoint returned invalid JSON"
            ) from exc

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise EmbeddingModelResolutionError(
                "Embedding models response missing 'data' list"
            )

        model_ids: List[str] = []
        for index, item in enumerate(data):
            model_id = item.get("id") if isinstance(item, dict) else None
            if not isinstance(model_id, str) or not model_id.strip():
                raise EmbeddingModelResolutionError(
                    f"Embedding models response item {index} missing a valid 'id'"
                )
            model_ids.append(model_id.strip())
        return model_ids

    def _select_model_id(self, model_ids: List[str]) -> str:
        if self.configured_model in model_ids:
            return self.configured_model

        configured_basename = _model_basename(self.configured_model)
        matches = sorted(
            {
                model_id
                for model_id in model_ids
                if _model_basename(model_id) == configured_basename
            }
        )
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise EmbeddingModelResolutionError(
                f"Configured embedding model {self.configured_model!r} is not "
                "advertised by the service (no exact or basename match)"
            )
        raise EmbeddingModelResolutionError(
            f"Configured embedding model {self.configured_model!r} has "
            f"multiple basename matches: {matches}"
        )
