"""Thread-safe append-only store for explorer traces.

Layout:
    experiments/{run}/explore_traces/{domain}.jsonl

One line per explore call. Writes are append-only: a resumed rollout may emit a
duplicate ``line_id`` for the same sample, and the convention is that consumers
keep the LAST record for a given ``line_id`` (see scripts/analyze_explore_traces.py).
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


_LOCKS_GUARD = threading.Lock()
_FILE_LOCKS: Dict[str, threading.Lock] = {}


def _lock_for(path: Path) -> threading.Lock:
    key = str(path)
    with _LOCKS_GUARD:
        lock = _FILE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _FILE_LOCKS[key] = lock
        return lock


class ExploreTraceStore:
    """Append explorer traces for one domain.

    Never raises: tracing must not be able to break a rollout.
    """

    def __init__(
        self,
        *,
        output_dir: Optional[str],
        domain: str,
        dir_name: str = "explore_traces",
        enabled: bool = True,
    ):
        self.domain = domain or "unknown_domain"
        self.enabled = bool(enabled and output_dir)
        self.last_error: Optional[str] = None
        self._path: Optional[Path] = None

        if not self.enabled:
            return
        try:
            trace_dir = Path(str(output_dir)) / dir_name
            trace_dir.mkdir(parents=True, exist_ok=True)
            self._path = trace_dir / f"{self._sanitize(self.domain)}.jsonl"
        except OSError as exc:
            logger.warning("Explorer trace store disabled (%s): %r", output_dir, exc)
            self.enabled = False
            self.last_error = str(exc)

    # ─────────────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return bool(self.enabled and self._path is not None)

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def relative_path(self, dir_name: str = "explore_traces") -> str:
        if self._path is None:
            return ""
        return f"{dir_name}/{self._path.name}"

    def append(self, record: Dict[str, Any]) -> bool:
        """Append one JSON record. Returns True on success."""
        if not self.available:
            return False
        assert self._path is not None
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as exc:
            logger.warning("Explorer trace serialization failed: %r", exc)
            self.last_error = str(exc)
            return False

        try:
            with _lock_for(self._path):
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                    handle.flush()
            return True
        except OSError as exc:
            logger.warning("Explorer trace write failed (%s): %r", self._path, exc)
            self.last_error = str(exc)
            return False

    @staticmethod
    def _sanitize(value: str) -> str:
        safe = "".join(
            ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value)
        )
        return safe.strip("._") or "domain"
