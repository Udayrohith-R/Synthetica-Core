"""In-memory TTL cache for document → AtomicClaim extraction results."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from threading import Lock

from app.models.schemas import AtomicClaim

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    claims: list[AtomicClaim]
    expires_at: float


class ClaimExtractionCache:
    """
    Process-local cache keyed by content hash.

    Cuts repeat LLM cost when the same Exa URL/body is seen again within TTL.
    """

    def __init__(self, *, ttl_seconds: float = 3600.0, max_entries: int = 256) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._store: dict[str, _CacheEntry] = {}
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(
        *,
        url: str,
        text: str,
        model: str,
        max_chars: int = 4000,
    ) -> str:
        payload = f"{url}\n{model}\n{text[:max_chars]}".encode("utf-8", errors="ignore")
        return hashlib.sha256(payload).hexdigest()

    def get(self, key: str) -> list[AtomicClaim] | None:
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            if entry.expires_at < now:
                self._store.pop(key, None)
                self.misses += 1
                return None
            self.hits += 1
            # Return copies so callers can mutate consensus without poisoning cache
            return [c.model_copy(deep=True) for c in entry.claims]

    def put(self, key: str, claims: list[AtomicClaim]) -> None:
        with self._lock:
            if len(self._store) >= self.max_entries:
                # Drop expired first, then oldest arbitrary key
                now = time.monotonic()
                expired = [k for k, v in self._store.items() if v.expires_at < now]
                for k in expired:
                    self._store.pop(k, None)
                while len(self._store) >= self.max_entries:
                    self._store.pop(next(iter(self._store)))
            self._store[key] = _CacheEntry(
                claims=[c.model_copy(deep=True) for c in claims],
                expires_at=time.monotonic() + self.ttl_seconds,
            )

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._store),
                "hits": self.hits,
                "misses": self.misses,
            }


# Shared singleton for the research API process
extraction_cache = ClaimExtractionCache()
