"""Exa API client wrapper for neural web search."""

from __future__ import annotations

import logging
from typing import Any

from exa_py import Exa

from app.config import Settings, get_settings
from app.models.schemas import SearchResult

logger = logging.getLogger(__name__)

# Cap page text so claim extraction stays token-bounded (Exa contents.text).
_DEFAULT_MAX_TEXT_CHARS = 20_000


class ExaService:
    """Thin async-friendly wrapper around the official Exa SDK."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.exa_api_key:
            logger.warning("EXA_API_KEY is not set; ExaService calls will fail")
        self._client = Exa(api_key=self._settings.exa_api_key or None)

    async def search(
        self,
        query: str,
        *,
        num_results: int = 10,
        include_text: bool = True,
        search_type: str = "auto",
        max_characters: int = _DEFAULT_MAX_TEXT_CHARS,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """
        Run a neural search against Exa and normalize hits.

        Uses the current ``Exa.search`` API with content nested under
        ``contents`` (not the deprecated ``search_and_contents`` / top-level
        ``text=True``).

        The Exa SDK is synchronous; we keep the call surface async so callers
        can later offload to a thread pool without changing signatures.
        """
        search_kwargs: dict[str, Any] = {
            "num_results": num_results,
            "type": search_type,
            **kwargs,
        }

        # Callers may pass ``contents=`` explicitly; otherwise build a default.
        if "contents" not in search_kwargs:
            if include_text:
                search_kwargs["contents"] = {
                    "text": {"max_characters": max_characters},
                }
            else:
                # URL/title-only retrieval (fastest).
                search_kwargs["contents"] = False

        response = self._client.search(query, **search_kwargs)
        return [self._to_search_result(item) for item in response.results]

    @staticmethod
    def _to_search_result(item: Any) -> SearchResult:
        """Map an Exa result object into our schema."""
        # Prefer full text; fall back to joined highlights when text is absent.
        text = getattr(item, "text", None)
        if not text:
            highlights = getattr(item, "highlights", None) or []
            if highlights:
                text = "\n".join(str(h) for h in highlights)

        return SearchResult(
            id=getattr(item, "id", None) or getattr(item, "url", "") or "",
            title=getattr(item, "title", None),
            url=getattr(item, "url", None),
            text=text,
            score=getattr(item, "score", None),
            published_date=getattr(item, "published_date", None),
            author=getattr(item, "author", None),
            metadata={},
        )
