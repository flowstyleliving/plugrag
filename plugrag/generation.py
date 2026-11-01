"""Optional OpenAI-compatible generation with the existing citation safeguards."""

from __future__ import annotations

import os
from typing import Any

from lens_rag.nl_query import AnswerGenerator
from lens_rag.query_engine import ArticleResult, ChunkMatch


class CitedGenerator:
    """Works with a local compatible server or a remote compatible endpoint.

    An explicit client may also be injected. Citation repair, out-of-range citation
    removal and rejection/pruning of uncited sentences use the shared generator.
    These checks enforce citation structure, not semantic truth.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str = "local",
        max_tokens: int = 900,
        timeout: float = 120,
        client: Any = None,
    ) -> None:
        if not model.strip() or not base_url.strip():
            raise ValueError("generation requires model and base_url")
        if max_tokens <= 0 or timeout <= 0:
            raise ValueError("max_tokens and timeout must be positive")
        self.backend = AnswerGenerator(
            model=model,
            base_url=base_url,
            api_key=api_key,
            provider="compatible",
            max_tokens=max_tokens,
            timeout=timeout,
            client=client,
            audience="the user",
        )

    def generate(self, query: str, sources: list[dict[str, Any]]) -> str:
        results = [
            ArticleResult(
                article_id=source["id"],
                title=source["title"],
                date=source["date"],
                url=source["url"],
                categories=source["categories"],
                tags=source["tags"],
                score=source["score"],
                matches=[ChunkMatch(**match) for match in source["matches"]],
            )
            for source in sources
        ]
        return self.backend.generate(query, results)


def generator_from_env() -> CitedGenerator | None:
    model = os.getenv("RAG_LLM_MODEL", "")
    base_url = os.getenv("RAG_LLM_BASE_URL", "")
    if not model and not base_url:
        return None
    return CitedGenerator(
        model=model,
        base_url=base_url,
        api_key=os.getenv("RAG_LLM_API_KEY", "local"),
        max_tokens=int(os.getenv("RAG_LLM_MAX_TOKENS", "900")),
        timeout=float(os.getenv("RAG_LLM_TIMEOUT", "120")),
    )
