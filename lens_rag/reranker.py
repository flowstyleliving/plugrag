"""Optional local cross-encoder reranking for the final retrieval candidates."""

from __future__ import annotations

import math
from typing import Any


class LocalCrossEncoderReranker:
    """Rerank article candidates with a local Sentence Transformers model."""

    def __init__(self, model_name: str, max_candidates: int = 40) -> None:
        self.model_name = model_name
        self.max_candidates = max_candidates
        self.model: Any | None = None

    def rerank(self, query: str, results: list[Any]) -> list[Any]:
        if not results or self.max_candidates <= 0:
            return results
        candidates = results[: self.max_candidates]
        pairs = [(query, _result_text(result)) for result in candidates]
        scores = self._model().predict(pairs, show_progress_bar=False)
        for result, raw_score in zip(candidates, scores):
            value = float(raw_score.item() if hasattr(raw_score, "item") else raw_score)
            cross_score = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))
            prior = float(getattr(result, "score", 0.0) or 0.0)
            result.score = round((0.7 * cross_score) + (0.3 * prior), 4)
        candidates.sort(key=lambda result: result.score, reverse=True)
        return candidates + results[self.max_candidates :]

    def _model(self) -> Any:
        if self.model is None:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)
        return self.model


def _result_text(result: Any) -> str:
    parts = [
        str(getattr(result, "title", "") or ""),
        str(getattr(result, "categories", "") or ""),
        str(getattr(result, "tags", "") or ""),
    ]
    for match in (getattr(result, "matches", None) or [])[:2]:
        parts.append(str(getattr(match, "excerpt", "") or ""))
    return "\n".join(part for part in parts if part)
