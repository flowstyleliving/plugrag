from __future__ import annotations

from lens_rag.query_engine import ArticleResult, ChunkMatch
from lens_rag.reranker import LocalCrossEncoderReranker


class FakeCrossEncoder:
    def predict(self, pairs, show_progress_bar=False):
        assert show_progress_bar is False
        return [-2.0 if "weak" in passage else 3.0 for _, passage in pairs]


def test_local_cross_encoder_reorders_candidates() -> None:
    weak = _result("1", "Weak result", "weak passage", 0.9)
    strong = _result("2", "Strong result", "direct answer", 0.5)
    reranker = LocalCrossEncoderReranker("fake")
    reranker.model = FakeCrossEncoder()

    results = reranker.rerank("reporter question", [weak, strong])

    assert [result.article_id for result in results] == ["2", "1"]


def _result(article_id: str, title: str, excerpt: str, score: float) -> ArticleResult:
    return ArticleResult(
        article_id=article_id,
        title=title,
        date="2024-01-01",
        url=f"https://example.test/{article_id}",
        categories="News",
        tags="",
        score=score,
        matches=[ChunkMatch(score=score, excerpt=excerpt, chunk_index=0)],
    )
