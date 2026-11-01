from __future__ import annotations

import pytest

from lens_rag.embedder import ArticleEmbedder, get_chroma_collection
from lens_rag.query_engine import QueryEngine
from lens_rag.scraper import LensScraper

from .conftest import FakeWordPressSession, make_wp_article


def test_end_to_end_scrape_embed_query_five_articles(tmp_path, fake_model) -> None:
    pytest.importorskip("chromadb")
    articles = [
        make_wp_article(1, "School Board budget", "School board education budget hearing.", [10], [100]),
        make_wp_article(2, "Jail oversight", "Jail conditions and sheriff oversight.", [20], [200]),
        make_wp_article(3, "Charter schools", "Charter school accountability and education policy.", [10], [100]),
        make_wp_article(4, "Police spending", "Police budget details and public safety.", [20], [200]),
        make_wp_article(5, "Storm repairs", "Storm drainage repairs and infrastructure.", [10], []),
    ]

    scraper = LensScraper(
        data_dir=tmp_path,
        rate_limit_seconds=0,
        session=FakeWordPressSession(articles),
    )
    scrape_stats = scraper.scrape_all(limit=5, show_progress=False)
    assert scrape_stats["downloaded"] == 5

    collection = get_chroma_collection(tmp_path / "chroma_db", "test_pipeline")
    embed_stats = ArticleEmbedder(
        data_dir=tmp_path,
        model=fake_model,
        collection=collection,
        chunk_size=50,
        overlap=5,
    ).embed_all()
    assert embed_stats["articles_embedded"] == 5

    results = QueryEngine(
        data_dir=tmp_path,
        model=fake_model,
        collection=collection,
    ).search("school board education", top_k=2, category="Education")
    assert results
    assert results[0].article_id in {"1", "3"}
    assert "Education" in results[0].categories
