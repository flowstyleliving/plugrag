from __future__ import annotations

import builtins

import pytest

from lens_rag.embedder import ArticleEmbedder, article_to_chunks, get_chroma_collection
from lens_rag.text import split_text_into_chunks

from .conftest import write_downloaded_article


def test_chunking_logic_uses_overlap(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("no tiktoken in this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    text = " ".join(f"word{i}" for i in range(12))
    chunks = split_text_into_chunks(text, chunk_size=5, overlap=2)
    assert chunks[0].split() == ["word0", "word1", "word2", "word3", "word4"]
    assert chunks[1].split()[:2] == ["word3", "word4"]


def test_fake_embedding_dimensions(fake_model) -> None:
    vector = fake_model.encode_texts(["school board budget"])[0]
    assert len(vector) == 384


def test_article_to_chunks_metadata(sample_wp_article, tmp_path) -> None:
    from lens_rag.scraper import LensScraper

    scraper = LensScraper(
        data_dir=tmp_path,
        rate_limit_seconds=0,
        session=object(),
    )
    article = scraper.normalize_article(
        sample_wp_article,
        {10: {"name": "Education", "slug": "education"}},
        {100: {"name": "Schools", "slug": "schools"}},
    )
    chunks = article_to_chunks(article, chunk_size=20, overlap=5)
    assert chunks
    assert chunks[0].metadata["article_id"] == "1"
    assert chunks[0].metadata["title"] == "School Board budget questions"
    assert chunks[0].metadata["categories"] == "Education"
    assert chunks[0].metadata["tags"] == "Schools"
    assert "Title: School Board budget questions" in chunks[0].document


def test_chromadb_insertion_and_retrieval(tmp_path, fake_model) -> None:
    pytest.importorskip("chromadb")
    article = {
        "id": 1,
        "date": "2024-01-15T10:00:00",
        "modified": "2024-01-15T10:00:00",
        "slug": "school-board-budget",
        "url": "https://thelensnola.org/school-board-budget/",
        "title": "School Board budget questions",
        "excerpt": "Education funding.",
        "plain_text": "The school board discussed education funding and charter schools.",
        "content_html": "<p>The school board discussed education funding.</p>",
        "categories": [10],
        "category_names": ["Education"],
        "category_slugs": ["education"],
        "tags": [100],
        "tag_names": ["Schools"],
        "tag_slugs": ["schools"],
        "content_hash": "abc123",
    }
    write_downloaded_article(tmp_path, article)
    collection = get_chroma_collection(tmp_path / "chroma_db", "test_articles")
    stats = ArticleEmbedder(
        data_dir=tmp_path,
        model=fake_model,
        collection=collection,
        chunk_size=50,
        overlap=5,
    ).embed_all()

    assert stats["articles_embedded"] == 1
    assert collection.count() >= 1

    query_embedding = fake_model.encode_texts(["school board"])[0]
    response = collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        include=["documents", "metadatas", "distances"],
    )
    assert response["metadatas"][0][0]["article_id"] == "1"
    assert "school board" in response["documents"][0][0].lower()
