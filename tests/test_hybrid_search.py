from __future__ import annotations

from pathlib import Path
from typing import Any

from lens_rag.hybrid_search import LexicalMatch, SQLiteFTSIndex
from lens_rag.query_engine import fuse_dense_and_lexical, lexical_query_terms


class FakeCollection:
    def __init__(self) -> None:
        self.rows = [
            (
                "1:0",
                "Content: residents sought relocation assistance.",
                {
                    "article_id": "1",
                    "title": "Gordon Plaza relocation",
                    "date": "2022-06-01",
                    "date_int": 20220601,
                    "chunk_index": 0,
                    "categories": "Land Use",
                    "tags": "Housing",
                    "url": "https://example.test/1",
                },
            ),
            (
                "3:0",
                "Title: Jail oversight hearing\nCategories: Criminal Justice\n\nContent: public testimony continued.",
                {
                    "article_id": "3",
                    "title": "",
                    "date": "2019-06-01",
                    "date_int": 20190601,
                    "chunk_index": 0,
                    "categories": "Criminal Justice",
                    "tags": "",
                    "url": "https://example.test/3",
                },
            ),
            (
                "2:0",
                "Content: The school board discussed its annual budget.",
                {
                    "article_id": "2",
                    "title": "School board budget",
                    "date": "2024-01-15",
                    "date_int": 20240115,
                    "chunk_index": 0,
                    "categories": "Schools",
                    "tags": "Budget",
                    "url": "https://example.test/2",
                },
            ),
        ]

    def count(self) -> int:
        return len(self.rows)

    def get(self, limit: int, offset: int, include: list[str]) -> dict[str, Any]:
        selected = self.rows[offset : offset + limit]
        return {
            "ids": [row[0] for row in selected],
            "documents": [row[1] for row in selected],
            "metadatas": [row[2] for row in selected],
        }


def test_fts_finds_full_corpus_exact_phrase_and_filters(tmp_path: Path) -> None:
    collection = FakeCollection()
    index = SQLiteFTSIndex(tmp_path / "search.sqlite3")

    matches = index.search(
        collection,
        ["Gordon Plaza"],
        category="Land Use",
        date_from=20200101,
        limit=10,
    )

    assert [match.chunk_id for match in matches] == ["1:0"]
    assert matches[0].metadata["url"] == "https://example.test/1"


def test_fts_multiword_terms_also_match_component_tokens(tmp_path: Path) -> None:
    collection = FakeCollection()
    index = SQLiteFTSIndex(tmp_path / "search.sqlite3")

    matches = index.search(collection, ["Gordon relocation"], limit=10)

    assert [match.chunk_id for match in matches] == ["1:0"]


def test_fts_indexes_title_when_body_does_not_contain_terms(tmp_path: Path) -> None:
    collection = FakeCollection()
    index = SQLiteFTSIndex(tmp_path / "search.sqlite3")

    matches = index.search(
        collection,
        ["jail oversight"],
        category="Criminal Justice",
        limit=10,
    )

    assert [match.chunk_id for match in matches] == ["3:0"]
    assert matches[0].metadata["title"] == "Jail oversight hearing"


def test_reciprocal_rank_fusion_promotes_dense_lexical_agreement() -> None:
    dense = {
        "ids": [["dense-only", "both"]],
        "documents": [["dense", "both"]],
        "metadatas": [[
            {"article_id": "1", "chunk_index": 0},
            {"article_id": "2", "chunk_index": 0},
        ]],
        "distances": [[0.2, 0.25]],
    }
    lexical = [
        LexicalMatch(
            chunk_id="both",
            document="both",
            metadata={"article_id": "2", "chunk_index": 0},
        )
    ]

    fused = fuse_dense_and_lexical(dense, lexical)

    assert fused["ids"][0][0] == "both"


def test_lexical_only_hits_are_scored_as_competitive_candidates() -> None:
    dense = {
        "ids": [["dense-only"]],
        "documents": [["dense"]],
        "metadatas": [[{"article_id": "1", "chunk_index": 0}]],
        "distances": [[0.4]],
    }
    lexical = [
        LexicalMatch(
            chunk_id="title-only",
            document="Title: Jail oversight hearing\n\nContent: public testimony",
            metadata={"article_id": "2", "chunk_index": 0},
        )
    ]

    fused = fuse_dense_and_lexical(dense, lexical)
    lexical_index = fused["ids"][0].index("title-only")

    assert fused["distances"][0][lexical_index] <= 0.32


def test_lexical_query_terms_preserve_rewritten_phrase() -> None:
    terms = lexical_query_terms(
        "jail oversight",
        ["jail", "oversight", "opp"],
    )

    assert terms[0] == "jail oversight"
    assert "opp" in terms
