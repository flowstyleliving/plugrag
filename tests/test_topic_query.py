from __future__ import annotations

from typing import Any

from lens_rag.query_engine import QueryEngine
from lens_rag.hybrid_search import LexicalMatch
from lens_rag.topics import TopicLexicon, TopicRecord


class FakeModel:
    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeCollection:
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.response = response or {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    def count(self) -> int:
        return 1

    def query(self, **_: Any) -> dict[str, Any]:
        return self.response


class CapturingFTS:
    def __init__(self) -> None:
        self.terms: list[str] = []

    def search(self, collection: Any, terms: list[str], **_: Any) -> list[Any]:
        self.terms = list(terms)
        return []


class GoldenFTS:
    def search(self, collection: Any, terms: list[str], **_: Any) -> list[LexicalMatch]:
        joined = " ".join(terms)
        matches: list[LexicalMatch] = []
        if "jail" in joined or "consent decree" in joined:
            matches.append(
                lexical_match("2:0", "2", "Jail oversight", "Jail consent decree monitor.")
            )
        if "school" in joined or "board" in joined:
            matches.append(
                lexical_match("1:0", "1", "School board budget", "School budget hearing.")
            )
        if "airbnb" in joined or "rental" in joined:
            matches.append(
                lexical_match("3:0", "3", "Airbnb regulations", "Short-term rental rules.")
            )
        return matches


def lexical_match(chunk_id: str, article_id: str, title: str, document: str) -> LexicalMatch:
    return LexicalMatch(
        chunk_id=chunk_id,
        document=f"Title: {title}\n\nContent: {document}",
        metadata={
            "article_id": article_id,
            "title": title,
            "date": "2020-01-01",
            "date_int": 20200101,
            "chunk_index": 0,
            "categories": "News",
            "tags": "",
            "url": f"https://example.test/{article_id}",
        },
    )


def test_topic_expansion_reaches_fts_after_original_terms() -> None:
    fts = CapturingFTS()
    lexicon = TopicLexicon(
        [
            TopicRecord(
                label="jail oversight",
                terms=["jail oversight", "consent decree", "compliance director"],
                count=3,
            )
        ]
    )

    QueryEngine(
        data_dir="missing-data",
        model=FakeModel(),
        collection=FakeCollection(),
        fts_index=fts,
        topic_lexicon=lexicon,
    ).search_results("jail oversight", nl_rewrite=False)

    assert fts.terms[:3] == ["jail oversight", "jail", "oversight"]
    assert "consent decree" in fts.terms[3:]
    assert "compliance director" in fts.terms[3:]


def test_missing_or_corrupt_topic_lexicon_is_noop(tmp_path) -> None:
    (tmp_path / "topic_lexicon.json").write_text("{not-json", encoding="utf-8")
    fts = CapturingFTS()

    QueryEngine(
        data_dir=tmp_path,
        model=FakeModel(),
        collection=FakeCollection(),
        fts_index=fts,
    ).search_results("jail oversight", nl_rewrite=False)

    assert fts.terms == ["jail oversight", "jail", "oversight", "jail", "oversight"]


def test_topic_expansion_does_not_enter_keyword_scoring() -> None:
    lexicon = TopicLexicon(
        [
            TopicRecord(
                label="jail oversight",
                terms=["jail oversight", "consent decree"],
                count=3,
            )
        ]
    )
    response = {
        "ids": [["1:0"]],
        "documents": [["Content: consent decree details but no query words"]],
        "metadatas": [[
            {
                "article_id": "1",
                "title": "Court filing",
                "date": "2020-01-01",
                "url": "https://example.test/1",
                "categories": "Criminal Justice",
                "tags": "",
                "chunk_index": 0,
            }
        ]],
        "distances": [[0.5]],
    }

    results = QueryEngine(
        data_dir="missing-data",
        model=FakeModel(),
        collection=FakeCollection(response),
        fts_index=CapturingFTS(),
        topic_lexicon=lexicon,
    ).search_results("jail oversight", nl_rewrite=False).results

    assert results
    assert "consent decree" not in results[0].score_info.matched_terms


def test_exact_title_only_fts_terms_remain_first_with_topic_expansion() -> None:
    fts = CapturingFTS()
    lexicon = TopicLexicon(
        [
            TopicRecord(
                label="jail oversight",
                terms=["jail oversight", "consent decree"],
                count=3,
            )
        ]
    )

    QueryEngine(
        data_dir="missing-data",
        model=FakeModel(),
        collection=FakeCollection(),
        fts_index=fts,
        topic_lexicon=lexicon,
    ).search_results("jail oversight", nl_rewrite=False)

    assert fts.terms[0] == "jail oversight"


def test_golden_queries_return_expected_articles_with_and_without_topics() -> None:
    golden = [
        ("jail oversight", "2"),
        ("school board budget", "1"),
        ("Airbnb regulations", "3"),
    ]
    lexicons = [
        TopicLexicon(),
        TopicLexicon(
            [
                TopicRecord(
                    label="jail oversight",
                    terms=["jail oversight", "consent decree", "compliance director"],
                    count=3,
                ),
                TopicRecord(
                    label="airbnb",
                    terms=["airbnb", "short term rental", "vacation rental"],
                    count=3,
                ),
            ]
        ),
    ]

    for lexicon in lexicons:
        engine = QueryEngine(
            data_dir="missing-data",
            model=FakeModel(),
            collection=FakeCollection(),
            fts_index=GoldenFTS(),
            topic_lexicon=lexicon,
        )
        for query, expected_id in golden:
            results = engine.search_results(query, nl_rewrite=False).results
            assert [result.article_id for result in results[:3]].count(expected_id) == 1
