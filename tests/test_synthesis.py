from __future__ import annotations

import json

from lens_rag.embedder import article_to_chunks
from lens_rag.query_engine import ArticleResult, ChunkMatch
from lens_rag.synthesis import (
    EvidenceLedger,
    _coverage_select,
    build_research_context,
    build_research_plan,
    evaluate_synthesis,
)


def article(article_id: str, date: str, excerpt: str) -> ArticleResult:
    return ArticleResult(
        article_id=article_id,
        title=f"Article {article_id}",
        date=date,
        url=f"https://example.test/{article_id}",
        categories="Criminal Justice",
        tags="Oversight",
        score=0.9,
        matches=[ChunkMatch(score=0.9, excerpt=excerpt, chunk_index=0)],
    )


def test_research_plan_covers_temporal_and_investigative_facets() -> None:
    plan = build_research_plan("jail medical care")

    assert [facet.key for facet in plan.facets] == [
        "core",
        "historical_baseline",
        "current_status",
        "accountability",
        "legal_policy",
        "stakes_outcomes",
    ]
    assert all("jail medical care" in facet.query for facet in plan.facets)
    assert "research_contract" in plan.to_dict()


def test_research_plan_adds_facets_required_by_query_intent() -> None:
    plan = build_research_plan(
        "How did jail funding and promised reforms change, and which claims are disputed?"
    )
    keys = {facet.key for facet in plan.facets}

    assert {
        "comparison",
        "money_mechanisms",
        "promises_outcomes",
        "conflicting_accounts",
    }.issubset(keys)
    assert any("like-for-like" in item for item in plan.contract)
    assert any("stated commitments" in item for item in plan.contract)


def test_evidence_ledger_is_chronological_and_source_addressable() -> None:
    results = [
        article("new", "2025-02-01T10:00:00", "A current court order."),
        article("old", "2018-03-01T10:00:00", "The original complaint."),
    ]
    ledger = EvidenceLedger.from_results(
        results,
        {"new": ("current_status",), "old": ("historical_baseline",)},
    )

    assert [item.article_id for item in ledger.items] == ["old", "new"]
    assert ledger.items[0].source_index == 2
    assert ledger.items[1].source_index == 1
    assert ledger.diagnostics() == {
        "source_count": 2,
        "evidence_item_count": 2,
        "facet_count": 2,
        "facets_covered": ["current_status", "historical_baseline"],
        "date_from": "2018-03-01",
        "date_to": "2025-02-01",
    }


def test_research_context_retrieves_facets_and_deduplicates_sources() -> None:
    class Engine:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def search(self, query: str, **_: object) -> list[ArticleResult]:
            self.calls.append(query)
            if "origin first reported" in query:
                return [article("history", "2016-01-01", "Earlier reporting.")]
            if "latest follow-up" in query:
                return [article("current", "2025-01-01", "Latest reporting.")]
            if "officials agencies" in query:
                return [article("shared", "2024-01-01", "Official response.")]
            if "lawsuit court" in query:
                return [article("shared", "2024-01-01", "The court ruled.")]
            return [article("impact", "2023-01-01", "Reported consequences.")]

    engine = Engine()
    context = build_research_context(
        engine,
        "jail medical care",
        [article("core", "2022-01-01", "Core finding.")],
        max_articles=10,
    )

    assert len(engine.calls) == 5
    assert {result.article_id for result in context.results} == {
        "core",
        "history",
        "current",
        "shared",
        "impact",
    }
    assert context.source_facets["shared"] == ("accountability", "legal_policy")
    assert context.diagnostics()["facet_result_counts"] == {
        "core": 1,
        "historical_baseline": 1,
        "current_status": 1,
        "accountability": 1,
        "legal_policy": 1,
        "stakes_outcomes": 1,
    }


def test_research_context_seeds_oldest_baseline_and_newest_status() -> None:
    class Engine:
        def search(self, query: str, **_: object) -> list[ArticleResult]:
            if "origin first reported" in query:
                return [
                    article("history-new", "2020-01-01", "Later baseline."),
                    article("history-old", "2012-01-01", "Earliest baseline."),
                ]
            if "latest follow-up" in query:
                return [
                    article("status-old", "2019-01-01", "Older status."),
                    article("status-new", "2025-01-01", "Newest status."),
                ]
            return []

    context = build_research_context(
        Engine(),
        "jail medical care",
        [article("core", "2022-01-01", "Core finding.")],
        max_articles=3,
    )

    assert [result.article_id for result in context.results] == [
        "core",
        "history-old",
        "status-new",
    ]


def test_coverage_selection_prefers_time_and_topic_diversity() -> None:
    core = article("core", "2024-01-01", "The jail board reviewed its plan.")
    near_duplicate = article(
        "near",
        "2024-03-01",
        "The jail board reviewed the same plan.",
    )
    near_duplicate.score = 0.99
    diverse = article(
        "diverse",
        "2015-01-01",
        "A federal court issued an earlier oversight order.",
    )
    diverse.score = 0.72
    diverse.categories = "Courts"
    diverse.tags = "Federal oversight"
    accountable = article(
        "official",
        "2021-01-01",
        "The sheriff responded to the monitor.",
    )
    accountable.categories = "Government"
    accountable.tags = "Sheriff"

    selected, _ = _coverage_select(
        {
            "core": [core],
            "current_status": [near_duplicate, diverse],
            "accountability": [accountable],
        },
        max_articles=3,
    )

    assert {result.article_id for result in selected} == {
        "core",
        "diverse",
        "official",
    }


def test_research_context_expands_adjacent_local_chunks(tmp_path) -> None:
    articles_dir = tmp_path / "articles"
    articles_dir.mkdir()
    article_data = {
        "id": "local",
        "title": "A long investigation",
        "date": "2024-01-01T00:00:00",
        "url": "https://example.test/local",
        "category_names": ["Criminal Justice"],
        "tag_names": ["Oversight"],
        "plain_text": " ".join(f"word{index}" for index in range(800)),
    }
    (articles_dir / "local.json").write_text(
        json.dumps(article_data),
        encoding="utf-8",
    )
    chunks = article_to_chunks(article_data)
    assert len(chunks) >= 3

    class Engine:
        data_dir = tmp_path

        def search(self, query: str, **_: object) -> list[ArticleResult]:
            return []

    initial = article(
        "local",
        "2024-01-01T00:00:00",
        "The central matching passage.",
    )
    initial.matches[0].chunk_index = 1
    context = build_research_context(
        Engine(),
        "jail medical care",
        [initial],
        max_articles=1,
    )

    assert context.adjacent_context_count == 2
    assert {match.chunk_index for match in context.results[0].matches} == {0, 1, 2}
    assert context.diagnostics()["adjacent_context_count"] == 2


def test_research_context_includes_graph_paths_and_prompt_communities() -> None:
    class Engine:
        def search(self, query: str, **_: object) -> list[ArticleResult]:
            return []

        def graph_search_results(
            self,
            query: str,
            top_k: int,
        ) -> tuple[list[ArticleResult], dict[str, object]]:
            assert query == "jail oversight"
            assert top_k == 8
            return [article("graph", "2024-01-01", "Related graph evidence.")], {
                "enabled": True,
                "matched_entities": ["Orleans Parish Jail"],
                "related_entities": ["Court Monitor"],
                "communities": [
                    {
                        "label": "Jail oversight",
                        "summary": "Archive community connecting jail oversight.",
                        "score": 4.2,
                    }
                ],
                "candidate_count": 1,
            }

    context = build_research_context(
        Engine(),
        "jail oversight",
        [article("core", "2023-01-01", "Core evidence.")],
        max_articles=4,
    )

    assert [result.article_id for result in context.results] == ["core", "graph"]
    assert context.source_facets["graph"] == ("relational_graph",)
    assert context.diagnostics()["knowledge_graph"]["candidate_count"] == 1
    assert context.prompt_plan()["knowledge_graph_communities"][0]["label"] == (
        "Jail oversight"
    )


def test_evaluate_synthesis_reports_citation_coverage_without_a_model() -> None:
    metrics = evaluate_synthesis(
        "One supported paragraph [1].\n\nAnother uses two sources [1, 3].",
        source_count=3,
    )

    assert metrics == {
        "answer_word_count": 7,
        "citation_marker_count": 2,
        "citation_reference_count": 3,
        "cited_source_count": 2,
        "cited_source_indices": [1, 3],
        "paragraph_citation_coverage": 1.0,
        "sentence_citation_coverage": 1.0,
        "source_citation_utilization": 0.667,
        "invalid_citation_count": 0,
    }
