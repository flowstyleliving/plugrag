from __future__ import annotations

import json
from pathlib import Path

from lens_rag.knowledge_graph import KnowledgeGraphIndex, build_knowledge_graph
from lens_rag.query_engine import QueryEngine


def write_article(
    data_dir: Path,
    article_id: str,
    title: str,
    date: str,
    tags: list[str],
    text: str,
) -> None:
    articles = data_dir / "articles"
    articles.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": int(article_id),
        "title": title,
        "date": date,
        "url": f"https://example.test/{article_id}",
        "plain_text": text,
        "category_names": ["Criminal Justice"],
        "tag_names": tags,
        "author_name": "Lens Reporter",
    }
    (articles / f"{article_id}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def graph_fixture(tmp_path: Path) -> Path:
    write_article(
        tmp_path,
        "1",
        "Court orders reforms at Orleans Parish Jail",
        "2014-05-01T10:00:00",
        ["Orleans Parish Jail", "Consent Decree", "Marlin Gusman"],
        "The federal court ordered staffing reforms at Orleans Parish Jail. "
        "The Orleans Parish Sheriff's Office (OPSO) was required to report progress.",
    )
    write_article(
        tmp_path,
        "2",
        "Monitor finds jail staffing delays",
        "2018-06-01T10:00:00",
        ["Orleans Parish Jail", "Consent Decree", "Court Monitor"],
        "A court monitor found staffing delays at Orleans Parish Jail under "
        "the federal consent decree.",
    )
    write_article(
        tmp_path,
        "3",
        "Sheriff responds to oversight report",
        "2023-07-01T10:00:00",
        ["Orleans Parish Jail", "Court Monitor", "Susan Hutson"],
        "Sheriff Susan Hutson responded to the monitor's latest oversight report.",
    )
    return tmp_path / "knowledge_graph.sqlite3"


def test_build_and_search_persistent_knowledge_graph(tmp_path: Path) -> None:
    graph_path = graph_fixture(tmp_path)

    stats = build_knowledge_graph(tmp_path, graph_path)
    index = KnowledgeGraphIndex(graph_path)
    result = index.search("Orleans Parish jail consent decree", limit=5)

    assert stats["article_count"] == 3
    assert stats["entity_count"] >= 5
    assert stats["edge_count"] > 0
    assert stats["community_count"] >= 1
    assert index.metadata()["available"] is True
    assert {hit.article_id for hit in result.hits} >= {"1", "2"}
    assert "Orleans Parish Jail" in result.matched_entities
    assert result.communities

    alias_result = index.search("What did OPSO report?", limit=3)
    assert "Orleans Parish Sheriff's Office" in alias_result.matched_entities


def test_query_engine_turns_graph_paths_into_article_evidence(tmp_path: Path) -> None:
    graph_path = graph_fixture(tmp_path)
    build_knowledge_graph(tmp_path, graph_path)
    engine = QueryEngine(data_dir=tmp_path)

    results, diagnostics = engine.graph_search_results(
        "jail consent decree oversight",
        top_k=3,
    )

    assert results
    assert diagnostics["enabled"] is True
    assert diagnostics["matched_entities"]
    assert all(result.matches for result in results)
    assert all(result.matches[0].chunk_index < 0 for result in results)
