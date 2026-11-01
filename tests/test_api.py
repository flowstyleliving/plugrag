from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from lens_rag.api import (
    DEFAULT_API_PORT,
    _answer_source_limit,
    _local_api_port,
    create_app,
)
from lens_rag.query_engine import ArticleResult, ChunkMatch


def test_local_api_port_defaults_to_project_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LENS_RAG_API_PORT", raising=False)
    assert DEFAULT_API_PORT == 5367
    assert _local_api_port() == DEFAULT_API_PORT


def test_local_api_port_accepts_override_and_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LENS_RAG_API_PORT", "5743")
    assert _local_api_port() == 5743

    for invalid_port in ("not-a-port", "1023", "65536"):
        monkeypatch.setenv("LENS_RAG_API_PORT", invalid_port)
        assert _local_api_port() == DEFAULT_API_PORT


class DummyCollection:
    def count(self) -> int:
        return 12


class DummyEngine:
    def __init__(self) -> None:
        self.collection = DummyCollection()
        self.model_name = "test-embedding-model"
        self.calls: list[dict[str, Any]] = []

    def _collection(self) -> DummyCollection:
        return self.collection

    def search(
        self,
        query: str,
        top_k: int = 10,
        category: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[ArticleResult]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "category": category,
                "date_from": date_from,
                "date_to": date_to,
            }
        )
        return [
            ArticleResult(
                article_id="1",
                title="School Board budget",
                date="2024-01-15T10:00:00",
                url="https://example.test/1",
                categories="Education",
                tags="Schools",
                score=0.91,
                matches=[
                    ChunkMatch(
                        score=0.91,
                        excerpt="The school board discussed education funding.",
                        chunk_index=0,
                    )
                ],
            )
        ]


class FakeAnswerGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[ArticleResult]]] = []

    def supports_answering(self) -> bool:
        return True

    def generate(self, query: str, results: list[ArticleResult]) -> str:
        self.calls.append((query, results))
        return "The archive shows education budget coverage [1]."


class FakeDeepSeekAnswerGenerator:
    provider = "deepseek"
    model = "deepseek-v4-flash"
    base_url = "https://api.deepseek.com/v1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[ArticleResult], str]] = []

    def supports_answering(self) -> bool:
        return True

    def generate(
        self,
        query: str,
        results: list[ArticleResult],
        draft_answer: str | None = None,
    ) -> str:
        self.calls.append((query, results, draft_answer))
        return "DeepSeek audited the education budget coverage [1]."


class FakeOpusAnswerGenerator:
    provider = "anthropic-direct"
    model = "claude-opus-5"
    base_url = "https://api.anthropic.com"
    last_diagnostics = {
        "verification_mode": "Opus claim audit",
        "provider_route": "direct",
    }

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[ArticleResult], dict[str, Any]]] = []

    def supports_answering(self) -> bool:
        return True

    def generate_research(
        self,
        query: str,
        results: list[ArticleResult],
        **kwargs: Any,
    ) -> str:
        self.calls.append((query, results, kwargs))
        return "Opus connected the education budget reporting over time [1]."


@pytest.fixture
def api_client(tmp_path: Path) -> tuple[TestClient, DummyEngine]:
    write_article(tmp_path, "1", "School Board budget", ["Education"], ["education"])
    write_article(tmp_path, "2", "Jail oversight", ["Criminal Justice"], ["criminal-justice"])
    write_manifest(tmp_path)
    engine = DummyEngine()
    return TestClient(create_app(data_dir=tmp_path, query_engine=engine)), engine


def test_health_returns_article_and_collection_counts(api_client) -> None:
    client, _ = api_client
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["articles"]["manifest_count"] == 2
    assert payload["articles"]["file_count"] == 2
    assert payload["collection"]["chunk_count"] == 12
    assert payload["collection"]["embedding_model"] == "test-embedding-model"


def test_frontend_response_disables_stale_browser_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frontend_dir = tmp_path / "frontend"
    frontend_dir.mkdir()
    (frontend_dir / "index.html").write_text("<h1>Lens RAG</h1>", encoding="utf-8")
    monkeypatch.setenv("LENS_RAG_FRONTEND_REVISION", "opus-tab-c-test")
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            frontend_dir=frontend_dir,
            query_engine=DummyEngine(),
        )
    )

    response = client.get("/app")

    assert response.status_code == 200
    assert str(response.url).endswith("/app?v=opus-tab-c-test")
    assert response.history[0].status_code == 307
    assert response.headers["cache-control"] == (
        "no-store, no-cache, must-revalidate, max-age=0, private"
    )
    assert response.headers["clear-site-data"] == '"cache"'
    assert response.headers["x-lens-frontend-revision"] == "opus-tab-c-test"


def test_health_reports_local_answer_backend(api_client, monkeypatch) -> None:
    monkeypatch.setenv("LENS_RAG_ENABLE_ANSWER", "true")
    monkeypatch.setenv("LENS_RAG_LLM_PROVIDER", "local")
    monkeypatch.setenv("LENS_RAG_ANSWER_MODEL", "mlx-community/Qwen3-8B-4bit")
    client, _ = api_client

    payload = client.get("/").json()

    assert payload["features"]["answer_generation"] is True
    assert payload["features"]["answer_backend"] == {
        "provider": "local",
        "model": "mlx-community/Qwen3-8B-4bit",
        "configured": True,
    }


def test_search_uses_query_engine_with_filters(api_client) -> None:
    client, engine = api_client
    response = client.post(
        "/search",
        json={
            "query": "school budget",
            "top_k": 3,
            "category": "Education",
            "date_from": "2024-01-01",
            "date_to": "2024-12-31",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["results"][0]["article_id"] == "1"
    assert payload["results"][0]["matches"][0]["excerpt"].startswith("The school")
    assert engine.calls[0]["category"] == "Education"
    assert engine.calls[0]["top_k"] == 3


def test_article_endpoint_returns_full_text(api_client) -> None:
    client, _ = api_client
    response = client.get("/article/1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["title"] == "School Board budget"
    assert payload["categories"] == ["Education"]
    assert "Full article text for School Board budget" in payload["content"]


def test_articles_endpoint_filters_by_category_and_date(api_client) -> None:
    client, _ = api_client
    response = client.get(
        "/articles",
        params={"category": "criminal-justice", "date_from": "2023-01-01"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["title"] == "Jail oversight"


def test_categories_returns_counts(api_client) -> None:
    client, _ = api_client
    response = client.get("/categories")
    assert response.status_code == 200
    payload = response.json()
    names = {item["name"]: item["count"] for item in payload["categories"]}
    assert names == {"Education": 1, "Criminal Justice": 1}


def test_search_answer_disabled_by_default(api_client, monkeypatch) -> None:
    monkeypatch.delenv("LENS_RAG_ENABLE_ANSWER", raising=False)
    client, _ = api_client
    response = client.post("/search/answer", json={"query": "school budget"})
    assert response.status_code == 501


def test_search_answer_returns_brief_and_sources(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LENS_RAG_ENABLE_ANSWER", "true")
    write_article(tmp_path, "1", "School Board budget", ["Education"], ["education"])
    write_manifest(tmp_path)
    engine = DummyEngine()
    generator = FakeAnswerGenerator()
    client = TestClient(
        create_app(data_dir=tmp_path, query_engine=engine, answer_generator=generator)
    )

    response = client.post(
        "/search/answer",
        json={"query": "school budget", "top_k": 3, "category": "Education"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "school budget"
    assert payload["answer"] == "The archive shows education budget coverage [1]."
    assert payload["result_count"] == 1
    assert payload["sources"] == [
        {
            "index": 1,
            "article_id": "1",
            "title": "School Board budget",
            "date": "2024-01-15T10:00:00",
            "url": "https://example.test/1",
        }
    ]
    assert engine.calls[0]["category"] == "Education"
    assert generator.calls[0][0] == "school budget"


def test_deepseek_answer_refines_completed_local_draft(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LENS_RAG_ENABLE_DEEPSEEK", "true")
    write_article(tmp_path, "1", "School Board budget", ["Education"], ["education"])
    write_manifest(tmp_path)
    generator = FakeDeepSeekAnswerGenerator()
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            query_engine=DummyEngine(),
            deepseek_answer_generator=generator,
        )
    )

    response = client.post(
        "/search/answer/deepseek",
        json={
            "query": "school budget",
            "top_k": 3,
            "category": "Education",
            "draft_answer": "The archive shows education budget coverage [1].",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["stage"] == "A"
    assert payload["model"] == "deepseek-v4-flash"
    assert payload["answer"].startswith("DeepSeek audited")
    assert generator.calls[0][2] == "The archive shows education budget coverage [1]."


def test_deepseek_answer_runs_without_local_draft(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LENS_RAG_ENABLE_DEEPSEEK", "true")
    write_article(tmp_path, "1", "School Board budget", ["Education"], ["education"])
    write_manifest(tmp_path)
    generator = FakeDeepSeekAnswerGenerator()
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            query_engine=DummyEngine(),
            deepseek_answer_generator=generator,
        )
    )

    response = client.post("/search/answer/deepseek", json={"query": "school budget"})

    assert response.status_code == 200
    assert response.json()["stage"] == "A"
    assert generator.calls[0][2] is None


def test_deepseek_pro_answer_runs_on_demand(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LENS_RAG_ENABLE_DEEPSEEK", "true")
    write_article(tmp_path, "1", "School Board budget", ["Education"], ["education"])
    write_manifest(tmp_path)
    generator = FakeDeepSeekAnswerGenerator()
    generator.model = "deepseek-v4-pro"
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            query_engine=DummyEngine(),
            deepseek_pro_answer_generator=generator,
        )
    )

    response = client.post(
        "/search/answer/deepseek-pro", json={"query": "school budget"}
    )

    assert response.status_code == 200
    assert response.headers["x-accel-buffering"] == "no"
    assert response.json()["stage"] == "B"
    assert response.json()["model"] == "deepseek-v4-pro"
    diagnostics = response.json()["diagnostics"]
    assert diagnostics["facet_count"] == 7
    assert "money_mechanisms" in diagnostics["facets_covered"]
    assert diagnostics["research_plan"]["research_contract"]
    assert response.json()["diagnostics"]["evidence_item_count"] == 1
    assert len(response.json()["diagnostics"]["facet_result_counts"]) == 7
    assert len(generator.calls[0][1]) == 1
    assert generator.calls[0][2] is None


def test_opus_answer_runs_independently_on_demand(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LENS_RAG_ENABLE_OPUS", "true")
    write_article(tmp_path, "1", "School Board budget", ["Education"], ["education"])
    write_manifest(tmp_path)
    generator = FakeOpusAnswerGenerator()
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            query_engine=DummyEngine(),
            opus_answer_generator=generator,
        )
    )

    health = client.get("/").json()["features"]
    assert health["opus_synthesis"] is True
    assert health["opus_backend"] == {
        "provider": "anthropic-direct",
        "model": "claude-opus-5",
        "base_url": "https://api.anthropic.com",
        "configured": True,
        "provider_route": "direct",
    }

    response = client.post("/search/answer/opus", json={"query": "school budget"})

    assert response.status_code == 200
    assert response.headers["x-accel-buffering"] == "no"
    payload = response.json()
    assert payload["stage"] == "C"
    assert payload["model"] == "claude-opus-5"
    assert payload["diagnostics"]["provider_route"] == "direct"
    assert generator.calls[0][0] == "school budget"
    assert len(generator.calls[0][1]) == 1
    assert "research_plan" in generator.calls[0][2]


def test_openai_answer_runs_independently_and_has_its_own_limit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LENS_RAG_ENABLE_OPENAI", "true")
    monkeypatch.setenv("LENS_RAG_ENABLE_OPUS", "true")
    write_article(tmp_path, "1", "School Board budget", ["Education"], ["education"])
    write_manifest(tmp_path)
    generator = FakeOpusAnswerGenerator()
    generator.provider = "openai"
    generator.model = "gpt-6.1-sol"
    generator.last_diagnostics = {"verification_mode": "atomic claim audit"}
    client = TestClient(create_app(
        data_dir=tmp_path, query_engine=DummyEngine(),
        openai_answer_generator=generator,
        opus_answer_generator=FakeOpusAnswerGenerator(),
    ))
    features = client.get("/health").json()["features"]
    assert features["openai_synthesis"] is True
    assert features["openai_backend"] == {"provider": "openai", "model": "gpt-6.1-sol", "configured": True}
    for _ in range(3):
        response = client.post("/search/answer/openai", json={"query": "school budget"})
        assert response.status_code == 200
        assert response.json()["stage"] == "D"
        assert response.json()["model"] == "gpt-6.1-sol"
        assert response.json()["sources"]
    assert "research_plan" in generator.calls[0][2]
    assert client.post("/search/answer/openai", json={"query": "school budget"}).status_code == 429
    assert client.post("/search/answer/opus", json={"query": "school budget"}).status_code == 200


def test_openai_answer_requires_configuration(api_client, monkeypatch) -> None:
    client, _ = api_client
    monkeypatch.delenv("LENS_RAG_ENABLE_OPENAI", raising=False)
    assert client.post("/search/answer/openai", json={"query": "school budget"}).status_code == 501
    monkeypatch.setenv("LENS_RAG_ENABLE_OPENAI", "true")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LENS_RAG_OPENAI_API_KEY", raising=False)
    assert client.get("/health").json()["features"]["openai_synthesis"] is False
    assert client.post("/search/answer/openai", json={"query": "school budget"}).status_code == 503
    assert client.post("/search/answer/openai", json={"query": "  "}).status_code == 400


def test_opus_answer_is_disabled_by_default(api_client, monkeypatch) -> None:
    monkeypatch.delenv("LENS_RAG_ENABLE_OPUS", raising=False)
    client, _ = api_client

    response = client.post("/search/answer/opus", json={"query": "school budget"})

    assert response.status_code == 501


def test_search_answer_rate_limited(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LENS_RAG_ENABLE_ANSWER", "true")
    write_article(tmp_path, "1", "School Board budget", ["Education"], ["education"])
    write_manifest(tmp_path)
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            query_engine=DummyEngine(),
            answer_generator=FakeAnswerGenerator(),
        )
    )

    for _ in range(10):
        assert client.post("/search/answer", json={"query": "school budget"}).status_code == 200

    response = client.post("/search/answer", json={"query": "school budget"})
    assert response.status_code == 429


def test_answer_source_limit_is_configurable_and_bounded(monkeypatch) -> None:
    monkeypatch.setenv("LENS_RAG_ANSWER_MAX_SOURCES", "3")
    assert _answer_source_limit() == 3

    monkeypatch.setenv("LENS_RAG_ANSWER_MAX_SOURCES", "99")
    assert _answer_source_limit() == 10

    monkeypatch.setenv("LENS_RAG_ANSWER_MAX_SOURCES", "invalid")
    assert _answer_source_limit() == 4


def write_article(
    data_dir: Path,
    article_id: str,
    title: str,
    categories: list[str],
    category_slugs: list[str],
) -> None:
    articles_dir = data_dir / "articles"
    articles_dir.mkdir(parents=True, exist_ok=True)
    article = {
        "id": int(article_id),
        "title": title,
        "date": "2024-01-15T10:00:00" if article_id == "1" else "2023-03-01T10:00:00",
        "modified": "2024-01-16T10:00:00",
        "url": f"https://example.test/{article_id}",
        "slug": f"article-{article_id}",
        "excerpt": f"Excerpt for {title}",
        "plain_text": f"Full article text for {title}.",
        "category_names": categories,
        "category_slugs": category_slugs,
        "tag_names": ["Schools"] if article_id == "1" else ["Jail"],
        "tag_slugs": ["schools"] if article_id == "1" else ["jail"],
    }
    (articles_dir / f"{article_id}.json").write_text(json.dumps(article), encoding="utf-8")


def write_manifest(data_dir: Path) -> None:
    manifest = {
        "schema_version": 1,
        "articles": {
            "1": {
                "id": 1,
                "title": "School Board budget",
                "date": "2024-01-15T10:00:00",
                "url": "https://example.test/1",
                "slug": "article-1",
                "content_hash": "hash-1",
                "embedded": True,
                "chunk_count": 3,
                "category_names": ["Education"],
                "category_slugs": ["education"],
                "tag_names": ["Schools"],
            },
            "2": {
                "id": 2,
                "title": "Jail oversight",
                "date": "2023-03-01T10:00:00",
                "url": "https://example.test/2",
                "slug": "article-2",
                "content_hash": "hash-2",
                "embedded": True,
                "chunk_count": 4,
                "category_names": ["Criminal Justice"],
                "category_slugs": ["criminal-justice"],
                "tag_names": ["Jail"],
            },
        },
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
