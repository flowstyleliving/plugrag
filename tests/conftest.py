from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from lens_rag.manifest import Manifest


VOCAB = {
    "school": 0,
    "schools": 0,
    "education": 1,
    "board": 2,
    "jail": 3,
    "prison": 3,
    "sheriff": 4,
    "police": 5,
    "zoning": 6,
    "storm": 7,
    "budget": 8,
    "charter": 9,
}


class FakeEmbeddingModel:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(text) for text in texts]

    def _encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for raw_token in text.lower().replace("-", " ").split():
            token = "".join(char for char in raw_token if char.isalnum())
            if token in VOCAB:
                vector[VOCAB[token]] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            vector[-1] = 1.0
            return vector
        return [value / norm for value in vector]


class FakeResponse:
    def __init__(
        self,
        payload: Any,
        headers: dict[str, str] | None = None,
        status_code: int = 200,
    ) -> None:
        self.payload = payload
        self.headers = headers or {}
        self.status_code = status_code

    def json(self) -> Any:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeWordPressSession:
    def __init__(self, articles: list[dict[str, Any]]) -> None:
        self.articles = articles
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, params: dict[str, Any], timeout: float) -> FakeResponse:
        endpoint = url.rstrip("/").rsplit("/", 1)[-1]
        self.calls.append((endpoint, dict(params)))
        if endpoint == "categories":
            return FakeResponse(
                [
                    {"id": 10, "name": "Education", "slug": "education"},
                    {"id": 20, "name": "Criminal Justice", "slug": "criminal-justice"},
                ],
                {"X-WP-TotalPages": "1"},
            )
        if endpoint == "tags":
            return FakeResponse(
                [
                    {"id": 100, "name": "Schools", "slug": "schools"},
                    {"id": 200, "name": "Jail", "slug": "jail"},
                ],
                {"X-WP-TotalPages": "1"},
            )
        if endpoint == "posts":
            per_page = int(params.get("per_page", 100))
            page = int(params.get("page", 1))
            start = (page - 1) * per_page
            end = start + per_page
            total_pages = max(1, math.ceil(len(self.articles) / per_page))
            return FakeResponse(
                self.articles[start:end],
                {
                    "X-WP-Total": str(len(self.articles)),
                    "X-WP-TotalPages": str(total_pages),
                },
            )
        raise AssertionError(f"Unexpected endpoint: {endpoint}")


@pytest.fixture
def fake_model() -> FakeEmbeddingModel:
    return FakeEmbeddingModel()


@pytest.fixture
def sample_wp_article() -> dict[str, Any]:
    return make_wp_article(
        1,
        "School Board budget questions",
        "The school board discussed education funding and charter schools.",
        categories=[10],
        tags=[100],
    )


def make_wp_article(
    article_id: int,
    title: str,
    body: str,
    categories: list[int] | None = None,
    tags: list[int] | None = None,
    date: str = "2024-01-15T10:00:00",
) -> dict[str, Any]:
    return {
        "id": article_id,
        "date": date,
        "date_gmt": date,
        "modified": date,
        "modified_gmt": date,
        "slug": f"article-{article_id}",
        "link": f"https://thelensnola.org/{article_id}/article-{article_id}/",
        "title": {"rendered": title},
        "excerpt": {"rendered": f"<p>{body[:80]}</p>"},
        "content": {
            "rendered": f"<p>{body}</p><p>Additional paragraph for article {article_id}.</p>"
        },
        "categories": categories or [],
        "tags": tags or [],
    }


def write_downloaded_article(data_dir: Path, article: dict[str, Any]) -> None:
    articles_dir = data_dir / "articles"
    articles_dir.mkdir(parents=True, exist_ok=True)
    path = articles_dir / f"{article['id']}.json"
    import json

    with path.open("w", encoding="utf-8") as handle:
        json.dump(article, handle)
    manifest = Manifest.load(data_dir / "manifest.json")
    manifest.mark_downloaded(article)
    manifest.save()
