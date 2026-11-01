from __future__ import annotations

import pytest

from lens_rag.scraper import LensAPIError, LensScraper
from lens_rag.text import html_to_plain_text

from .conftest import FakeResponse, FakeWordPressSession, make_wp_article


def test_api_connectivity_optional() -> None:
    requests = pytest.importorskip("requests")
    try:
        response = requests.get(
            "https://thelensnola.org/wp-json/wp/v2/posts",
            params={"per_page": 1},
            timeout=5,
        )
    except Exception as exc:
        pytest.skip(f"Live API unavailable in this environment: {exc}")
    assert response.status_code == 200
    assert "X-WP-Total" in response.headers


def test_html_stripping_preserves_paragraphs() -> None:
    text = html_to_plain_text("<p>First&nbsp;paragraph.</p><p>Second<br>line.</p>")
    assert "<p>" not in text
    assert "First paragraph." in text
    assert "Second" in text
    assert "line." in text
    assert "\n\n" in text


def test_article_parsing_resolves_terms(tmp_path, sample_wp_article) -> None:
    scraper = LensScraper(
        data_dir=tmp_path,
        rate_limit_seconds=0,
        session=FakeWordPressSession([sample_wp_article]),
    )
    article = scraper.normalize_article(
        sample_wp_article,
        {10: {"name": "Education", "slug": "education"}},
        {100: {"name": "Schools", "slug": "schools"}},
    )
    assert article["title"] == "School Board budget questions"
    assert article["category_names"] == ["Education"]
    assert article["tag_names"] == ["Schools"]
    assert "school board discussed" in article["plain_text"].lower()
    assert article["content_hash"]


def test_manifest_tracking_and_resume_logic(tmp_path) -> None:
    articles = [
        make_wp_article(1, "School article", "Education and school board coverage.", [10], [100]),
        make_wp_article(2, "Jail article", "Jail oversight and sheriff coverage.", [20], [200]),
    ]
    session = FakeWordPressSession(articles)
    scraper = LensScraper(data_dir=tmp_path, rate_limit_seconds=0, session=session)
    first_stats = scraper.scrape_all(show_progress=False)
    assert first_stats["downloaded"] == 2
    assert first_stats["skipped"] == 0
    assert (tmp_path / "articles" / "1.json").exists()

    second_scraper = LensScraper(
        data_dir=tmp_path,
        rate_limit_seconds=0,
        session=FakeWordPressSession(articles),
    )
    second_stats = second_scraper.scrape_all(show_progress=False)
    assert second_stats["downloaded"] == 0
    assert second_stats["skipped"] == 2


def test_fetch_json_retries_transient_errors(tmp_path) -> None:
    class FlakySession:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, url, params, timeout):
            self.calls += 1
            if self.calls == 1:
                return FakeResponse({}, status_code=503)
            return FakeResponse({"ok": True}, status_code=200)

    session = FlakySession()
    scraper = LensScraper(
        data_dir=tmp_path,
        rate_limit_seconds=0,
        max_retries=2,
        session=session,
    )
    payload, _ = scraper.fetch_json("posts", {})
    assert payload == {"ok": True}
    assert session.calls == 2


def test_fetch_json_raises_after_retries(tmp_path) -> None:
    class BrokenSession:
        def get(self, url, params, timeout):
            return FakeResponse({}, status_code=503)

    scraper = LensScraper(
        data_dir=tmp_path,
        rate_limit_seconds=0,
        max_retries=2,
        session=BrokenSession(),
    )
    with pytest.raises(LensAPIError):
        scraper.fetch_json("posts", {})
