"""WordPress REST API scraper for The Lens New Orleans."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from .config import (
    ARTICLES_DIRNAME,
    DEFAULT_DATA_DIR,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RATE_LIMIT_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    MANIFEST_FILENAME,
    WP_API_BASE,
)
from .manifest import Manifest
from .text import html_to_plain_text, sha256_text

try:
    import requests
except ImportError:  # pragma: no cover - exercised only without dependencies
    requests = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only without dependencies

    def tqdm(iterable: Any | None = None, **_: Any) -> Any:
        return iterable if iterable is not None else _NullProgress()


class _NullProgress:
    def update(self, _: int) -> None:
        pass

    def close(self) -> None:
        pass


class LensAPIError(RuntimeError):
    """Raised when the Lens WordPress API cannot be fetched successfully."""


class LensScraper:
    """Download and normalize articles from The Lens WordPress API."""

    def __init__(
        self,
        data_dir: Path = DEFAULT_DATA_DIR,
        api_base: str = WP_API_BASE,
        rate_limit_seconds: float = DEFAULT_RATE_LIMIT_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        session: Any | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.articles_dir = self.data_dir / ARTICLES_DIRNAME
        self.manifest = Manifest.load(self.data_dir / MANIFEST_FILENAME)
        self.api_base = api_base.rstrip("/")
        self.rate_limit_seconds = rate_limit_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._last_request_at = 0.0

        if session is not None:
            self.session = session
        else:
            if requests is None:
                raise ImportError("requests is required to scrape articles")
            self.session = requests.Session()

    def scrape_all(
        self,
        limit: int | None = None,
        force: bool = False,
        show_progress: bool = True,
    ) -> dict[str, int]:
        """Fetch all article pages, saving new or changed articles to disk."""

        self.articles_dir.mkdir(parents=True, exist_ok=True)
        categories = self.fetch_terms("categories")
        tags = self.fetch_terms("tags")

        first_page, headers = self.fetch_json(
            "posts",
            {"per_page": 100, "page": 1, "orderby": "date", "order": "desc"},
        )
        if not isinstance(first_page, list):
            raise LensAPIError("Unexpected posts response: expected a JSON list")

        total_articles = _header_int(headers, "X-WP-Total", len(first_page))
        total_pages = _header_int(headers, "X-WP-TotalPages", 1)
        progress_total = min(total_articles, limit) if limit else total_articles
        progress = tqdm(
            total=progress_total,
            desc="Scraping articles",
            unit="article",
            disable=not show_progress,
        )

        stats = {
            "seen": 0,
            "downloaded": 0,
            "skipped": 0,
            "failed": 0,
            "pages": 0,
        }

        try:
            for page_number in range(1, total_pages + 1):
                if page_number == 1:
                    page_articles = first_page
                else:
                    page_articles, _ = self.fetch_json(
                        "posts",
                        {
                            "per_page": 100,
                            "page": page_number,
                            "orderby": "date",
                            "order": "desc",
                        },
                    )
                    if not isinstance(page_articles, list):
                        raise LensAPIError(
                            f"Unexpected posts response for page {page_number}"
                        )

                stats["pages"] += 1
                if not page_articles:
                    break

                for raw_article in page_articles:
                    if limit is not None and stats["seen"] >= limit:
                        break

                    stats["seen"] += 1
                    article_id = raw_article.get("id")
                    article_path = self.article_path(article_id)
                    modified = raw_article.get("modified")
                    if (
                        not force
                        and article_id is not None
                        and self.manifest.is_downloaded_current(
                            article_id, modified, article_path
                        )
                    ):
                        stats["skipped"] += 1
                        progress.update(1)
                        continue

                    try:
                        article = self.normalize_article(raw_article, categories, tags)
                        self.save_article(article)
                        self.manifest.mark_downloaded(article)
                        # Batch save manifest every 50 articles to avoid O(n²) disk writes
                        if stats["downloaded"] % 50 == 0:
                            self.manifest.save()
                        stats["downloaded"] += 1
                    except Exception:
                        stats["failed"] += 1
                        continue
                    finally:
                        progress.update(1)

                if limit is not None and stats["seen"] >= limit:
                    break
        finally:
            progress.close()
            # Always save manifest at end (batch saves every 50 during loop)
            self.manifest.save()

        return stats

    def fetch_terms(self, taxonomy: str) -> dict[int, dict[str, Any]]:
        """Fetch all categories or tags and return them keyed by id."""

        terms: dict[int, dict[str, Any]] = {}
        page_number = 1
        while True:
            payload, headers = self.fetch_json(
                taxonomy,
                {"per_page": 100, "page": page_number, "hide_empty": False},
            )
            if not isinstance(payload, list):
                raise LensAPIError(f"Unexpected {taxonomy} response")
            for item in payload:
                if "id" in item:
                    terms[int(item["id"])] = {
                        "id": item.get("id"),
                        "name": html_to_plain_text(item.get("name", "")),
                        "slug": item.get("slug"),
                    }
            total_pages = _header_int(headers, "X-WP-TotalPages", 1)
            if page_number >= total_pages or not payload:
                break
            page_number += 1
        return terms

    def fetch_json(
        self,
        endpoint: str,
        params: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Mapping[str, str]]:
        """Fetch a JSON endpoint with retries and polite rate limiting."""

        url = f"{self.api_base}/{endpoint.strip('/')}"
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            self._respect_rate_limit()
            try:
                response = self.session.get(
                    url,
                    params=dict(params or {}),
                    timeout=self.timeout_seconds,
                )
                self._last_request_at = time.monotonic()
                status_code = getattr(response, "status_code", 200)
                if status_code == 429:
                    retry_after = _header_int(
                        getattr(response, "headers", {}), "Retry-After", 0
                    )
                    wait = max(retry_after, 2**attempt)
                    time.sleep(wait)
                    continue  # retry without consuming attempt counter
                if status_code in {500, 502, 503, 504}:
                    raise LensAPIError(
                        f"Transient API status {status_code} for {url}"
                    )
                if hasattr(response, "raise_for_status"):
                    response.raise_for_status()
                return response.json(), getattr(response, "headers", {})
            except Exception as exc:  # noqa: BLE001 - retry boundary
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(2 ** (attempt - 1))

        raise LensAPIError(
            f"Failed to fetch {url} after {self.max_retries} attempts"
        ) from last_error

    def normalize_article(
        self,
        raw_article: Mapping[str, Any],
        categories: Mapping[int, Mapping[str, Any]],
        tags: Mapping[int, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Normalize a WordPress article payload for local storage."""

        article_id = raw_article.get("id")
        if article_id is None:
            raise ValueError(
                f"Article missing 'id' field: {list(raw_article.keys())[:10]}"
            )
        content_html = _rendered(raw_article, "content")
        title = html_to_plain_text(_rendered(raw_article, "title"))
        excerpt = html_to_plain_text(_rendered(raw_article, "excerpt"))
        plain_text = html_to_plain_text(content_html)

        category_ids = [
            x for x in (_safe_int(v) for v in raw_article.get("categories", [])) if x is not None
        ]
        tag_ids = [
            x for x in (_safe_int(v) for v in raw_article.get("tags", [])) if x is not None
        ]
        category_names = [
            str(categories[item]["name"])
            for item in category_ids
            if item in categories and categories[item].get("name")
        ]
        tag_names = [
            str(tags[item]["name"])
            for item in tag_ids
            if item in tags and tags[item].get("name")
        ]
        category_slugs = [
            str(categories[item]["slug"])
            for item in category_ids
            if item in categories and categories[item].get("slug")
        ]
        tag_slugs = [
            str(tags[item]["slug"])
            for item in tag_ids
            if item in tags and tags[item].get("slug")
        ]

        content_hash = sha256_text(
            "\n".join(
                [
                    str(raw_article.get("modified", "")),
                    str(title),
                    str(content_html),
                ]
            )
        )

        return {
            "id": article_id,
            "date": raw_article.get("date"),
            "date_gmt": raw_article.get("date_gmt"),
            "modified": raw_article.get("modified"),
            "modified_gmt": raw_article.get("modified_gmt"),
            "slug": raw_article.get("slug"),
            "url": raw_article.get("link"),
            "link": raw_article.get("link"),
            "title": title,
            "title_html": _rendered(raw_article, "title"),
            "excerpt": excerpt,
            "excerpt_html": _rendered(raw_article, "excerpt"),
            "content_html": content_html,
            "plain_text": plain_text,
            "categories": category_ids,
            "category_names": category_names,
            "category_slugs": category_slugs,
            "tags": tag_ids,
            "tag_names": tag_names,
            "tag_slugs": tag_slugs,
            "content_hash": content_hash,
        }

    def save_article(self, article: Mapping[str, Any]) -> Path:
        """Save one normalized article JSON file atomically."""

        path = self.article_path(article["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(article, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
        return path

    def article_path(self, article_id: int | str) -> Path:
        """Return the local JSON path for an article id."""

        return self.articles_dir / f"{article_id}.json"

    def _respect_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        sleep_for = self.rate_limit_seconds - elapsed
        if self._last_request_at and sleep_for > 0:
            time.sleep(sleep_for)


def _safe_int(v: Any) -> int | None:
    """Convert a value to int, returning None on failure."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _rendered(raw_article: Mapping[str, Any], key: str) -> str:
    value = raw_article.get(key, {})
    if isinstance(value, Mapping):
        rendered = value.get("rendered", "")
        return str(rendered or "")
    return str(value or "")


def _header_int(headers: Mapping[str, Any], key: str, default: int) -> int:
    for candidate in (key, key.lower(), key.upper()):
        if candidate in headers:
            try:
                return int(headers[candidate])
            except (TypeError, ValueError):
                return default
    return default


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the scraper CLI argument parser."""

    parser = argparse.ArgumentParser(description="Scrape The Lens article archive")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rate-limit", type=float, default=DEFAULT_RATE_LIMIT_SECONDS)
    parser.add_argument("--api-base", default=WP_API_BASE)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the scraper CLI."""

    args = build_arg_parser().parse_args(argv)
    try:
        scraper = LensScraper(
            data_dir=args.data_dir,
            api_base=args.api_base,
            rate_limit_seconds=args.rate_limit,
        )
        stats = scraper.scrape_all(limit=args.limit, force=args.force)
    except (ImportError, LensAPIError) as exc:
        print(f"Scrape failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0
