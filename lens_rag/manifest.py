"""Manifest persistence for downloaded and embedded articles."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


class Manifest:
    """Manage the manifest that tracks scrape and embedding state."""

    schema_version = 1

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = self._load()

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        """Load a manifest from disk, creating an empty one when absent."""

        return cls(path)

    @property
    def articles(self) -> dict[str, dict[str, Any]]:
        """Return manifest article entries keyed by article id as a string."""

        articles = self.data.setdefault("articles", {})
        if not isinstance(articles, dict):
            raise ValueError(f"Invalid manifest articles section in {self.path}")
        return articles

    def get(self, article_id: int | str) -> dict[str, Any] | None:
        """Return the manifest entry for an article id, if present."""

        entry = self.articles.get(str(article_id))
        return entry if isinstance(entry, dict) else None

    def is_downloaded_current(
        self,
        article_id: int | str,
        modified: str | None,
        article_path: Path,
    ) -> bool:
        """Return true when an article file exists and matches manifest state."""

        entry = self.get(article_id)
        if entry is None or not article_path.exists():
            return False
        if modified and entry.get("modified") and entry["modified"] != modified:
            return False
        return bool(entry.get("content_hash"))

    def mark_downloaded(self, article: dict[str, Any]) -> None:
        """Record that an article has been downloaded and parsed."""

        article_id = str(article["id"])
        existing = self.articles.get(article_id, {})
        was_embedded_hash = existing.get("embedded_hash")
        content_hash = article.get("content_hash")

        entry: dict[str, Any] = {
            "id": article["id"],
            "slug": article.get("slug"),
            "date": article.get("date"),
            "date_gmt": article.get("date_gmt"),
            "modified": article.get("modified"),
            "modified_gmt": article.get("modified_gmt"),
            "title": article.get("title"),
            "url": article.get("url"),
            "categories": article.get("categories", []),
            "category_names": article.get("category_names", []),
            "tags": article.get("tags", []),
            "tag_names": article.get("tag_names", []),
            "download_date": utc_now_iso(),
            "content_hash": content_hash,
            "embedded": existing.get("embedded", False)
            and was_embedded_hash == content_hash,
            "embedded_at": existing.get("embedded_at")
            if was_embedded_hash == content_hash
            else None,
            "embedded_hash": was_embedded_hash
            if was_embedded_hash == content_hash
            else None,
            "chunk_count": existing.get("chunk_count", 0)
            if was_embedded_hash == content_hash
            else 0,
            "embedding_model": existing.get("embedding_model")
            if was_embedded_hash == content_hash
            else None,
        }
        self.articles[article_id] = entry
        self.data["updated_at"] = utc_now_iso()

    def needs_embedding(
        self,
        article_id: int | str,
        content_hash: str,
        embedding_model: str | None = None,
    ) -> bool:
        """Return true when an article needs to be embedded or re-embedded."""

        entry = self.get(article_id)
        if entry is None:
            return True
        return not (
            entry.get("embedded") is True
            and entry.get("embedded_hash") == content_hash
            and int(entry.get("chunk_count") or 0) > 0
            and (
                embedding_model is None
                or entry.get("embedding_model") == embedding_model
            )
        )

    def mark_embedded(
        self,
        article_id: int | str,
        content_hash: str,
        chunk_count: int,
        embedding_model: str,
    ) -> None:
        """Record successful embedding for an article."""

        entry = self.articles.setdefault(str(article_id), {"id": article_id})
        entry.update(
            {
                "embedded": True,
                "embedded_at": utc_now_iso(),
                "embedded_hash": content_hash,
                "chunk_count": chunk_count,
                "embedding_model": embedding_model,
            }
        )
        self.data["updated_at"] = utc_now_iso()

    def mark_embedding_started(self, article_id: int | str) -> None:
        """Write a sentinel that forces re-embed on next run if interrupted.

        Call this BEFORE deleting old chunks so a crash between delete and add
        leaves the manifest saying 'not embedded' rather than 'embedded with
        zero chunks in ChromaDB'.
        """
        entry = self.articles.setdefault(str(article_id), {"id": article_id})
        entry["embedded"] = False
        entry["embedded_hash"] = None
        entry["chunk_count"] = 0
        self.data["updated_at"] = utc_now_iso()

    def downloaded_count(self) -> int:
        """Return the number of articles tracked as downloaded."""

        return sum(1 for entry in self.articles.values() if entry.get("content_hash"))

    def embedded_count(self) -> int:
        """Return the number of articles tracked as embedded."""

        return sum(1 for entry in self.articles.values() if entry.get("embedded"))

    def save(self) -> None:
        """Persist the manifest atomically with file locking.

        Uses fcntl.flock with a 30-second timeout; raises TimeoutError on contention.
        """

        import fcntl
        import time

        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("w") as lock_handle:
            deadline = time.monotonic() + 30.0
            while True:
                try:
                    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            f"Could not acquire manifest lock after 30s: {self.path}"
                        )
                    time.sleep(0.1)
            try:
                tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
                with tmp_path.open("w", encoding="utf-8") as handle:
                    json.dump(self.data, handle, ensure_ascii=False, indent=2, sort_keys=True)
                    handle.write("\n")
                os.replace(tmp_path, self.path)
            finally:
                fcntl.flock(lock_handle, fcntl.LOCK_UN)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "schema_version": self.schema_version,
                "created_at": utc_now_iso(),
                "updated_at": utc_now_iso(),
                "articles": {},
            }

        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid manifest JSON object in {self.path}")
        data.setdefault("schema_version", self.schema_version)
        data.setdefault("articles", {})
        return data
