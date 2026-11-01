"""Persistent SQLite FTS5 retrieval for local hybrid archive search."""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import CHROMA_DIRNAME, COLLECTION_NAME, DEFAULT_DATA_DIR
from .embedder import get_chroma_collection, safe_collection_count
from .text import document_labeled_field

FTS_FILENAME = "search_fts.sqlite3"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)


@dataclass(frozen=True)
class LexicalMatch:
    chunk_id: str
    document: str
    metadata: dict[str, Any]


class SQLiteFTSIndex:
    """Build and query a disk-backed full-corpus BM25 index."""

    def __init__(self, path: Path, revision: str | None = None) -> None:
        self.path = Path(path)
        self.revision = revision
        self._lock = threading.Lock()

    def ensure(self, collection: Any) -> None:
        """Build the index when missing or stale relative to Chroma."""

        expected_count = safe_collection_count(collection)
        if self._is_current(expected_count):
            return
        with self._lock:
            if self._is_current(expected_count):
                return
            self.rebuild(collection)

    def rebuild(self, collection: Any, batch_size: int = 1000) -> int:
        """Atomically rebuild FTS from the documents already stored in Chroma."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        temporary.unlink(missing_ok=True)
        connection = sqlite3.connect(temporary)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                """
                CREATE VIRTUAL TABLE chunks USING fts5(
                    chunk_id UNINDEXED,
                    article_id UNINDEXED,
                    date UNINDEXED,
                    date_int UNINDEXED,
                    chunk_index UNINDEXED,
                    url UNINDEXED,
                    title,
                    categories,
                    tags,
                    document,
                    tokenize='porter unicode61 remove_diacritics 2'
                )
                """
            )
            connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

            total = safe_collection_count(collection)
            inserted = 0
            for offset in range(0, total, batch_size):
                payload = collection.get(
                    limit=min(batch_size, total - offset),
                    offset=offset,
                    include=["documents", "metadatas"],
                )
                rows = list(_fts_rows(payload))
                if not rows:
                    continue
                connection.executemany(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                inserted += len(rows)
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('collection_count', ?)",
                (str(total),),
            )
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('revision', ?)",
                (self.revision or "",),
            )
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary, self.path)
        return inserted

    def search(
        self,
        collection: Any,
        terms: Iterable[str],
        limit: int = 100,
        category: str | None = None,
        date_from: int | None = None,
        date_to: int | None = None,
    ) -> list[LexicalMatch]:
        """Return BM25-ranked chunks, honoring the same filters as dense search."""

        if limit <= 0:
            return []
        expression = _fts_expression(terms)
        if not expression:
            return []
        self.ensure(collection)

        clauses = ["chunks MATCH ?"]
        parameters: list[Any] = [expression]
        if category:
            clauses.append("instr(lower(categories), lower(?)) > 0")
            parameters.append(category)
        if date_from is not None:
            clauses.append("CAST(date_int AS INTEGER) >= ?")
            parameters.append(date_from)
        if date_to is not None:
            clauses.append("CAST(date_int AS INTEGER) <= ?")
            parameters.append(date_to)
        parameters.append(limit)

        sql = f"""
            SELECT chunk_id, article_id, date, date_int, chunk_index, url,
                   title, categories, tags, document
            FROM chunks
            WHERE {' AND '.join(clauses)}
            ORDER BY bm25(chunks, 0, 0, 0, 0, 0, 0, 5.0, 2.0, 3.0, 1.0)
            LIMIT ?
        """
        connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        try:
            rows = connection.execute(sql, parameters).fetchall()
        finally:
            connection.close()

        return [
            LexicalMatch(
                chunk_id=str(row[0]),
                document=str(row[9] or ""),
                metadata={
                    "article_id": str(row[1] or ""),
                    "date": str(row[2] or ""),
                    "date_int": int(row[3] or 0),
                    "chunk_index": int(row[4] or 0),
                    "url": str(row[5] or ""),
                    "title": str(row[6] or ""),
                    "categories": str(row[7] or ""),
                    "tags": str(row[8] or ""),
                },
            )
            for row in rows
        ]

    def _indexed_count(self) -> int | None:
        if not self.path.exists():
            return None
        try:
            connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
            try:
                row = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'collection_count'"
                ).fetchone()
                return int(row[0]) if row else None
            finally:
                connection.close()
        except (OSError, sqlite3.Error, TypeError, ValueError):
            return None

    def _is_current(self, expected_count: int) -> bool:
        if expected_count <= 0 or self._indexed_count() != expected_count:
            return False
        if self.revision is None:
            return True
        try:
            connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
            try:
                row = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'revision'"
                ).fetchone()
                return bool(row and row[0] == self.revision)
            finally:
                connection.close()
        except sqlite3.Error:
            return False


def _fts_rows(payload: dict[str, Any]) -> Iterable[tuple[Any, ...]]:
    ids = payload.get("ids") or []
    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []
    for chunk_id, document, raw_metadata in zip(ids, documents, metadatas):
        metadata = raw_metadata or {}
        document_text = str(document or "")
        title = str(metadata.get("title", "")) or document_labeled_field(
            document_text,
            "Title",
        )
        yield (
            str(chunk_id),
            str(metadata.get("article_id", "")),
            str(metadata.get("date", "")),
            int(metadata.get("date_int") or 0),
            int(metadata.get("chunk_index") or 0),
            str(metadata.get("url", "")),
            title,
            str(metadata.get("categories", "")),
            str(metadata.get("tags", "")),
            document_text,
        )


def _fts_expression(terms: Iterable[str]) -> str:
    normalized: list[str] = []
    seen: set[str] = set()
    for term in terms:
        tokens = TOKEN_RE.findall(str(term).lower())
        if not tokens:
            continue
        phrase = " ".join(tokens)
        if len(tokens) > 1:
            _add_fts_term(normalized, seen, phrase)
        for token in tokens:
            _add_fts_term(normalized, seen, token)
        if len(normalized) >= 40:
            break
    return " OR ".join(normalized)


def _add_fts_term(parts: list[str], seen: set[str], term: str) -> None:
    if term in seen or len(parts) >= 40:
        return
    seen.add(term)
    parts.append('"' + term.replace('"', '""') + '"')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the local Lens FTS5 index")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)
    collection = get_chroma_collection(args.data_dir / CHROMA_DIRNAME, COLLECTION_NAME)
    count = SQLiteFTSIndex(
        args.data_dir / FTS_FILENAME,
        revision=_file_revision(args.data_dir / "manifest.json"),
    ).rebuild(collection)
    print(f"Indexed {count} chunks in {args.data_dir / FTS_FILENAME}")
    return 0


def _file_revision(path: Path) -> str | None:
    try:
        stat = path.stat()
        return f"{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
