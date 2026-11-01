"""Generic integration boundary around the existing retrieval engine."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from lens_rag.embedder import (
    ArticleEmbedder,
    SentenceTransformerEmbeddingModel,
    _default_query_prefix,
    get_chroma_collection,
)
from lens_rag.hybrid_search import FTS_FILENAME, SQLiteFTSIndex
from lens_rag.manifest import Manifest
from lens_rag.query_engine import QueryEngine, manifest_revision
from lens_rag.text import parse_date_to_int, sha256_text


@dataclass(frozen=True)
class Document:
    """A stable ID and plain text, with optional retrieval and citation metadata."""

    id: str
    text: str
    title: str = ""
    url: str = ""
    date: str = ""
    categories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.id
        ):
            raise ValueError(
                "document id must be 1-128 letters, digits, dots, underscores or hyphens, starting with a letter or digit"
            )
        for name in ("text", "title", "url", "date"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"document {name} must be a string")
        if not self.text.strip():
            raise ValueError("document text must not be empty")
        if self.date:
            parse_date_to_int(self.date)
        for name in ("categories", "tags"):
            values = getattr(self, name)
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(f"document {name} must be a list of nonempty strings")

    def _article(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "plain_text": self.text,
            "title": self.title or self.id,
            "url": self.url,
            "date": self.date,
            "category_names": self.categories,
            "tag_names": self.tags,
            # Metadata changes also invalidate vectors and the lexical index.
            "content_hash": sha256_text(json.dumps(asdict(self), sort_keys=True)),
        }


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(".rag-data")
    embedding_model: str = "all-MiniLM-L6-v2"
    model_path: str = ""
    chunk_size: int = 200
    overlap: int = 50
    hybrid: bool = True
    reranker_model: str = ""

    def __post_init__(self) -> None:
        if not self.embedding_model.strip():
            raise ValueError("embedding_model must not be empty")
        if self.chunk_size <= 0 or not 0 <= self.overlap < self.chunk_size:
            raise ValueError("require chunk_size > 0 and 0 <= overlap < chunk_size")

    @classmethod
    def from_env(cls) -> "Settings":
        """Read only RAG_* settings; never read repository or home-directory files."""

        hybrid = os.getenv("RAG_HYBRID", "true").lower()
        if hybrid not in {"true", "false", "1", "0"}:
            raise ValueError("RAG_HYBRID must be true, false, 1 or 0")
        return cls(
            data_dir=Path(os.getenv("RAG_DATA_DIR", ".rag-data")),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            model_path=os.getenv("RAG_MODEL_PATH", ""),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "200")),
            overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "50")),
            hybrid=hybrid in {"true", "1"},
            reranker_model=os.getenv("RAG_RERANKER_MODEL", ""),
        )


class RAG:
    """Ingest, retrieve and optionally generate answers in an isolated store.

    One writer per directory. Stop serving while updating a store, then recreate
    readers. Model and collection injection let applications use their own backend.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model: Any = None,
        collection: Any = None,
        generator: Any = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.data_dir = Path(self.settings.data_dir)
        self.model = model
        self.collection = collection
        self.generator = generator
        self._backend_lock = threading.Lock()
        self._engine: QueryEngine | None = None
        self._check_store()

    def _store_settings(self) -> dict[str, Any]:
        return {
            key: getattr(self.settings, key)
            for key in ("embedding_model", "chunk_size", "overlap")
        }

    def _check_store(self) -> Manifest:
        manifest = Manifest.load(self.data_dir / "manifest.json")
        stored = manifest.data.get("rag_settings")
        if stored is None and manifest.articles:
            raise ValueError(
                "This is an existing archive, not a PlugRAG store. Choose a new RAG_DATA_DIR."
            )
        if stored is not None and stored != self._store_settings():
            raise ValueError(
                "Embedding model or chunk settings differ from this store. Use its original settings or ingest into a new RAG_DATA_DIR."
            )
        return manifest

    def _backend(self) -> tuple[Any, Any]:
        # Concurrent first requests must not load multiple copies of model weights.
        with self._backend_lock:
            if self.collection is None:
                self.collection = get_chroma_collection(
                    self.data_dir / "chroma_db",
                    "documents",
                    stage_snapshot=False,
                )
            if self.model is None:
                prefix = _default_query_prefix(self.settings.embedding_model)
                if "qwen3-embedding" in self.settings.embedding_model.lower():
                    prefix = "Instruct: Retrieve relevant passages that answer the query.\nQuery: "
                self.model = SentenceTransformerEmbeddingModel(
                    self.settings.embedding_model,
                    query_prefix=prefix,
                    model_path=self.settings.model_path,
                )
            return self.model, self.collection

    def ingest(self, documents: Iterable[Document]) -> dict[str, Any]:
        """Upsert documents by ID, skip unchanged content, and refresh hybrid search.

        A failed embedding raises; retry the same ingestion to resume. Each source
        file and manifest write is atomic; the entire batch is not a transaction.
        """

        documents = list(documents)
        if not documents:
            raise ValueError("provide at least one document")
        if any(not isinstance(document, Document) for document in documents):
            raise ValueError("ingest expects Document objects")
        ids = [document.id for document in documents]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate document IDs in ingestion batch")
        manifest = self._check_store()
        model, collection = self._backend()
        articles_dir = self.data_dir / "articles"
        articles_dir.mkdir(parents=True, exist_ok=True)
        manifest.data["rag_settings"] = self._store_settings()
        for document in documents:
            article = document._article()
            # Replace complete JSON files, so a failed write cannot truncate a source.
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=articles_dir, delete=False
            ) as handle:
                temporary = Path(handle.name)
                try:
                    json.dump(article, handle, ensure_ascii=False)
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            try:
                os.replace(temporary, articles_dir / f"{document.id}.json")
            finally:
                temporary.unlink(missing_ok=True)
            manifest.mark_downloaded(article)
        manifest.save()
        stats = ArticleEmbedder(
            data_dir=self.data_dir,
            model_name=self.settings.embedding_model,
            model=model,
            collection=collection,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.overlap,
        ).embed_all()
        self._engine = None
        if stats["articles_failed"]:
            raise RuntimeError(
                f"{stats['articles_failed']} document(s) failed embedding; retry ingestion to resume"
            )
        if self.settings.hybrid:
            SQLiteFTSIndex(
                self.data_dir / FTS_FILENAME, revision=manifest_revision(self.data_dir)
            ).rebuild(collection)
        return stats

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        category: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve ranked documents with excerpts and citation metadata."""

        if not isinstance(query, str) or not query.strip() or len(query) > 10000:
            raise ValueError("query must contain 1-10000 characters")
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        for date in (date_from, date_to):
            if date:
                parse_date_to_int(date)
        if (
            date_from
            and date_to
            and parse_date_to_int(date_from) > parse_date_to_int(date_to)
        ):
            raise ValueError("date_from must be on or before date_to")
        self._check_store()
        if self._engine is None:
            model, collection = self._backend()
            self._engine = QueryEngine(
                data_dir=self.data_dir,
                model_name=self.settings.embedding_model,
                model=model,
                collection=collection,
                hybrid_enabled=self.settings.hybrid,
                reranker_model=self.settings.reranker_model,
                fts_index=SQLiteFTSIndex(
                    self.data_dir / FTS_FILENAME,
                    revision=manifest_revision(self.data_dir),
                ),
            )
        results = self._engine.search_results(
            query,
            page_size=top_k,
            category=category,
            date_from=date_from,
            date_to=date_to,
            nl_rewrite=False,
        ).results
        return [
            {
                "id": result.article_id,
                **{
                    key: value
                    for key, value in result.to_dict().items()
                    if key != "article_id"
                },
            }
            for result in results
        ]

    def answer(self, query: str, **filters: Any) -> dict[str, Any]:
        """Return an answer with numbered source context. Generation is opt-in."""

        if self.generator is None:
            raise RuntimeError("Configure a generator before requesting answers")
        results = self.search(query, **filters)
        sources = [
            {"index": index, **result} for index, result in enumerate(results, 1)
        ]
        if not results:
            return {
                "query": query,
                "answer": "No matching documents found.",
                "sources": [],
            }
        return {
            "query": query,
            "answer": self.generator.generate(query, sources),
            "sources": sources,
        }
