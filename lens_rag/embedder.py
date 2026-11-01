"""Embedding pipeline backed by sentence-transformers and ChromaDB."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import (
    ARTICLES_DIRNAME,
    CHROMA_DIRNAME,
    COLLECTION_NAME,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_NAME,
    MANIFEST_FILENAME,
)
from .manifest import Manifest
from .text import join_names, parse_date_to_int, safe_date_int, sha256_text, split_text_into_chunks


class EmbeddingModel(Protocol):
    """Protocol for model objects that can embed a list of strings."""

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


class SentenceTransformerEmbeddingModel:
    """Sentence-transformers implementation for article and query embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        query_prefix: str | None = None,
        model_path: str | None = None,
    ) -> None:
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency path
            raise ImportError(
                "sentence-transformers is required for embedding. "
                "Install requirements.txt first."
            ) from exc
        load_path = model_path if model_path is not None else os.getenv("LENS_RAG_EMBEDDING_MODEL_PATH", "").strip()
        self.model = SentenceTransformer(load_path or model_name)
        self.query_prefix = (
            query_prefix
            if query_prefix is not None
            else os.getenv("LENS_RAG_QUERY_PREFIX", _default_query_prefix(model_name))
        )

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        """Return normalized sentence-transformer embeddings."""

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [
            vector.tolist() if hasattr(vector, "tolist") else list(vector)
            for vector in vectors
        ]

    def encode_query(self, query: str) -> list[float]:
        """Encode a retrieval query with the model's recommended instruction."""

        return self.encode_texts([f"{self.query_prefix}{query}"])[0]


def _default_query_prefix(model_name: str) -> str:
    lowered = model_name.lower()
    if "bge-" in lowered or "snowflake-arctic-embed" in lowered:
        return "Represent this sentence for searching relevant passages: "
    if "qwen3-embedding" in lowered:
        return (
            "Instruct: Retrieve relevant passages from a local news archive that answer "
            "the reporter's query.\nQuery: "
        )
    return ""


@dataclass(frozen=True)
class ArticleChunk:
    """A single article chunk ready for insertion into ChromaDB."""

    chunk_id: str
    document: str
    metadata: dict[str, str | int | float | bool]


class ArticleEmbedder:
    """Embed downloaded Lens articles into a persistent ChromaDB collection."""

    def __init__(
        self,
        data_dir: Path = DEFAULT_DATA_DIR,
        model_name: str = DEFAULT_MODEL_NAME,
        model: EmbeddingModel | None = None,
        collection: Any | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.articles_dir = self.data_dir / ARTICLES_DIRNAME
        self.db_dir = self.data_dir / CHROMA_DIRNAME
        self.manifest = Manifest.load(self.data_dir / MANIFEST_FILENAME)
        self.model_name = model_name
        self.model = model
        self.collection = collection
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.batch_size = batch_size

    def embed_all(
        self,
        limit: int | None = None,
        force: bool = False,
    ) -> dict[str, int | float]:
        """Embed downloaded article JSON files into ChromaDB."""

        import logging

        logger = logging.getLogger(__name__)
        collection = self._collection()
        model = self._model()

        stats: dict[str, int | float] = {
            "articles_seen": 0,
            "articles_embedded": 0,
            "articles_skipped": 0,
            "articles_failed": 0,
            "chunks_added": 0,
            "db_size": 0,
            "avg_chunks_per_article": 0.0,
        }

        for article_path in self._article_files():
            if limit is not None and stats["articles_seen"] >= limit:
                break
            stats["articles_seen"] = int(stats["articles_seen"]) + 1

            try:
                article = load_article(article_path)
                article_id = article["id"]
                content_hash = article.get("content_hash") or article_content_hash(
                    article
                )
                if (
                    not force
                    and not self.manifest.needs_embedding(
                        article_id,
                        content_hash,
                        embedding_model=self.model_name,
                    )
                ):
                    stats["articles_skipped"] = int(stats["articles_skipped"]) + 1
                    continue

                chunks = article_to_chunks(
                    article,
                    chunk_size=self.chunk_size,
                    overlap=self.overlap,
                )
                if not chunks:
                    stats["articles_skipped"] = int(stats["articles_skipped"]) + 1
                    continue

                self.manifest.mark_embedding_started(article_id)
                self.manifest.save()  # Sentinel save — required for crash recovery
                delete_article_chunks(collection, article_id)
                for batch in batched(chunks, self.batch_size):
                    documents = [chunk.document for chunk in batch]
                    embeddings = model.encode_texts(documents)
                    collection.add(
                        ids=[chunk.chunk_id for chunk in batch],
                        documents=documents,
                        metadatas=[chunk.metadata for chunk in batch],
                        embeddings=embeddings,
                    )

                self.manifest.mark_embedded(
                    article_id=article_id,
                    content_hash=content_hash,
                    chunk_count=len(chunks),
                    embedding_model=self.model_name,
                )
                stats["articles_embedded"] = int(stats["articles_embedded"]) + 1
                stats["chunks_added"] = int(stats["chunks_added"]) + len(chunks)
                # Batch success saves every 50 articles to avoid O(n²) disk writes
                if stats["articles_embedded"] % 50 == 0:
                    try:
                        self.manifest.save()
                    except Exception as save_exc:
                        logger.error(
                            "Failed to save manifest after article %s: %s",
                            article_id,
                            save_exc,
                        )
                        # Don't fail the article — embedding succeeded, save can retry
            except Exception as exc:
                logger.warning(
                    "Failed to embed article %s (%s): %s",
                    article_path.stem,
                    article_path,
                    exc,
                )
                stats["articles_failed"] = int(stats["articles_failed"]) + 1
                continue

        # Final save to persist last batch of success marks
        if int(stats["articles_embedded"]) > 0:
            self.manifest.save()

        db_size = safe_collection_count(collection)
        stats["db_size"] = db_size
        embedded = int(stats["articles_embedded"])
        stats["avg_chunks_per_article"] = (
            round(int(stats["chunks_added"]) / embedded, 2) if embedded else 0.0
        )
        return stats

    def _model(self) -> EmbeddingModel:
        if self.model is None:
            self.model = SentenceTransformerEmbeddingModel(self.model_name)
        return self.model

    def _collection(self) -> Any:
        if self.collection is None:
            self.collection = get_chroma_collection(
                self.db_dir,
                collection_name=COLLECTION_NAME,
            )
        return self.collection

    def _article_files(self) -> list[Path]:
        if not self.articles_dir.exists():
            return []
        return sorted(
            self.articles_dir.glob("*.json"),
            key=lambda path: (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem),
        )


def get_chroma_collection(
    db_dir: Path, collection_name: str = COLLECTION_NAME, *, stage_snapshot: bool = True
) -> Any:
    """Open or create the persistent ChromaDB collection."""

    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - dependency path
        raise ImportError(
            "chromadb is required for vector storage. Install requirements.txt first."
        ) from exc

    from .serverless_storage import writable_chroma_directory

    if stage_snapshot:
        db_dir = writable_chroma_directory(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_dir))
    # Use get_collection first to avoid migration issues with
    # get_or_create_collection when collection config lacks _type
    try:
        return client.get_collection(name=collection_name)
    except Exception:
        return client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )


def load_article(path: Path) -> dict[str, Any]:
    """Load one normalized article JSON file."""

    with path.open("r", encoding="utf-8") as handle:
        article = json.load(handle)
    if not isinstance(article, dict):
        raise ValueError(f"Article file is not a JSON object: {path}")
    return article


def article_content_hash(article: dict[str, Any]) -> str:
    """Compute a fallback content hash for an article JSON object."""

    return sha256_text(
        "\n".join(
            [
                str(article.get("modified", "")),
                str(article.get("title", "")),
                str(article.get("content_html", "")),
                str(article.get("plain_text", "")),
            ]
        )
    )


def article_to_chunks(
    article: dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ArticleChunk]:
    """Convert one article into Chroma-ready overlapping chunks."""

    article_id = str(article["id"])
    content_hash = article.get("content_hash") or article_content_hash(article)
    text = (
        str(article.get("plain_text") or "").strip()
        or str(article.get("excerpt") or "").strip()
        or str(article.get("title") or "").strip()
    )
    if not text:
        return []

    text_chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
    metadata_base = article_metadata(article)
    chunks: list[ArticleChunk] = []
    for chunk_index, chunk_text in enumerate(text_chunks):
        document = article_chunk_document(article, chunk_text)
        metadata = dict(metadata_base)
        metadata.update(
            {
                "chunk_index": chunk_index,
                "chunk_count": len(text_chunks),
                "content_hash": str(content_hash),
            }
        )
        chunks.append(
            ArticleChunk(
                chunk_id=f"{article_id}:{chunk_index}:{str(content_hash)[:12]}",
                document=document,
                metadata=metadata,
            )
        )
    return chunks


def article_metadata(article: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Build scalar Chroma metadata for one article."""

    return {
        "article_id": str(article.get("id", "")),
        "title": str(article.get("title", "")),
        "date": str(article.get("date", "")),
        "date_int": safe_date_int(str(article.get("date", "")))
        if article.get("date")
        else 0,
        "categories": join_names(article.get("category_names", [])),
        "category_slugs": join_names(article.get("category_slugs", [])),
        "tags": join_names(article.get("tag_names", [])),
        "tag_slugs": join_names(article.get("tag_slugs", [])),
        "slug": str(article.get("slug", "")),
        "url": str(article.get("url") or article.get("link") or ""),
    }


def article_chunk_document(article: dict[str, Any], chunk_text: str) -> str:
    """Build the searchable document text for one article chunk."""

    categories = join_names(article.get("category_names", [])) or "Uncategorized"
    tags = join_names(article.get("tag_names", []))
    parts = [
        f"Title: {article.get('title', '')}",
        f"Categories: {categories}",
        f"Tags: {tags}" if tags else "",
        "",
        "Content:",
        chunk_text.strip(),
    ]
    return "\n".join(parts).strip()


def delete_article_chunks(collection: Any, article_id: int | str) -> None:
    """Delete existing Chroma chunks for an article if they exist."""

    try:
        collection.delete(where={"article_id": str(article_id)})
    except Exception as exc:
        msg = str(exc).lower()
        if "no documents" in msg or "not found" in msg or "empty" in msg:
            return  # nothing to delete — expected
        raise


def safe_collection_count(collection: Any) -> int:
    """Return a collection count, treating unsupported count calls as zero."""

    try:
        return int(collection.count())
    except Exception:
        return 0


def batched(items: list[ArticleChunk], batch_size: int) -> list[list[ArticleChunk]]:
    """Split article chunks into insertion batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the embedding CLI argument parser."""

    parser = argparse.ArgumentParser(description="Embed downloaded Lens articles")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the embedding CLI."""

    args = build_arg_parser().parse_args(argv)
    try:
        embedder = ArticleEmbedder(
            data_dir=args.data_dir,
            model_name=args.model,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
        )
        stats = embedder.embed_all(limit=args.limit, force=args.force)
    except (ImportError, ValueError) as exc:
        print(f"Embedding failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0
