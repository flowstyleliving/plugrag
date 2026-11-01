"""Query interface for the Lens ChromaDB archive."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import (
    CHROMA_DIRNAME,
    COLLECTION_NAME,
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_NAME,
)
from .embedder import (
    EmbeddingModel,
    SentenceTransformerEmbeddingModel,
    get_chroma_collection,
    load_article,
    safe_collection_count,
)
from .hybrid_search import FTS_FILENAME, LexicalMatch, SQLiteFTSIndex
from .knowledge_graph import GRAPH_FILENAME, KnowledgeGraphIndex
from .nl_query import (
    KeywordBooster,
    QueryRewriter,
    RelevanceScore,
    RelevanceScorer,
    RewriteResult,
    SearchResults,
    keywords_from_query,
)
from .reranker import LocalCrossEncoderReranker
from .text import document_labeled_field, join_names, parse_date_to_int
from .topics import TOPIC_LEXICON_FILENAME, TopicLexicon

logger = logging.getLogger(__name__)


@dataclass
class ChunkMatch:
    """One matching chunk returned from ChromaDB."""

    score: float
    excerpt: str
    chunk_index: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize this match to a JSON-compatible dictionary."""

        return {
            "score": self.score,
            "excerpt": self.excerpt,
            "chunk_index": self.chunk_index,
        }


@dataclass
class ArticleResult:
    """Grouped search result for one article."""

    article_id: str
    title: str
    date: str
    url: str
    categories: str
    tags: str
    score: float
    score_info: RelevanceScore | None = None
    matches: list[ChunkMatch] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result to a JSON-compatible dictionary."""

        return {
            "article_id": self.article_id,
            "title": self.title,
            "date": self.date,
            "url": self.url,
            "categories": self.categories,
            "tags": self.tags,
            "score": self.score,
            "score_info": self.score_info.to_dict() if self.score_info else None,
            "matches": [match.to_dict() for match in self.matches],
        }


class QueryEngine:
    """Search Lens article chunks and group matching chunks by article."""

    def __init__(
        self,
        data_dir: Path = DEFAULT_DATA_DIR,
        model_name: str | None = None,
        model: EmbeddingModel | None = None,
        collection: Any | None = None,
        rewriter: QueryRewriter | None = None,
        fts_index: SQLiteFTSIndex | None = None,
        hybrid_enabled: bool | None = None,
        reranker: Any | None = None,
        topic_lexicon: TopicLexicon | None = None,
        knowledge_graph_index: KnowledgeGraphIndex | None = None,
        reranker_model: str | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.db_dir = self.data_dir / CHROMA_DIRNAME
        self.model_name = (
            model_name
            or os.getenv("LENS_RAG_EMBEDDING_MODEL", "").strip()
            or manifest_embedding_model(self.data_dir)
            or DEFAULT_MODEL_NAME
        )
        self.model = model
        self.collection = collection
        self.rewriter = rewriter
        self.fts_index = fts_index
        self.hybrid_enabled = (
            hybrid_enabled
            if hybrid_enabled is not None
            else os.getenv("LENS_RAG_ENABLE_HYBRID", "true").strip().lower()
            not in {"0", "false", "no", "off"}
        )
        self.reranker = reranker
        self.reranker_model = (
            reranker_model if reranker_model is not None
            else os.getenv("LENS_RAG_RERANKER_MODEL", "").strip()
        )
        self.topic_lexicon = topic_lexicon
        self.knowledge_graph_index = knowledge_graph_index

    def search(
        self,
        query: str,
        top_k: int = 10,
        category: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        expansion_terms: list[str] | None = None,
    ) -> list[ArticleResult]:
        """Search the archive and return grouped article results."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        return self.search_results(
            query,
            page=1,
            page_size=top_k,
            category=category,
            date_from=date_from,
            date_to=date_to,
            expansion_terms=expansion_terms,
            nl_rewrite=True,
        ).results

    def search_results(
        self,
        query: str,
        page: int = 1,
        page_size: int = 10,
        category: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        expansion_terms: list[str] | None = None,
        nl_rewrite: bool = True,
        available_categories: list[str] | tuple[str, ...] | None = None,
    ) -> SearchResults:
        """Search the archive and return paginated results plus metadata."""

        query = query.strip()
        if not query:
            return SearchResults(
                query="",
                effective_query="",
                results=[],
                total=0,
                page=page,
                page_size=page_size,
                has_more=False,
            )
        if page <= 0:
            raise ValueError("page must be positive")
        if page_size <= 0:
            raise ValueError("page_size must be positive")

        rewrite: RewriteResult | None = None
        effective_query = query
        effective_category = category
        effective_date_from = date_from
        effective_date_to = date_to
        terms = list(expansion_terms or [])
        if nl_rewrite:
            if available_categories is None:
                available_categories = manifest_categories(self.data_dir)
            rewrite = self._rewriter().rewrite(query, available_categories)
            effective_query = rewrite.rewritten_query or query
            effective_category = category or rewrite.category
            effective_date_from = date_from or rewrite.date_from
            effective_date_to = date_to or rewrite.date_to
            terms.extend(rewrite.expansion_terms)
        if not terms:
            terms = keywords_from_query(query)
        terms.extend(keywords_from_query(effective_query))
        lexical_terms = list(terms)
        topic_expansion = self._topic_lexicon().expand(effective_query, lexical_terms)
        lexical_terms.extend(topic_expansion.terms)

        collection_size = safe_collection_count(self._collection())
        candidate_count = candidate_chunk_count(
            page=page,
            page_size=page_size,
            collection_size=collection_size,
            category=effective_category,
        )
        grouped = self._search_grouped(
            effective_query,
            candidate_count=candidate_count,
            category=effective_category,
            date_from=effective_date_from,
            date_to=effective_date_to,
            expansion_terms=terms,
            lexical_expansion_terms=lexical_terms,
        )
        start = (page - 1) * page_size
        end = start + page_size
        page_results = grouped[start:end]
        has_more = len(grouped) > end or (
            len(grouped) >= end and candidate_count < collection_size
        )
        return SearchResults(
            query=query,
            effective_query=effective_query,
            results=page_results,
            total=len(grouped),
            page=page,
            page_size=page_size,
            has_more=has_more,
            category=effective_category,
            date_from=effective_date_from,
            date_to=effective_date_to,
            rewrite=rewrite,
            keywords=keywords_from_query(" ".join(terms)),
        )

    def supports_query_rewriting(self) -> bool:
        """Return true when local or explicitly enabled remote rewriting is available."""

        return self._rewriter().supports_rewriting()

    def graph_search_results(
        self,
        query: str,
        top_k: int = 8,
    ) -> tuple[list[ArticleResult], dict[str, Any]]:
        """Return article evidence reached through persistent entity relations."""

        graph_result = self._knowledge_graph().search(query, limit=top_k)
        if not graph_result.hits:
            return [], graph_result.diagnostics()
        maximum = max(hit.score for hit in graph_result.hits) or 1.0
        results: list[ArticleResult] = []
        for hit in graph_result.hits:
            path = self.data_dir / "articles" / f"{hit.article_id}.json"
            try:
                article = load_article(path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            content = str(article.get("plain_text") or "")
            entities = [*hit.matched_entities, *hit.related_entities]
            excerpts = _graph_excerpts(content, query, entities)
            normalized_score = max(0.0, min(1.0, hit.score / maximum))
            results.append(
                ArticleResult(
                    article_id=hit.article_id,
                    title=str(article.get("title") or ""),
                    date=str(article.get("date") or ""),
                    url=str(article.get("url") or article.get("link") or ""),
                    categories=join_names(article.get("category_names") or []),
                    tags=join_names(article.get("tag_names") or []),
                    score=normalized_score,
                    matches=[
                        ChunkMatch(
                            score=max(0.0, normalized_score - (index * 0.02)),
                            excerpt=excerpt,
                            chunk_index=-(index + 1),
                        )
                        for index, excerpt in enumerate(excerpts)
                    ],
                )
            )
        return results, graph_result.diagnostics()

    def _search_grouped(
        self,
        query: str,
        candidate_count: int,
        category: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        expansion_terms: list[str] | None = None,
        lexical_expansion_terms: list[str] | None = None,
    ) -> list[ArticleResult]:
        """Run Chroma search, group chunks by article, and apply scoring helpers."""

        collection = self._collection()
        collection_size = safe_collection_count(collection)
        if collection_size == 0:
            return []

        model = self._model()
        encode_query = getattr(model, "encode_query", None)
        query_embedding = (
            encode_query(query) if callable(encode_query) else model.encode_texts([query])[0]
        )
        n_results = min(candidate_count, collection_size)
        where = build_date_where(date_from=date_from, date_to=date_to)
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            query_kwargs["where"] = where
        response = collection.query(**query_kwargs)
        if self.hybrid_enabled:
            try:
                lexical = self._fts_index().search(
                    collection,
                    lexical_query_terms(query, lexical_expansion_terms or expansion_terms),
                    limit=n_results,
                    category=category,
                    date_from=parse_date_to_int(date_from) if date_from else None,
                    date_to=parse_date_to_int(date_to) if date_to else None,
                )
                response = fuse_dense_and_lexical(response, lexical)
            except Exception as exc:  # noqa: BLE001 - dense search remains available
                logger.warning("FTS hybrid search unavailable; using dense results: %s", exc)
        results = group_query_response(response, top_k=n_results, category=category)
        results = rerank_results(results, expansion_terms or [])
        reranker = self._local_reranker()
        if reranker is not None:
            try:
                results = reranker.rerank(query, results)
                RelevanceScorer(expansion_terms or []).apply(results)
            except Exception as exc:  # noqa: BLE001 - retrieval must degrade safely
                logger.warning("Local reranker unavailable; keeping fused ranking: %s", exc)
        return results

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

    def _rewriter(self) -> QueryRewriter:
        if self.rewriter is None:
            self.rewriter = QueryRewriter()
        return self.rewriter

    def _fts_index(self) -> SQLiteFTSIndex:
        if self.fts_index is None:
            self.fts_index = SQLiteFTSIndex(
                self.data_dir / FTS_FILENAME,
                # Deployment archives can round file timestamps. Their bundled
                # FTS snapshot is validated before upload and cannot be rebuilt.
                revision=None
                if os.getenv("LENS_RAG_IMMUTABLE_ARCHIVE") == "1"
                else manifest_revision(self.data_dir),
            )
        return self.fts_index

    def _local_reranker(self) -> Any | None:
        if self.reranker is None and self.reranker_model:
            self.reranker = LocalCrossEncoderReranker(self.reranker_model)
        return self.reranker

    def _topic_lexicon(self) -> TopicLexicon:
        if self.topic_lexicon is None:
            self.topic_lexicon = TopicLexicon.load(
                self.data_dir / TOPIC_LEXICON_FILENAME
            )
        return self.topic_lexicon

    def _knowledge_graph(self) -> KnowledgeGraphIndex:
        if self.knowledge_graph_index is None:
            self.knowledge_graph_index = KnowledgeGraphIndex(
                self.data_dir / GRAPH_FILENAME
            )
        return self.knowledge_graph_index


def fuse_dense_and_lexical(
    dense_response: dict[str, Any],
    lexical_matches: list[LexicalMatch],
    rank_constant: int = 60,
) -> dict[str, list[list[Any]]]:
    """Fuse dense and lexical chunk ranks using reciprocal-rank fusion."""

    records: dict[str, dict[str, Any]] = {}
    dense_ids = _first(dense_response.get("ids"))
    dense_documents = _first(dense_response.get("documents"))
    dense_metadatas = _first(dense_response.get("metadatas"))
    dense_distances = _first(dense_response.get("distances"))
    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(dense_ids, dense_documents, dense_metadatas, dense_distances),
        start=1,
    ):
        dense_rrf = 1.0 / (rank_constant + rank)
        key = str(chunk_id) or _chunk_key(metadata or {})
        records[key] = {
            "id": key,
            "document": str(document or ""),
            "metadata": metadata or {},
            "rrf": dense_rrf,
            "dense_rrf": dense_rrf,
            "lexical_rrf": 0.0,
            "dense_score": distance_to_score(distance),
        }

    for rank, match in enumerate(lexical_matches, start=1):
        lexical_rrf = 1.0 / (rank_constant + rank)
        key = match.chunk_id or _chunk_key(match.metadata)
        record = records.setdefault(
            key,
            {
                "id": key,
                "document": match.document,
                "metadata": match.metadata,
                "rrf": 0.0,
                "dense_rrf": 0.0,
                "lexical_rrf": 0.0,
                "dense_score": 0.0,
            },
        )
        record["rrf"] += lexical_rrf
        record["lexical_rrf"] += lexical_rrf

    ordered = sorted(records.values(), key=lambda item: item["rrf"], reverse=True)
    maximum_rrf = 2.0 / (rank_constant + 1)
    maximum_single_retriever_rrf = 1.0 / (rank_constant + 1)
    distances: list[float] = []
    for item in ordered:
        fusion_score = min(1.0, float(item["rrf"]) / maximum_rrf)
        lexical_score = (
            min(1.0, float(item["lexical_rrf"]) / maximum_single_retriever_rrf) * 0.68
        )
        # Preserve a strong semantic score while letting agreement between the
        # two retrievers promote exact-name and exact-phrase matches.
        score = max(float(item["dense_score"]), fusion_score, lexical_score)
        distances.append(1.0 - score)
    return {
        "ids": [[item["id"] for item in ordered]],
        "documents": [[item["document"] for item in ordered]],
        "metadatas": [[item["metadata"] for item in ordered]],
        "distances": [distances],
    }


def lexical_query_terms(
    query: str,
    expansion_terms: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Return phrase-first FTS terms plus token/alias expansion terms."""

    terms: list[str] = []
    query_terms = keywords_from_query(query)
    if len(query_terms) > 1:
        terms.append(" ".join(query_terms))
    elif query.strip():
        terms.append(query.strip())
    terms.extend(expansion_terms or query_terms)
    return terms


def _chunk_key(metadata: dict[str, Any]) -> str:
    return f"{metadata.get('article_id', '')}:{metadata.get('chunk_index', 0)}"


def _graph_excerpts(
    content: str,
    query: str,
    entities: list[str],
    limit: int = 3,
) -> list[str]:
    """Select local paragraphs that explain why an article lies on a graph path."""

    candidates = [
        " ".join(paragraph.split())
        for paragraph in re.split(r"\n{2,}", content)
        if len(" ".join(paragraph.split())) >= 80
    ]
    if not candidates and content.strip():
        candidates = [" ".join(content.split())]
    terms = keywords_from_query(" ".join([query, *entities]))
    scored = sorted(
        enumerate(candidates),
        key=lambda item: (
            -sum(item[1].lower().count(term.lower()) for term in terms),
            item[0],
        ),
    )
    excerpts: list[str] = []
    for _, paragraph in scored:
        excerpt = paragraph[:1_200].rsplit(" ", 1)[0].strip()
        if excerpt and excerpt not in excerpts:
            excerpts.append(excerpt)
        if len(excerpts) >= limit:
            break
    return excerpts


def manifest_embedding_model(data_dir: Path) -> str | None:
    """Return the single model recorded for the completed local index."""

    path = Path(data_dir) / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        articles = payload.get("articles", {})
        models = {
            str(entry.get("embedding_model"))
            for entry in articles.values()
            if isinstance(entry, dict) and entry.get("embedded") and entry.get("embedding_model")
        }
        return next(iter(models)) if len(models) == 1 else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def manifest_revision(data_dir: Path) -> str | None:
    """Return a cheap revision token for indexes derived from the manifest."""

    path = Path(data_dir) / "manifest.json"
    try:
        stat = path.stat()
        return f"{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        return None


def manifest_categories(data_dir: Path) -> list[str]:
    """Return distinct category names recorded in the local archive manifest."""

    path = Path(data_dir) / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        articles = payload.get("articles", {})
        categories = {
            str(category).strip()
            for entry in articles.values()
            if isinstance(entry, dict)
            for category in (entry.get("category_names") or [])
            if str(category).strip()
        }
        return sorted(categories)
    except (OSError, ValueError, TypeError, AttributeError):
        return []


def candidate_chunk_count(
    page: int,
    page_size: int,
    collection_size: int,
    category: str | None = None,
) -> int:
    """Return how many chunks to fetch for one paginated article page."""

    if collection_size <= 0:
        return 0
    multiplier = 10 if category else 5
    requested_articles = (page * page_size) + 1
    target = max(50, requested_articles * multiplier)
    return min(target, 500, collection_size)


def rerank_results(
    results: list[ArticleResult],
    expansion_terms: list[str] | tuple[str, ...],
) -> list[ArticleResult]:
    """Apply keyword boosting and attach interpretable score metadata."""

    terms = list(expansion_terms)
    if not results:
        return results
    if terms:
        booster = KeywordBooster(terms)
        for result in results:
            combined_text = " ".join(
                [
                    result.title,
                    result.categories,
                    result.tags,
                    result.matches[0].excerpt if result.matches else "",
                ]
            )
            result.score = booster.boost(combined_text, result.score)
            for match in result.matches:
                match.score = booster.boost(match.excerpt, match.score)
            result.matches.sort(key=lambda item: item.score, reverse=True)
        results.sort(key=lambda item: item.score, reverse=True)
    RelevanceScorer(terms).apply(results)
    return results


def build_date_where(
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any] | None:
    """Build a Chroma metadata filter for an optional date range."""

    clauses: list[dict[str, Any]] = []
    if date_from:
        clauses.append({"date_int": {"$gte": parse_date_to_int(date_from)}})
    if date_to:
        clauses.append({"date_int": {"$lte": parse_date_to_int(date_to)}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def group_query_response(
    response: dict[str, Any],
    top_k: int,
    category: str | None = None,
) -> list[ArticleResult]:
    """Group a raw Chroma query response into article-level results."""

    documents = _first(response.get("documents"))
    metadatas = _first(response.get("metadatas"))
    distances = _first(response.get("distances"))
    grouped: dict[str, ArticleResult] = {}

    for document, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        if category and not metadata_matches_category(metadata, category):
            continue

        article_id = str(metadata.get("article_id", ""))
        if not article_id:
            continue
        score = distance_to_score(distance)
        document_text = str(document or "")
        match = ChunkMatch(
            score=score,
            excerpt=make_excerpt(document_text),
            chunk_index=int(metadata.get("chunk_index") or 0),
        )
        title = str(metadata.get("title", "")) or document_labeled_field(
            document_text,
            "Title",
        )

        if article_id not in grouped:
            grouped[article_id] = ArticleResult(
                article_id=article_id,
                title=title,
                date=str(metadata.get("date", "")),
                url=str(metadata.get("url", "")),
                categories=str(metadata.get("categories", "")),
                tags=str(metadata.get("tags", "")),
                score=score,
                matches=[match],
            )
        else:
            result = grouped[article_id]
            result.score = max(result.score, score)
            result.matches.append(match)

    results = sorted(grouped.values(), key=lambda item: item.score, reverse=True)
    for result in results:
        result.matches.sort(key=lambda item: item.score, reverse=True)
    return results[:top_k]


def metadata_matches_category(metadata: dict[str, Any], category: str) -> bool:
    """Return true when metadata category names or slugs match the filter."""

    needle = category.strip().lower()
    if not needle:
        return True
    haystack: list[str] = []
    for field in ("categories", "category_slugs"):
        value = str(metadata.get(field, "")).lower()
        haystack.extend(part.strip() for part in value.split(";") if part.strip())
    return any(needle == value or needle in value for value in haystack)


def distance_to_score(distance: float | int | None) -> float:
    """Convert a cosine distance to an easy-to-read relevance score."""

    if distance is None:
        return 0.0
    return round(max(0.0, min(1.0, 1.0 - float(distance))), 4)


def make_excerpt(document: str, max_chars: int = 500) -> str:
    """Return a compact excerpt from a stored chunk document."""

    content = document.split("Content:", 1)[-1] if "Content:" in document else document
    content = re.sub(r"\s+", " ", content).strip()
    if len(content) <= max_chars:
        return content
    return content[: max_chars - 1].rstrip() + "..."


def format_results_text(results: list[ArticleResult]) -> str:
    """Format article search results for terminal display."""

    if not results:
        return "No matching articles found."

    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result.title} ({result.date})")
        lines.append(f"   URL: {result.url}")
        if result.categories:
            lines.append(f"   Categories: {result.categories}")
        lines.append(f"   Relevance: {result.score:.3f}")
        if result.matches:
            lines.append(f"   Match: {result.matches[0].excerpt}")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_results_json(results: list[ArticleResult]) -> str:
    """Format article search results as JSON."""

    return json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2)


def _first(value: Any) -> list[Any]:
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    if isinstance(value, list):
        return value
    return []


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the query CLI argument parser."""

    parser = argparse.ArgumentParser(description="Query The Lens RAG archive")
    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--category", default=None)
    parser.add_argument("--date-from", default=None)
    parser.add_argument("--date-to", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--interactive", action="store_true")
    return parser


def run_interactive(engine: QueryEngine, args: argparse.Namespace) -> int:
    """Run an interactive query REPL. Uses Rich when available."""

    try:
        from .query_rich import run_interactive_rich

        return run_interactive_rich(engine, top_k=args.top_k)
    except ImportError:
        pass

    print("Lens RAG interactive search. Type 'exit' or press Ctrl-D to quit.")
    while True:
        try:
            query = input("lens-rag> ").strip()
        except EOFError:
            print()
            return 0
        if query.lower() in {"exit", "quit"}:
            return 0
        if not query:
            continue
        try:
            results = engine.search(
                query,
                top_k=args.top_k,
                category=args.category,
                date_from=args.date_from,
                date_to=args.date_to,
            )
        except (ImportError, ValueError) as exc:
            print(f"Query failed: {exc}", file=sys.stderr)
            return 1
        print(format_results_text(results))
        print()


def main(argv: list[str] | None = None) -> int:
    """Run the query CLI."""

    from .nl_query import load_local_env

    load_local_env()
    args = build_arg_parser().parse_args(argv)
    engine = QueryEngine(data_dir=args.data_dir, model_name=args.model)
    if args.interactive:
        return run_interactive(engine, args)
    if not args.query:
        raise SystemExit("Provide a query string or use --interactive")
    try:
        results = engine.search(
            args.query,
            top_k=args.top_k,
            category=args.category,
            date_from=args.date_from,
            date_to=args.date_to,
        )
    except (ImportError, ValueError) as exc:
        print(f"Query failed: {exc}", file=sys.stderr)
        return 1
    if args.format == "json":
        print(format_results_json(results))
    else:
        try:
            from .query_rich import format_results

            print(format_results(results, query=args.query))
        except ImportError:
            print(format_results_text(results))
    return 0
