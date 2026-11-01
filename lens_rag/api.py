"""FastAPI app for searching and browsing the Lens RAG archive."""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Callable, Optional, Union
from urllib.parse import urlencode

# Load local .env before any other imports so LENS_RAG_LLM_API_KEY
# is available for the NL query module.
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    with _ENV_PATH.open("r", encoding="utf-8") as _env_handle:
        for _line in _env_handle:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _value = _line.partition("=")
                if _key.strip() and _key.strip() not in os.environ:
                    os.environ[_key.strip()] = _value.strip()

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import (
    ARTICLES_DIRNAME,
    CHROMA_DIRNAME,
    COLLECTION_NAME,
    DEFAULT_DATA_DIR,
    MANIFEST_FILENAME,
)
from .embedder import load_article, safe_collection_count
from .manifest import Manifest
from .knowledge_graph import GRAPH_FILENAME, KnowledgeGraphIndex
from .nl_query import (
    AnswerGenerator,
    DeepSeekAnswerGenerator,
    DeepSeekProAnswerGenerator,
    OpusAnswerGenerator,
    OpenAIAnswerGenerator,
)
from .query_engine import QueryEngine, manifest_embedding_model
from .text import html_to_plain_text, parse_date_to_int, safe_date_int


DEFAULT_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 5367


def _local_api_port() -> int:
    """Return a valid local API port, falling back to the project default."""

    try:
        port = int(os.getenv("LENS_RAG_API_PORT", str(DEFAULT_API_PORT)))
    except (TypeError, ValueError):
        return DEFAULT_API_PORT
    return port if 1024 <= port <= 65535 else DEFAULT_API_PORT


class SearchRequest(BaseModel):
    """Payload for archive search."""

    query: str = Field(..., min_length=1, max_length=500)
    top_k: Optional[int] = Field(None, ge=1, le=50)
    page: int = Field(1, ge=1)
    page_size: Optional[int] = Field(None, ge=1, le=50)
    nl_rewrite: bool = True
    category: Optional[str] = Field(None, max_length=200)
    date_from: Optional[str] = Field(None, max_length=10)
    date_to: Optional[str] = Field(None, max_length=10)


class DeepSeekAnswerRequest(SearchRequest):
    """DeepSeek synthesis request with an optional local draft A."""

    draft_answer: Optional[str] = Field(None, max_length=20_000)


def create_app(
    data_dir: Optional[Union[Path, str]] = None,
    frontend_dir: Optional[Union[Path, str]] = None,
    query_engine: Optional[Any] = None,
    answer_generator: Optional[Any] = None,
    deepseek_answer_generator: Optional[Any] = None,
    deepseek_pro_answer_generator: Optional[Any] = None,
    opus_answer_generator: Optional[Any] = None,
    openai_answer_generator: Optional[Any] = None,
) -> FastAPI:
    """Create the FastAPI application with injectable state for tests."""

    resolved_data_dir = Path(
        data_dir or os.getenv("LENS_RAG_DATA_DIR", str(DEFAULT_DATA_DIR))
    )
    resolved_frontend_dir = Path(
        frontend_dir or os.getenv("LENS_RAG_FRONTEND_DIR", str(DEFAULT_FRONTEND_DIR))
    )

    app = FastAPI(
        title="Lens NOLA RAG API",
        description="Search and browse The Lens New Orleans article archive.",
        version="0.1.0",
    )
    app.state.data_dir = resolved_data_dir
    app.state.query_engine = query_engine
    app.state.answer_generator = answer_generator
    app.state.deepseek_answer_generator = deepseek_answer_generator
    app.state.deepseek_pro_answer_generator = deepseek_pro_answer_generator
    app.state.opus_answer_generator = opus_answer_generator
    app.state.openai_answer_generator = openai_answer_generator
    app.state.openai_answer_rate_limiter = AnswerRateLimiter(limit=3, window_seconds=3600)
    app.state.answer_rate_limiter = AnswerRateLimiter(limit=10, window_seconds=3600)
    app.state.deepseek_answer_rate_limiter = AnswerRateLimiter(
        limit=10, window_seconds=3600
    )
    app.state.deepseek_pro_answer_rate_limiter = AnswerRateLimiter(
        limit=5, window_seconds=3600
    )
    app.state.opus_answer_rate_limiter = AnswerRateLimiter(
        limit=3, window_seconds=3600
    )

    origins_raw = os.getenv("LENS_RAG_CORS_ORIGINS", "")
    origins = (
        [origin.strip() for origin in origins_raw.split(",") if origin.strip()]
        if origins_raw
        else []
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins if origins else [],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/")
    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        data_path = _data_dir(request)
        manifest = _manifest(data_path)
        articles_dir = data_path / ARTICLES_DIRNAME
        chroma_dir = data_path / CHROMA_DIRNAME
        file_count = (
            sum(1 for _ in articles_dir.glob("*.json")) if articles_dir.exists() else 0
        )

        collection_count: int | None = None
        collection_error: str | None = None
        embedding_model = manifest_embedding_model(data_path)
        answerer = _answer_generator(request)
        answer_configured = bool(answerer.supports_answering())
        deepseek_answerer = _deepseek_answer_generator(request)
        deepseek_configured = bool(deepseek_answerer.supports_answering())
        deepseek_pro_answerer = _deepseek_pro_answer_generator(request)
        deepseek_pro_configured = bool(deepseek_pro_answerer.supports_answering())
        opus_answerer = _opus_answer_generator(request)
        opus_configured = bool(opus_answerer.supports_answering())
        openai_answerer = _openai_answer_generator(request)
        openai_configured = bool(openai_answerer.supports_answering())
        graph_metadata = KnowledgeGraphIndex(data_path / GRAPH_FILENAME).metadata()
        try:
            # In test mode, use injected engine's collection. In production,
            # load collection directly to avoid pulling the ~90MB embedding model.
            injected = getattr(request.app.state, "query_engine", None)
            if injected is not None:
                collection = injected._collection()
                embedding_model = getattr(injected, "model_name", None) or embedding_model
            else:
                from .embedder import get_chroma_collection

                collection = get_chroma_collection(chroma_dir)
            collection_count = safe_collection_count(collection)
        except Exception as exc:  # noqa: BLE001 - health should report, not crash
            import logging
            import traceback
            logging.getLogger(__name__).error(
                "Health check collection load failed: %s\n%s",
                exc,
                traceback.format_exc(),
            )
            collection_error = "unavailable"

        return {
            "status": "ok" if file_count and collection_count else "degraded",
            "data_dir": str(data_path),
            "articles": {
                "manifest_count": manifest.downloaded_count(),
                "file_count": file_count,
                "embedded_count": manifest.embedded_count(),
            },
            "collection": {
                "name": COLLECTION_NAME,
                "path": str(chroma_dir),
                "chunk_count": collection_count,
                "embedding_model": embedding_model,
                "error": collection_error,
            },
            "knowledge_graph": graph_metadata,
            "features": {
                "query_rewriting": _supports_query_rewriting(request),
                "answer_generation": _answer_enabled() and answer_configured,
                "answer_backend": {
                    "provider": getattr(answerer, "provider", None),
                    "model": getattr(answerer, "model", None),
                    "configured": answer_configured,
                },
                "deepseek_synthesis": _deepseek_enabled() and deepseek_configured,
                "deepseek_backend": {
                    "provider": getattr(deepseek_answerer, "provider", None),
                    "model": getattr(deepseek_answerer, "model", None),
                    "base_url": getattr(deepseek_answerer, "base_url", None),
                    "configured": deepseek_configured,
                },
                "deepseek_pro_synthesis": _deepseek_enabled() and deepseek_pro_configured,
                "deepseek_pro_backend": {
                    "provider": getattr(deepseek_pro_answerer, "provider", None),
                    "model": getattr(deepseek_pro_answerer, "model", None),
                    "base_url": getattr(deepseek_pro_answerer, "base_url", None),
                    "configured": deepseek_pro_configured,
                },
                "opus_synthesis": _opus_enabled() and opus_configured,
                "openai_synthesis": _openai_enabled() and openai_configured,
                "openai_backend": {
                    "provider": openai_answerer.provider,
                    "model": openai_answerer.model,
                    "configured": openai_configured,
                },
                "opus_backend": {
                    "provider": getattr(opus_answerer, "provider", None),
                    "model": getattr(opus_answerer, "model", None),
                    "base_url": getattr(opus_answerer, "base_url", None),
                    "configured": opus_configured,
                    "provider_route": "direct",
                },
            },
        }

    @app.post("/search")
    def search(payload: SearchRequest, request: Request) -> dict[str, Any]:
        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="query is required")

        try:
            page, page_size = _search_pagination(payload)
            engine = _engine(request)
            if hasattr(engine, "search_results"):
                search_page = engine.search_results(
                    query,
                    page=page,
                    page_size=page_size,
                    category=payload.category,
                    date_from=payload.date_from,
                    date_to=payload.date_to,
                    nl_rewrite=payload.nl_rewrite,
                    available_categories=_available_category_names(request)
                    if payload.nl_rewrite
                    else None,
                )
                results = search_page.results
                response = {
                    "query": query,
                    "effective_query": search_page.effective_query,
                    "top_k": payload.top_k if payload.top_k is not None else page_size,
                    "page": search_page.page,
                    "page_size": search_page.page_size,
                    "total": search_page.total,
                    "has_more": search_page.has_more,
                    "filters": {
                        "category": search_page.category,
                        "date_from": search_page.date_from,
                        "date_to": search_page.date_to,
                    },
                    "rewrite": search_page.rewrite.to_dict()
                    if search_page.rewrite
                    else None,
                    "keywords": search_page.keywords,
                    "count": len(results),
                    "results": [result.to_dict() for result in results],
                }
                return response

            results = engine.search(
                query,
                top_k=page_size,
                category=payload.category,
                date_from=payload.date_from,
                date_to=payload.date_to,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "query": query,
            "effective_query": query,
            "top_k": payload.top_k if payload.top_k is not None else page_size,
            "page": page,
            "page_size": page_size,
            "total": len(results),
            "has_more": False,
            "filters": {
                "category": payload.category,
                "date_from": payload.date_from,
                "date_to": payload.date_to,
            },
            "rewrite": None,
            "keywords": [],
            "count": len(results),
            "results": [result.to_dict() for result in results],
        }

    @app.post("/search/answer")
    def search_answer(payload: SearchRequest, request: Request) -> dict[str, Any]:
        if not _answer_enabled():
            raise HTTPException(
                status_code=501,
                detail="Answer generation is disabled. Set LENS_RAG_ENABLE_ANSWER=true to enable it.",
            )

        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="query is required")

        try:
            results, effective_query = _search_answer_context(payload, request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        generator = _answer_generator(request)
        if not generator.supports_answering():
            raise HTTPException(
                status_code=503,
                detail="Answer generation requires a configured local or OpenAI backend.",
            )

        limiter = _answer_rate_limiter(request)
        if not limiter.allow():
            raise HTTPException(
                status_code=429,
                detail="Answer rate limit exceeded: max 10 answers/hour.",
            )

        answer_results = results[:_answer_source_limit()]
        try:
            answer = generator.generate(effective_query, answer_results)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - translate LLM failures to API errors
            raise HTTPException(status_code=502, detail="Answer generation failed") from exc

        return {
            "query": query,
            "stage": "A",
            "provider": getattr(generator, "provider", "local"),
            "model": getattr(generator, "model", None),
            "answer": answer,
            "sources": _answer_sources(answer_results),
            "result_count": len(answer_results),
        }

    @app.post("/search/answer/deepseek")
    def search_answer_deepseek(
        payload: DeepSeekAnswerRequest,
        request: Request,
    ) -> dict[str, Any]:
        if not _deepseek_enabled():
            raise HTTPException(
                status_code=501,
                detail=(
                    "DeepSeek synthesis is disabled. Set "
                    "LENS_RAG_ENABLE_DEEPSEEK=true to enable it."
                ),
            )

        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="query is required")

        try:
            results, effective_query = _search_answer_context(payload, request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        generator = _deepseek_answer_generator(request)
        if not generator.supports_answering():
            raise HTTPException(
                status_code=503,
                detail=(
                    "DeepSeek synthesis requires DEEPSEEK_API_KEY (or "
                    "LENS_RAG_DEEPSEEK_API_KEY)."
                ),
            )

        limiter = _deepseek_answer_rate_limiter(request)
        if not limiter.allow():
            raise HTTPException(
                status_code=429,
                detail="DeepSeek rate limit exceeded: max 10 syntheses/hour.",
            )

        answer_results = results[:_answer_source_limit()]
        try:
            answer = generator.generate(
                effective_query,
                answer_results,
                payload.draft_answer,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - translate upstream failures
            raise HTTPException(
                status_code=502,
                detail="Direct DeepSeek synthesis failed",
            ) from exc

        return {
            "query": query,
            "stage": "A",
            "provider": getattr(generator, "provider", "deepseek"),
            "model": getattr(generator, "model", None),
            "answer": answer,
            "sources": _answer_sources(answer_results),
            "result_count": len(answer_results),
        }

    @app.post("/search/answer/deepseek-pro")
    def search_answer_deepseek_pro(
        payload: SearchRequest,
        request: Request,
    ) -> StreamingResponse:
        if not _deepseek_enabled():
            raise HTTPException(status_code=501, detail="DeepSeek synthesis is disabled.")

        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="query is required")

        try:
            results, effective_query = _search_answer_context(payload, request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        generator = _deepseek_pro_answer_generator(request)
        if not generator.supports_answering():
            raise HTTPException(
                status_code=503,
                detail="DeepSeek V4 Pro requires a configured DeepSeek API key.",
            )

        limiter = _deepseek_pro_answer_rate_limiter(request)
        if not limiter.allow():
            raise HTTPException(
                status_code=429,
                detail="DeepSeek V4 Pro rate limit exceeded: max 5 syntheses/hour.",
            )

        def synthesize() -> dict[str, Any]:
            from .synthesis import build_research_context

            source_limit = _pro_answer_source_limit()
            research = build_research_context(
                _engine(request),
                effective_query,
                results,
                category=payload.category,
                date_from=payload.date_from,
                date_to=payload.date_to,
                max_articles=source_limit,
                facet_top_k=min(10, source_limit),
            )
            answer_results = research.results[:source_limit]
            if hasattr(generator, "generate_research"):
                answer = generator.generate_research(
                    effective_query,
                    answer_results,
                    source_facets=research.source_facets,
                    research_plan=research.prompt_plan(),
                )
            else:
                answer = generator.generate(effective_query, answer_results)

            return {
                "query": query,
                "stage": "B",
                "provider": getattr(generator, "provider", "deepseek"),
                "model": getattr(generator, "model", None),
                "answer": answer,
                "sources": _answer_sources(answer_results),
                "result_count": len(answer_results),
                "diagnostics": {
                    **research.diagnostics(),
                    **getattr(generator, "last_diagnostics", {}),
                },
            }

        return _stream_long_json(
            synthesize,
            failure_detail="DeepSeek V4 Pro synthesis failed",
        )

    @app.post("/search/answer/openai")
    @app.post("/search/answer/opus")
    def search_answer_independent(
        payload: SearchRequest,
        request: Request,
    ) -> StreamingResponse:
        use_openai = request.url.path.endswith("/openai")
        label = "GPT-6.1 Sol" if use_openai else "Claude Opus 5"
        enabled = _openai_enabled() if use_openai else _opus_enabled()
        enable_variable = "LENS_RAG_ENABLE_OPENAI" if use_openai else "LENS_RAG_ENABLE_OPUS"
        if not enabled:
            raise HTTPException(
                status_code=501,
                detail=(
                    f"{label} synthesis is disabled. Set "
                    f"{enable_variable}=true to enable it."
                ),
            )

        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="query is required")

        try:
            results, effective_query = _search_answer_context(payload, request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        generator = _openai_answer_generator(request) if use_openai else _opus_answer_generator(request)
        if not generator.supports_answering():
            raise HTTPException(
                status_code=503,
                detail=(
                    "GPT-6.1 Sol requires OPENAI_API_KEY for direct OpenAI access."
                    if use_openai else
                    "Claude Opus 5 requires ANTHROPIC_API_KEY for direct Anthropic access."
                ),
            )

        limiter = request.app.state.openai_answer_rate_limiter if use_openai else _opus_answer_rate_limiter(request)
        if not limiter.allow():
            raise HTTPException(
                status_code=429,
                detail=f"{label} rate limit exceeded: max 3 syntheses/hour.",
            )

        def synthesize() -> dict[str, Any]:
            from .synthesis import build_research_context

            source_limit = _openai_answer_source_limit() if use_openai else _opus_answer_source_limit()
            research = build_research_context(
                _engine(request),
                effective_query,
                results,
                category=payload.category,
                date_from=payload.date_from,
                date_to=payload.date_to,
                max_articles=source_limit,
                facet_top_k=min(12, source_limit),
            )
            answer_results = research.results[:source_limit]
            answer = generator.generate_research(
                effective_query,
                answer_results,
                source_facets=research.source_facets,
                research_plan=research.prompt_plan(),
            )

            return {
                "query": query,
                "stage": "D" if use_openai else "C",
                "provider": getattr(generator, "provider", "openai" if use_openai else "anthropic-direct"),
                "model": getattr(generator, "model", None),
                "answer": answer,
                "sources": _answer_sources(answer_results),
                "result_count": len(answer_results),
                "diagnostics": {
                    **research.diagnostics(),
                    **getattr(generator, "last_diagnostics", {}),
                },
            }

        return _stream_long_json(
            synthesize,
            failure_detail=f"{label} direct synthesis failed",
        )

    @app.get("/article/{article_id}")
    def article(article_id: str, request: Request) -> dict[str, Any]:
        article_data = _load_article_by_id(_data_dir(request), article_id)
        return _article_response(article_data)

    @app.get("/articles")
    def articles(
        request: Request,
        page: int = Query(1, ge=1),
        page_size: int = Query(25, ge=1, le=100),
        category: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> dict[str, Any]:
        try:
            items = _filtered_manifest_articles(
                _manifest(_data_dir(request)),
                category=category,
                date_from=date_from,
                date_to=date_to,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        total = len(items)
        start = (page - 1) * page_size
        end = start + page_size
        return {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": math.ceil(total / page_size) if total else 0,
            "items": items[start:end],
        }

    @app.get("/categories")
    def categories(request: Request) -> dict[str, Any]:
        rows = _category_counts(_manifest(_data_dir(request)))
        return {"count": len(rows), "categories": rows}

    if resolved_frontend_dir.exists():
        index_path = resolved_frontend_dir / "index.html"

        @app.get("/app", include_in_schema=False)
        def frontend(request: Request) -> Any:
            if not index_path.exists():
                raise HTTPException(status_code=404, detail="frontend not found")
            revision = (
                os.getenv("LENS_RAG_FRONTEND_REVISION", "").strip()
                or str(index_path.stat().st_mtime_ns)
            )
            cache_headers = {
                "Cache-Control": (
                    "no-store, no-cache, must-revalidate, max-age=0, private"
                ),
                "Pragma": "no-cache",
                "Expires": "0",
                "Surrogate-Control": "no-store",
                "Clear-Site-Data": '"cache"',
                "X-Lens-Frontend-Revision": revision,
            }
            if request.query_params.get("v") != revision:
                params = dict(request.query_params)
                params["v"] = revision
                return RedirectResponse(
                    url=f"{request.url.path}?{urlencode(params)}",
                    status_code=307,
                    headers=cache_headers,
                )
            return FileResponse(
                index_path,
                headers=cache_headers,
            )

        app.mount(
            "/static",
            StaticFiles(directory=resolved_frontend_dir),
            name="static",
        )

    return app


def _data_dir(request: Request) -> Path:
    return Path(request.app.state.data_dir)


def _stream_long_json(
    work: Callable[[], dict[str, Any]],
    *,
    failure_detail: str,
    heartbeat_seconds: float = 10.0,
) -> StreamingResponse:
    """Stream JSON-safe whitespace while a long synthesis runs in one worker."""

    def body():
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(work)
        try:
            # A substantial whitespace prefix forces intermediaries to flush response
            # headers. JSON parsers legally ignore it before the final object.
            yield (" " * 2048 + "\n").encode("utf-8")
            while True:
                try:
                    payload = future.result(timeout=heartbeat_seconds)
                    break
                except FutureTimeoutError:
                    yield (" " * 2048 + "\n").encode("utf-8")
                except RuntimeError as exc:
                    payload = {"detail": str(exc)}
                    break
                except Exception:  # noqa: BLE001 - serialize after headers were sent
                    import logging

                    logging.getLogger(__name__).exception(failure_detail)
                    payload = {"detail": failure_detail}
                    break
            yield json.dumps(payload, ensure_ascii=False).encode("utf-8")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    return StreamingResponse(
        body(),
        media_type="application/json",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


import threading

_engine_lock = threading.Lock()


class AnswerRateLimiter:
    """Simple process-local sliding-window limiter for answer generation."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque()

    def allow(self) -> bool:
        now = time.time()
        cutoff = now - self.window_seconds
        with self._lock:
            while self._timestamps and self._timestamps[0] <= cutoff:
                self._timestamps.popleft()
            if len(self._timestamps) >= self.limit:
                return False
            self._timestamps.append(now)
            return True


def _engine(request: Request) -> QueryEngine:
    engine = getattr(request.app.state, "query_engine", None)
    if engine is None:
        with _engine_lock:
            engine = getattr(request.app.state, "query_engine", None)
            if engine is None:
                engine = QueryEngine(data_dir=_data_dir(request))
                request.app.state.query_engine = engine
    return engine


def _answer_generator(request: Request) -> AnswerGenerator:
    generator = getattr(request.app.state, "answer_generator", None)
    if generator is None:
        generator = AnswerGenerator()
        request.app.state.answer_generator = generator
    return generator


def _deepseek_answer_generator(request: Request) -> DeepSeekAnswerGenerator:
    generator = getattr(request.app.state, "deepseek_answer_generator", None)
    if generator is None:
        generator = DeepSeekAnswerGenerator()
        request.app.state.deepseek_answer_generator = generator
    return generator


def _deepseek_pro_answer_generator(request: Request) -> DeepSeekProAnswerGenerator:
    generator = getattr(request.app.state, "deepseek_pro_answer_generator", None)
    if generator is None:
        generator = DeepSeekProAnswerGenerator()
        request.app.state.deepseek_pro_answer_generator = generator
    return generator


def _opus_answer_generator(request: Request) -> OpusAnswerGenerator:
    generator = getattr(request.app.state, "opus_answer_generator", None)
    if generator is None:
        generator = OpusAnswerGenerator()
        request.app.state.opus_answer_generator = generator
    return generator


def _openai_answer_generator(request: Request) -> OpenAIAnswerGenerator:
    generator = request.app.state.openai_answer_generator
    if generator is None:
        generator = OpenAIAnswerGenerator()
        request.app.state.openai_answer_generator = generator
    return generator


def _answer_rate_limiter(request: Request) -> AnswerRateLimiter:
    limiter = getattr(request.app.state, "answer_rate_limiter", None)
    if limiter is None:
        limiter = AnswerRateLimiter(limit=10, window_seconds=3600)
        request.app.state.answer_rate_limiter = limiter
    return limiter


def _deepseek_answer_rate_limiter(request: Request) -> AnswerRateLimiter:
    limiter = getattr(request.app.state, "deepseek_answer_rate_limiter", None)
    if limiter is None:
        limiter = AnswerRateLimiter(limit=10, window_seconds=3600)
        request.app.state.deepseek_answer_rate_limiter = limiter
    return limiter


def _deepseek_pro_answer_rate_limiter(request: Request) -> AnswerRateLimiter:
    limiter = getattr(request.app.state, "deepseek_pro_answer_rate_limiter", None)
    if limiter is None:
        limiter = AnswerRateLimiter(limit=5, window_seconds=3600)
        request.app.state.deepseek_pro_answer_rate_limiter = limiter
    return limiter


def _opus_answer_rate_limiter(request: Request) -> AnswerRateLimiter:
    limiter = getattr(request.app.state, "opus_answer_rate_limiter", None)
    if limiter is None:
        limiter = AnswerRateLimiter(limit=3, window_seconds=3600)
        request.app.state.opus_answer_rate_limiter = limiter
    return limiter


def _supports_query_rewriting(request: Request) -> bool:
    engine = _engine(request)
    if hasattr(engine, "supports_query_rewriting"):
        return bool(engine.supports_query_rewriting())
    return False


def _answer_enabled() -> bool:
    return os.getenv("LENS_RAG_ENABLE_ANSWER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _deepseek_enabled() -> bool:
    return os.getenv("LENS_RAG_ENABLE_DEEPSEEK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _opus_enabled() -> bool:
    return os.getenv("LENS_RAG_ENABLE_OPUS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _openai_enabled() -> bool:
    return os.getenv("LENS_RAG_ENABLE_OPENAI", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _openai_answer_source_limit() -> int:
    try:
        configured = int(os.getenv("LENS_RAG_OPENAI_MAX_SOURCES", "12"))
    except ValueError:
        configured = 12
    return min(16, max(6, configured))


def _answer_source_limit() -> int:
    """Return the bounded number of top search results supplied to synthesis."""

    try:
        configured = int(os.getenv("LENS_RAG_ANSWER_MAX_SOURCES", "4"))
    except ValueError:
        configured = 4
    return min(10, max(1, configured))


def _pro_answer_source_limit() -> int:
    """Return the larger bounded source set for in-depth Pro synthesis."""

    try:
        configured = int(os.getenv("LENS_RAG_DEEPSEEK_PRO_MAX_SOURCES", "10"))
    except ValueError:
        configured = 10
    return min(12, max(4, configured))


def _opus_answer_source_limit() -> int:
    """Return the bounded source set for Opus 5's in-depth synthesis."""

    try:
        configured = int(os.getenv("LENS_RAG_OPUS_MAX_SOURCES", "12"))
    except ValueError:
        configured = 12
    return min(16, max(6, configured))


def _search_answer_context(
    payload: SearchRequest,
    request: Request,
) -> tuple[list[Any], str]:
    """Run the shared retrieval step used by both synthesis stages."""

    query = payload.query.strip()
    page, page_size = _search_pagination(payload)
    engine = _engine(request)
    if hasattr(engine, "search_results"):
        search_page = engine.search_results(
            query,
            page=page,
            page_size=page_size,
            category=payload.category,
            date_from=payload.date_from,
            date_to=payload.date_to,
            nl_rewrite=payload.nl_rewrite,
            available_categories=_available_category_names(request)
            if payload.nl_rewrite
            else None,
        )
        return search_page.results, search_page.effective_query

    return (
        engine.search(
            query,
            top_k=page_size,
            category=payload.category,
            date_from=payload.date_from,
            date_to=payload.date_to,
        ),
        query,
    )


def _search_pagination(payload: SearchRequest) -> tuple[int, int]:
    fields_set = getattr(payload, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(payload, "__fields_set__", set())
    page = payload.page
    page_size = payload.page_size
    if payload.top_k is not None and page_size is None:
        page_size = payload.top_k
        if "page" not in fields_set:
            page = 1
    if page_size is None:
        page_size = 10
    return page, page_size


def _answer_sources(results: list[Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for index, result in enumerate(results, start=1):
        sources.append(
            {
                "index": index,
                "article_id": str(getattr(result, "article_id", "") or ""),
                "title": str(getattr(result, "title", "") or ""),
                "date": getattr(result, "date", "") or "",
                "url": getattr(result, "url", "") or "",
            }
        )
    return sources


def _available_category_names(request: Request) -> list[str]:
    return [str(row["name"]) for row in _category_counts(_manifest(_data_dir(request)))]


def _manifest(data_dir: Path) -> Manifest:
    return Manifest.load(data_dir / MANIFEST_FILENAME)


def _load_article_by_id(data_dir: Path, article_id: str) -> dict[str, Any]:
    articles_dir = (data_dir / ARTICLES_DIRNAME).resolve()
    path = (articles_dir / f"{article_id}.json").resolve()
    if path.parent != articles_dir:
        raise HTTPException(status_code=400, detail="invalid article id")
    if not path.exists():
        raise HTTPException(status_code=404, detail="article not found")
    try:
        return load_article(path)
    except Exception as exc:  # noqa: BLE001 - translate bad local data to HTTP
        raise HTTPException(status_code=500, detail="Failed to load article") from exc


def _article_response(article: dict[str, Any]) -> dict[str, Any]:
    content = str(article.get("plain_text") or "").strip()
    if not content:
        content = html_to_plain_text(str(article.get("content_html") or ""))
    return {
        "id": str(article.get("id", "")),
        "title": str(article.get("title", "")),
        "date": article.get("date"),
        "modified": article.get("modified"),
        "author": article.get("author_name") or article.get("author") or "",
        "categories": _string_list(article.get("category_names", [])),
        "category_slugs": _string_list(article.get("category_slugs", [])),
        "tags": _string_list(article.get("tag_names", [])),
        "tag_slugs": _string_list(article.get("tag_slugs", [])),
        "excerpt": str(article.get("excerpt") or ""),
        "content": content,
        "url": article.get("url") or article.get("link") or "",
        "slug": article.get("slug") or "",
    }


def _filtered_manifest_articles(
    manifest: Manifest,
    category: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> list[dict[str, Any]]:
    start_int = safe_date_int(date_from) if date_from else None
    end_int = safe_date_int(date_to) if date_to else None
    rows: list[dict[str, Any]] = []

    for article_id, entry in manifest.articles.items():
        if not entry.get("content_hash"):
            continue
        if category and not _entry_matches_category(entry, category):
            continue
        date_int = safe_date_int(str(entry.get("date") or "")) if entry.get("date") else 0
        if start_int is not None and date_int < start_int:
            continue
        if end_int is not None and date_int > end_int:
            continue

        rows.append(
            {
                "id": str(entry.get("id") or article_id),
                "title": str(entry.get("title") or ""),
                "date": entry.get("date"),
                "url": entry.get("url") or "",
                "slug": entry.get("slug") or "",
                "categories": _string_list(entry.get("category_names", [])),
                "tags": _string_list(entry.get("tag_names", [])),
                "embedded": bool(entry.get("embedded")),
                "chunk_count": int(entry.get("chunk_count") or 0),
            }
        )

    return sorted(rows, key=lambda item: str(item.get("date") or ""), reverse=True)


def _category_counts(manifest: Manifest) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    for entry in manifest.articles.values():
        if not entry.get("content_hash"):
            continue
        names = _string_list(entry.get("category_names", []))
        slugs = _string_list(entry.get("category_slugs", []))
        for index, name in enumerate(names):
            slug = slugs[index] if index < len(slugs) else _slugify(name)
            key = slug or name.lower()
            row = counts.setdefault(key, {"name": name, "slug": slug, "count": 0})
            row["count"] += 1
    return sorted(counts.values(), key=lambda item: (-int(item["count"]), item["name"]))


def _entry_matches_category(entry: dict[str, Any], category: str) -> bool:
    needle = category.strip().lower()
    values = _string_list(entry.get("category_names", [])) + _string_list(
        entry.get("category_slugs", [])
    )
    return any(
        needle == value.lower() or needle in value.lower()
        for value in values
    )


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in value.split(";") if part.strip()]
    return []


def _slugify(value: str) -> str:
    return "-".join(value.lower().replace("&", "and").split())


app = create_app()


def main() -> int:
    """Run the API with uvicorn for local development."""

    import uvicorn

    host = os.getenv("LENS_RAG_API_HOST", DEFAULT_API_HOST).strip() or DEFAULT_API_HOST
    uvicorn.run("lens_rag.api:app", host=host, port=_local_api_port(), reload=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
