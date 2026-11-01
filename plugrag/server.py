"""Read-only HTTP service; ingestion runs through the CLI or Python SDK."""

from __future__ import annotations

import hmac
import logging
import os
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .core import RAG


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10000)
    top_k: int = Field(default=5, ge=1, le=100)
    category: str | None = None
    date_from: str | None = None
    date_to: str | None = None


def create_app(rag: RAG | None = None, *, api_key: str | None = None) -> FastAPI:
    """Use directly or mount in another FastAPI application."""

    if rag is None:
        from .generation import generator_from_env

        rag = RAG(generator=generator_from_env())
    key = api_key if api_key is not None else os.getenv("RAG_API_KEY", "")
    bearer = HTTPBearer(auto_error=False)

    def authorize(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> None:
        if key and (
            credentials is None
            or not hmac.compare_digest(credentials.credentials.encode(), key.encode())
        ):
            raise HTTPException(
                status_code=401,
                detail="Invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    app = FastAPI(title="PlugRAG", version="0.1.0")
    app.state.rag = rag

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "answers_enabled": rag.generator is not None}

    def run(payload: SearchRequest, answer: bool) -> dict[str, Any]:
        if answer and rag.generator is None:
            raise HTTPException(
                status_code=501,
                detail="Configure RAG_LLM_MODEL and RAG_LLM_BASE_URL to enable answers",
            )
        try:
            values = payload.model_dump()
            if answer:
                return rag.answer(**values)
            results = rag.search(**values)
            return {"query": payload.query, "results": results, "count": len(results)}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logging.getLogger(__name__).exception("RAG request failed")
            raise HTTPException(
                status_code=503, detail="RAG backend failed; check server logs"
            ) from exc

    @app.post("/search", dependencies=[Depends(authorize)])
    def search(payload: SearchRequest) -> dict[str, Any]:
        return run(payload, False)

    @app.post("/answer", dependencies=[Depends(authorize)])
    def answer(payload: SearchRequest) -> dict[str, Any]:
        return run(payload, True)

    return app
