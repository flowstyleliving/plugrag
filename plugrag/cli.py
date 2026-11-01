"""Console entry point; imports models and server extras only when needed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .core import Document, RAG, Settings


def read_documents(path: Path) -> list[Document]:
    """Accept JSON arrays, JSONL, or a directory of UTF-8 text/Markdown files."""

    if path.is_dir():
        paths = sorted(
            file
            for file in path.rglob("*")
            if file.is_file() and file.suffix.lower() in {".txt", ".md"}
        )
        if not paths:
            raise ValueError("directory contains no .txt or .md files")
        from lens_rag.text import sha256_text

        return [
            Document(
                id=sha256_text(file.relative_to(path).as_posix())[:32],
                title=file.stem,
                text=file.read_text(encoding="utf-8"),
            )
            for file in paths
        ]
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        values = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        values = json.loads(raw)
    if not isinstance(values, list) or any(
        not isinstance(value, dict) for value in values
    ):
        raise ValueError("input must be a JSON array of document objects or JSONL")
    return [Document(**value) for value in values]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Standalone document RAG")
    parser.add_argument(
        "--data-dir", type=Path, help="override RAG_DATA_DIR (default .rag-data)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser(
        "ingest", help="upsert JSON, JSONL, or a text/Markdown directory"
    )
    ingest.add_argument("source", type=Path)
    for name in ("search", "answer"):
        command = commands.add_parser(name)
        command.add_argument("query")
        command.add_argument("--top-k", type=int, default=5)
        command.add_argument("--category")
        command.add_argument("--date-from")
        command.add_argument("--date-to")
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    try:
        from dataclasses import replace

        settings = Settings.from_env()
        if args.data_dir is not None:
            settings = replace(settings, data_dir=args.data_dir)
        rag = RAG(settings)
        if args.command == "ingest":
            result: Any = rag.ingest(read_documents(args.source))
        elif args.command == "serve":
            import uvicorn
            from .generation import generator_from_env
            from .server import create_app

            rag.generator = generator_from_env()
            uvicorn.run(create_app(rag), host=args.host, port=args.port)
            return 0
        else:
            filters = {
                name: getattr(args, name)
                for name in ("top_k", "category", "date_from", "date_to")
            }
            if args.command == "answer":
                from .generation import generator_from_env

                rag.generator = generator_from_env()
                result = rag.answer(args.query, **filters)
            else:
                result = rag.search(args.query, **filters)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, TypeError, OSError, RuntimeError, ImportError) as exc:
        print(f"plugrag: {exc}", file=sys.stderr)
        return 1
