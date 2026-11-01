"""End-to-end orchestration for scraping, embedding, and verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import (
    CHROMA_DIRNAME,
    DEFAULT_DATA_DIR,
    DEFAULT_RATE_LIMIT_SECONDS,
    MANIFEST_FILENAME,
)
from .embedder import ArticleEmbedder, get_chroma_collection, safe_collection_count
from .manifest import Manifest
from .scraper import LensAPIError, LensScraper


def run_pipeline(
    data_dir: Path = DEFAULT_DATA_DIR,
    scrape: bool = True,
    embed: bool = True,
    limit: int | None = None,
    force: bool = False,
    rate_limit_seconds: float = DEFAULT_RATE_LIMIT_SECONDS,
) -> dict[str, Any]:
    """Run scrape and/or embedding stages and return summary statistics."""

    data_dir = Path(data_dir)
    summary: dict[str, Any] = {"data_dir": str(data_dir)}

    if scrape:
        scraper = LensScraper(
            data_dir=data_dir,
            rate_limit_seconds=rate_limit_seconds,
        )
        summary["scrape"] = scraper.scrape_all(limit=limit, force=force)

    if embed:
        embedder = ArticleEmbedder(data_dir=data_dir)
        summary["embed"] = embedder.embed_all(limit=limit, force=force)

    summary["verify"] = verify_pipeline(data_dir)
    return summary


def verify_pipeline(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, int]:
    """Return basic manifest and ChromaDB consistency statistics, logging warnings."""

    import logging

    logger = logging.getLogger(__name__)
    data_dir = Path(data_dir)
    manifest = Manifest.load(data_dir / MANIFEST_FILENAME)
    db_size = 0
    try:
        collection = get_chroma_collection(data_dir / CHROMA_DIRNAME)
        db_size = safe_collection_count(collection)
    except ImportError:
        db_size = 0

    embedded_count = manifest.embedded_count()
    # Emit warning if ChromaDB has far fewer chunks than expected
    if embedded_count > 0 and db_size < embedded_count:
        logger.warning(
            f"ChromaDB has {db_size} chunks but manifest says "
            f"{embedded_count} articles embedded. "
            "Possible data loss — re-run with --force."
        )

    return {
        "downloaded_articles": manifest.downloaded_count(),
        "embedded_articles": embedded_count,
        "vector_chunks": db_size,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the pipeline CLI argument parser."""

    parser = argparse.ArgumentParser(description="Run The Lens RAG pipeline")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Run scrape then embed")
    mode.add_argument("--scrape-only", action="store_true", help="Only scrape articles")
    mode.add_argument("--embed-only", action="store_true", help="Only embed articles")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rate-limit", type=float, default=DEFAULT_RATE_LIMIT_SECONDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline CLI."""

    args = build_arg_parser().parse_args(argv)
    scrape = True
    embed = True
    if args.scrape_only:
        embed = False
    elif args.embed_only:
        scrape = False

    try:
        summary = run_pipeline(
            data_dir=args.data_dir,
            scrape=scrape,
            embed=embed,
            limit=args.limit,
            force=args.force,
            rate_limit_seconds=args.rate_limit,
        )
    except (ImportError, LensAPIError, ValueError) as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
