"""Rich-powered terminal output for Lens RAG queries.

Matches the web app's design language: dark aesthetic, gold accents,
score bars, highlighted excerpts, category badges.

Auto-detects TTY — falls back to plain text for pipes/scripts.
"""

from __future__ import annotations

import re
import sys
from typing import Any

from .query_engine import ArticleResult, ChunkMatch, QueryEngine

try:
    from rich import box
    from rich.color import Color
    from rich.console import Console, Group
    from rich.highlighter import RegexHighlighter
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.style import Style
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Web app palette
GOLD = "#d6aa4c"
CYAN = "#74b9c7"
RED = "#e06c75"
BG = "#0b0d10"
PANEL_BG = "#14181d"
MUTED = "#a8b0ba"
TEXT_COLOR = "#f3f1ea"


def is_tty() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def format_results(results: list[ArticleResult], query: str = "") -> str:
    """Format results — Rich if TTY, plain text otherwise."""
    if RICH_AVAILABLE and is_tty():
        return _format_rich(results, query)
    from .query_engine import format_results_text as _text

    return _text(results)


def run_interactive_rich(engine: QueryEngine, top_k: int = 10) -> int:
    """Rich-powered interactive REPL."""
    if not RICH_AVAILABLE:
        print("Rich not installed. Install with: pip install rich", file=sys.stderr)
        return 1

    console = Console()
    console.print()
    console.print(
        Panel(
            "[bold]Lens RAG[/] — Search 13 years of New Orleans investigative reporting",
            border_style=Style(color=GOLD),
            box=box.HEAVY,
        )
    )
    console.print(
        "  Type a search query, or "
        "[dim]exit[/dim] / [dim]Ctrl-D[/dim] to quit.\n"
    )

    import readline  # noqa: F401 — enables line editing

    while True:
        try:
            query = input("  Search > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n")
            return 0
        if query.lower() in {"exit", "quit", "q"}:
            return 0
        if not query:
            continue
        try:
            results = engine.search(query, top_k=top_k)
        except (ImportError, ValueError) as exc:
            console.print(f"[{RED}]Query failed:[/] {exc}")
            continue
        console.print(_render_results(results, query))
        console.print()


def _format_rich(results: list[ArticleResult], query: str = "") -> str:
    """Render results to a string via Rich console capture."""
    from io import StringIO

    console = Console(file=StringIO(), force_terminal=True, width=100)
    console.print(_render_results(results, query))
    return console.file.getvalue().rstrip()  # type: ignore[union-attr]


def _render_results(
    results: list[ArticleResult], query: str = ""
) -> Panel | Rule | Group:
    if not results:
        return Panel(
            "No matching articles found. Try fewer words or remove filters.",
            border_style=Style(color=MUTED),
            box=box.ROUNDED,
        )

    rendered: list[Any] = []
    for i, result in enumerate(results):
        rendered.append(_render_card(result, query, index=i + 1))
        if i < len(results) - 1:
            rendered.append("")

    return Group(*rendered)


def _render_card(
    result: ArticleResult, query: str, index: int
) -> Panel:
    """Render one article result as a Rich panel card."""
    # Title with score
    score_pct = round(result.score * 100)
    score_color = GOLD if score_pct >= 70 else (CYAN if score_pct >= 40 else MUTED)
    title = Text.assemble(
        (f"{index}. ", Style(color=MUTED)),
        (result.title, Style(bold=True, color=TEXT_COLOR)),
    )

    # Metadata line: date · categories · score bar
    meta_parts: list[Text] = []
    date_str = result.date.split("T")[0] if result.date else ""
    if date_str:
        meta_parts.append(Text(date_str, style=Style(color=MUTED, italic=True)))

    # Category badges
    cats = [c.strip() for c in result.categories.split(";") if c.strip()]
    for cat in cats[:4]:
        meta_parts.append(
            Text(f" {cat} ", style=Style(color=TEXT_COLOR, bgcolor="#1a2028"))
        )

    # Score bar
    bar_width = 10
    filled = round(score_pct / 100 * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)
    meta_parts.append(Text(f" {bar} {score_pct}%", style=Style(color=score_color)))

    # Assemble metadata
    meta = Text()
    for part in meta_parts:
        if meta.cell_len:
            meta.append("  ")
        meta.append(part)

    # Match excerpt with query-term highlighting
    match_excerpt = ""
    if result.matches:
        match_excerpt = result.matches[0].excerpt[:400]

    excerpt_text = _highlight_terms(match_excerpt, query) if match_excerpt else Text("")

    # URL
    url_text = Text(result.url or "", style=Style(color=MUTED, dim=True))
    url_text.truncate(85, overflow="ellipsis")

    # Full content assembly
    content = Group(title, meta, Text(), excerpt_text, Text(), url_text)

    return Panel(
        content,
        border_style=Style(color="#2a3038"),
        box=box.ROUNDED,
        padding=(1, 2),
    )


def _highlight_terms(text: str, query: str) -> Text:
    """Highlight query terms in excerpt text with gold background."""
    rich_text = Text(text, style=Style(color="#d9dde2"))

    if not query:
        return rich_text

    # Extract meaningful words from query
    terms = [w for w in re.findall(r"\w{3,}", query.lower())]
    if not terms:
        return rich_text

    pattern = re.compile(
        "(" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + ")",
        re.IGNORECASE,
    )
    rich_text.highlight_regex(
        pattern,
        style=Style(bgcolor=Color.from_rgb(214, 170, 76), color=TEXT_COLOR),
    )
    return rich_text
