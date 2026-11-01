"""Text extraction, hashing, date parsing, and chunking helpers."""

from __future__ import annotations

import hashlib
import html
import re
from datetime import datetime
from typing import Iterable


BLOCK_TAGS = (
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
)


def html_to_plain_text(raw_html: str | None) -> str:
    """Strip HTML while preserving readable paragraph breaks."""

    if not raw_html:
        return ""

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        text = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", raw_html)
        text = re.sub(r"(?i)</\s*(p|div|li|h[1-6]|blockquote)\s*>", "\n\n", text)
        text = re.sub(r"<[^>]+>", " ", text)
        return normalize_whitespace(html.unescape(text))

    try:
        soup = BeautifulSoup(raw_html, "lxml")
    except Exception:
        soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for tag in soup.find_all(BLOCK_TAGS):
        if tag.name == "br":
            tag.replace_with("\n")
        else:
            tag.insert_before("\n")
            tag.insert_after("\n")

    text = normalize_whitespace(soup.get_text("\n"))
    # Strip WordPress shortcodes like [gallery ids="1,2,3"], [embed], [caption]
    text = re.sub(r'\[[^\]]{1,100}\]', '', text)
    return normalize_whitespace(text)


def normalize_whitespace(text: str) -> str:
    """Normalize text whitespace while keeping paragraph boundaries."""

    text = html.unescape(text).replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]

    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if line:
            current.append(line)
        elif current:
            paragraphs.append(" ".join(current).strip())
            current = []
    if current:
        paragraphs.append(" ".join(current).strip())

    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def sha256_text(text: str) -> str:
    """Return the SHA-256 hash of a UTF-8 text string."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_date_to_int(value: str | None) -> int:
    """Convert a date-like string to YYYYMMDD integer form for filtering."""

    if not value:
        return 0

    date_part = value.split("T", 1)[0].strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return int(datetime.strptime(date_part, fmt).strftime("%Y%m%d"))
        except ValueError:
            continue
    raise ValueError(f"Invalid date value: {value!r}")


def safe_date_int(value: str | None) -> int:
    """Parse a date string to YYYYMMDD int, returning 0 on failure."""
    try:
        return parse_date_to_int(value)
    except ValueError:
        return 0


def document_labeled_field(document: str, label: str) -> str:
    """Return a simple ``Label: value`` field from a stored chunk document."""

    needle = f"{label.strip().lower()}:"
    for line in document.splitlines()[:20]:
        stripped = line.strip()
        if stripped.lower().startswith(needle):
            return stripped.split(":", 1)[1].strip()
    return ""


def split_text_into_chunks(
    text: str,
    chunk_size: int = 200,  # all-MiniLM-L6-v2 max_seq_length=256; margin for header
    overlap: int = 50,
) -> list[str]:
    """Split text into overlapping chunks using tiktoken when available."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        return _split_words(text.split(), chunk_size, overlap)

    tokens = encoding.encode(text)
    if len(tokens) <= chunk_size:
        return [text]

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(tokens), step):
        end = min(start + chunk_size, len(tokens))
        chunk = encoding.decode(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(tokens):
            break
    return chunks


def _split_words(words: list[str], chunk_size: int, overlap: int) -> list[str]:
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
    return chunks


def join_names(values: Iterable[str]) -> str:
    """Join metadata names into a Chroma-compatible scalar string."""

    return "; ".join(str(value).strip() for value in values if str(value).strip())
