"""Deterministic archive-native topic lexicon generation and expansion."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import ARTICLES_DIRNAME, DEFAULT_DATA_DIR
from .embedder import load_article
from .nl_query import STOPWORDS, keywords_from_query

TOPIC_LEXICON_FILENAME = "topic_lexicon.json"
TOPIC_LEXICON_VERSION = 1
MAX_TOPICS = 200
MAX_TERMS_PER_TOPIC = 24
MAX_MATCHED_TOPICS = 2
MAX_EXPANSION_TERMS = 6
MIN_SEED_COUNT = 2

GENERIC_TERMS = {
    "article",
    "articles",
    "behind",
    "blog",
    "city",
    "coverage",
    "government",
    "http",
    "index",
    "lens",
    "live",
    "more",
    "new",
    "news",
    "nola",
    "orleans",
    "politics",
    "report",
    "reported",
    "reporting",
    "reports",
    "read",
    "story",
    "stories",
    "staff",
    "top",
    "thelensnola",
    "uncategorized",
    "writer",
    "www",
}
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)


@dataclass(frozen=True)
class TopicRecord:
    """One generated archive topic."""

    label: str
    terms: list[str]
    article_ids: list[str] = field(default_factory=list)
    count: int = 0
    source: str = "generated"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TopicRecord":
        return cls(
            label=str(payload.get("label", "")).strip(),
            terms=_string_list(payload.get("terms")),
            article_ids=_string_list(payload.get("article_ids")),
            count=int(payload.get("count") or 0),
            source=str(payload.get("source") or "generated"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "terms": self.terms,
            "article_ids": self.article_ids,
            "count": self.count,
            "source": self.source,
        }


@dataclass(frozen=True)
class TopicExpansion:
    """Topic expansion terms selected for one query."""

    terms: list[str] = field(default_factory=list)
    matched_topics: list[str] = field(default_factory=list)
    confidence: float = 0.0


class TopicLexicon:
    """Generated topic records that can expand lexical search terms."""

    def __init__(self, topics: list[TopicRecord] | None = None) -> None:
        self.topics = [topic for topic in topics or [] if topic.label and topic.terms]

    @classmethod
    def load(cls, path: Path) -> "TopicLexicon":
        """Load a topic lexicon, returning an empty lexicon on missing/bad files."""

        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            topics = payload.get("topics", [])
            if not isinstance(topics, list):
                return cls()
            return cls(
                [
                    TopicRecord.from_dict(item)
                    for item in topics
                    if isinstance(item, dict)
                ]
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return cls()

    def expand(
        self,
        query: str,
        terms: list[str] | tuple[str, ...],
    ) -> TopicExpansion:
        """Return archive-native expansion terms for a query."""

        if not self.topics:
            return TopicExpansion()
        query_tokens = set(_meaningful_tokens([query, *terms]))
        if not query_tokens:
            return TopicExpansion()
        is_single_word = len(query_tokens) == 1
        scored: list[tuple[float, TopicRecord, set[str]]] = []
        for topic in self.topics:
            label_tokens = set(_meaningful_tokens([topic.label]))
            term_tokens = set(_meaningful_tokens(topic.terms))
            all_tokens = label_tokens | term_tokens
            overlap = query_tokens & all_tokens
            label_overlap = query_tokens & label_tokens
            phrase_match = _has_phrase_match(query, terms, topic)
            if is_single_word:
                if not _single_word_exact_match(next(iter(query_tokens)), topic):
                    continue
            elif not phrase_match:
                short_label_match = len(label_overlap) == 1 and len(query_tokens) <= 3
                if len(label_overlap) < 2 and not short_label_match:
                    continue
            score = len(overlap) + (2.0 * len(label_overlap)) + (2.0 if phrase_match else 0.0)
            score += min(1.0, topic.count / 25.0)
            scored.append((score, topic, overlap))
        if not scored:
            return TopicExpansion()

        scored.sort(key=lambda item: (-item[0], item[1].label.lower()))
        selected = scored[:MAX_MATCHED_TOPICS]
        expansion_terms: list[str] = []
        seen = set(query_tokens)
        for _, topic, _ in selected:
            for term in topic.terms:
                normalized = normalize_term(term)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                expansion_terms.append(normalized)
                if len(expansion_terms) >= MAX_EXPANSION_TERMS:
                    break
            if len(expansion_terms) >= MAX_EXPANSION_TERMS:
                break
        confidence = round(min(1.0, selected[0][0] / 6.0), 3)
        return TopicExpansion(
            terms=expansion_terms,
            matched_topics=[topic.label for _, topic, _ in selected],
            confidence=confidence,
        )


def build_topic_lexicon(
    data_dir: Path = DEFAULT_DATA_DIR,
    article_vectors: dict[str, list[float]] | None = None,
    max_topics: int = MAX_TOPICS,
) -> TopicLexicon:
    """Build a deterministic topic lexicon from local article JSON files."""

    articles = _load_articles(Path(data_dir) / ARTICLES_DIRNAME)
    records = build_topic_records(
        articles,
        article_vectors=article_vectors,
        max_topics=max_topics,
    )
    return TopicLexicon(records)


def build_topic_records(
    articles: list[dict[str, Any]],
    article_vectors: dict[str, list[float]] | None = None,
    max_topics: int = MAX_TOPICS,
) -> list[TopicRecord]:
    """Build topic records from normalized article dictionaries."""

    seeds = _seed_articles(articles)
    article_by_id = {str(article.get("id", "")): article for article in articles}
    records: list[TopicRecord] = []
    for seed, article_ids in seeds.items():
        if len(article_ids) < MIN_SEED_COUNT:
            continue
        source = _seed_source(seed)
        label = _seed_label(seed)
        related_ids = _related_article_ids(seed, article_ids, article_vectors)
        all_ids = _stable_article_ids([*article_ids, *related_ids])
        selected_articles = [article_by_id[item] for item in all_ids if item in article_by_id]
        terms = _topic_terms(label, selected_articles)
        if not terms:
            continue
        records.append(
            TopicRecord(
                label=label,
                terms=terms,
                article_ids=all_ids[:40],
                count=len(article_ids),
                source=source,
            )
        )
    records.sort(key=lambda item: (-item.count, item.label.lower()))
    return records[:max_topics]


def write_topic_lexicon(path: Path, lexicon: TopicLexicon) -> None:
    """Write a generated topic lexicon atomically."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": TOPIC_LEXICON_VERSION,
        "topics": [topic.to_dict() for topic in lexicon.topics],
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def normalize_term(value: str) -> str:
    tokens = _meaningful_tokens([value])
    return " ".join(tokens)


def _load_articles(articles_dir: Path) -> list[dict[str, Any]]:
    if not articles_dir.exists():
        return []
    articles: list[dict[str, Any]] = []
    for path in sorted(articles_dir.glob("*.json")):
        try:
            article = load_article(path)
        except Exception:
            continue
        if article.get("id") is not None:
            articles.append(article)
    return articles


def _seed_articles(articles: list[dict[str, Any]]) -> dict[str, list[str]]:
    seeds: dict[str, list[str]] = defaultdict(list)
    phrase_counts: Counter[str] = Counter()
    article_phrases: dict[str, list[str]] = {}
    for article in articles:
        article_id = str(article.get("id", ""))
        categories = _string_list(article.get("category_names"))
        tags = _string_list(article.get("tag_names"))
        for category in categories:
            _add_seed(seeds, f"category:{category}", article_id)
        for tag in tags:
            _add_seed(seeds, f"tag:{tag}", article_id)
        for category in categories:
            for tag in tags:
                _add_seed(seeds, f"pair:{category} / {tag}", article_id)
        phrases = _article_phrases(article)
        article_phrases[article_id] = phrases
        phrase_counts.update(set(phrases))

    for article_id, phrases in article_phrases.items():
        for phrase in phrases:
            if phrase_counts[phrase] >= MIN_SEED_COUNT:
                _add_seed(seeds, f"phrase:{phrase}", article_id)
    return seeds


def _add_seed(seeds: dict[str, list[str]], key: str, article_id: str) -> None:
    label = _seed_label(key)
    if article_id and normalize_term(label):
        seeds[key].append(article_id)


def _seed_label(seed: str) -> str:
    return seed.split(":", 1)[1] if ":" in seed else seed


def _seed_source(seed: str) -> str:
    return seed.split(":", 1)[0] if ":" in seed else "generated"


def _article_phrases(article: dict[str, Any]) -> list[str]:
    text = " ".join(
        [
            str(article.get("title") or ""),
            str(article.get("excerpt") or ""),
        ]
    )
    tokens = _meaningful_tokens([text])
    phrases: list[str] = []
    for size in (2, 3):
        for index in range(0, max(0, len(tokens) - size + 1)):
            phrase = " ".join(tokens[index : index + size])
            if phrase and not all(token in STOPWORDS for token in phrase.split()):
                phrases.append(phrase)
    return phrases[:20]


def _topic_terms(label: str, articles: list[dict[str, Any]]) -> list[str]:
    counter: Counter[str] = Counter()
    label_term = normalize_term(label)
    if label_term:
        counter[label_term] += 1000
    for article in articles:
        for value in _string_list(article.get("tag_names")):
            term = normalize_term(value)
            if term:
                counter[term] += 12
        for value in _string_list(article.get("category_names")):
            term = normalize_term(value)
            if term:
                counter[term] += 8
        for phrase in _article_phrases(article):
            term = normalize_term(phrase)
            if term:
                counter[term] += 4
        title_terms = keywords_from_query(str(article.get("title") or ""), limit=8)
        for term in title_terms:
            normalized = normalize_term(term)
            if normalized:
                counter[normalized] += 2
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    terms: list[str] = []
    for term, _ in ordered:
        if term and term not in terms:
            terms.append(term)
        if len(terms) >= MAX_TERMS_PER_TOPIC:
            break
    return terms


def _related_article_ids(
    seed: str,
    article_ids: list[str],
    article_vectors: dict[str, list[float]] | None,
    limit: int = 5,
) -> list[str]:
    if not article_vectors:
        return []
    centroid = _centroid(
        [article_vectors[item] for item in article_ids if item in article_vectors]
    )
    if not centroid:
        return []
    seed_ids = set(article_ids)
    scored: list[tuple[float, str]] = []
    for article_id, vector in article_vectors.items():
        if article_id in seed_ids:
            continue
        score = _cosine(centroid, vector)
        if score >= 0.65:
            scored.append((score, article_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [article_id for _, article_id in scored[:limit]]


def _centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = min(len(vector) for vector in vectors)
    if dim <= 0:
        return []
    centroid = [0.0] * dim
    for vector in vectors:
        for index in range(dim):
            centroid[index] += float(vector[index])
    scale = 1.0 / len(vectors)
    return [value * scale for value in centroid]


def _cosine(left: list[float], right: list[float]) -> float:
    dim = min(len(left), len(right))
    if dim <= 0:
        return 0.0
    dot = sum(float(left[index]) * float(right[index]) for index in range(dim))
    left_norm = math.sqrt(sum(float(left[index]) ** 2 for index in range(dim)))
    right_norm = math.sqrt(sum(float(right[index]) ** 2 for index in range(dim)))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _stable_article_ids(article_ids: Iterable[str]) -> list[str]:
    return sorted(set(article_ids), key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))


def _meaningful_tokens(values: Iterable[str]) -> list[str]:
    tokens: list[str] = []
    for value in values:
        for token in TOKEN_RE.findall(str(value).lower()):
            if len(token) <= 2 or token in STOPWORDS or token in GENERIC_TERMS:
                continue
            tokens.append(token)
    return tokens


def _has_phrase_match(
    query: str,
    terms: list[str] | tuple[str, ...],
    topic: TopicRecord,
) -> bool:
    needles = [
        term
        for term in [normalize_term(query), *(normalize_term(term) for term in terms)]
        if len(term.split()) > 1
    ]
    haystack = [normalize_term(topic.label), *(normalize_term(term) for term in topic.terms)]
    return any(needle and needle in haystack for needle in needles)


def _single_word_exact_match(token: str, topic: TopicRecord) -> bool:
    return normalize_term(topic.label) == token


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in value.split(";") if part.strip()]
    return []


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build archive-native topic lexicon")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-topics", type=int, default=MAX_TOPICS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.max_topics <= 0:
        print("max-topics must be positive", file=sys.stderr)
        return 1
    lexicon = build_topic_lexicon(args.data_dir, max_topics=args.max_topics)
    output = args.output or args.data_dir / TOPIC_LEXICON_FILENAME
    write_topic_lexicon(output, lexicon)
    print(f"Wrote {len(lexicon.topics)} topics to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
