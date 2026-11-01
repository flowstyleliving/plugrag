"""Persistent, no-credit entity graph and community retrieval for Lens reporting."""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from .config import ARTICLES_DIRNAME, DEFAULT_DATA_DIR
from .embedder import load_article


GRAPH_FILENAME = "knowledge_graph.sqlite3"
GRAPH_SCHEMA_VERSION = 1
MAX_STRUCTURED_ENTITIES = 12
MAX_NAMED_ENTITIES = 10
MAX_GRAPH_ENTITIES_PER_ARTICLE = 18
MIN_NAMED_ENTITY_ARTICLES = 2
MIN_COMMUNITY_EDGE_WEIGHT = 2

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9&.-]{1,10}\b")
CASE_RE = re.compile(
    r"\b([A-Z][\w'’.-]+(?:\s+[A-Z][\w'’.-]+){0,4})\s+v\.?\s+"
    r"([A-Z][\w'’.-]+(?:\s+[A-Z][\w'’.-]+){0,4})\b"
)
PROPER_RE = re.compile(
    r"\b(?:[A-Z][a-z][A-Za-z'’.-]*|[A-Z]{2,10})"
    r"(?:\s+(?:of|the|and|for|de|la|[A-Z][a-z][A-Za-z'’.-]*|[A-Z]{2,10})){1,6}\b"
)
LONG_ACRONYM_RE = re.compile(
    r"\b((?:[A-Z][A-Za-z'’.-]*\s+){1,7}[A-Z][A-Za-z'’.-]*)\s*"
    r"\(([A-Z][A-Z0-9]{1,9})\)"
)

GENERIC_ENTITY_TOKENS = {
    "a",
    "an",
    "and",
    "article",
    "at",
    "by",
    "for",
    "from",
    "in",
    "lens",
    "new",
    "news",
    "of",
    "on",
    "or",
    "report",
    "reported",
    "reporting",
    "said",
    "story",
    "the",
    "this",
    "to",
    "with",
}
GENERIC_ENTITY_NAMES = {
    "according to",
    "city council",
    "download subscribe",
    "new orleans",
    "the city",
    "the lens",
    "united states",
}
GENERIC_ACRONYMS = {"APPROVED", "DEFERRED", "NO", "OK", "RIP", "UPDATE", "YES"}
GENERIC_RELATION_NODES = {
    "behind the lens",
    "ceo",
    "charter schools",
    "criminal justice",
    "download subscribe",
    "environment",
    "government politics",
    "investigations",
    "land use",
    "news",
    "opinion",
    "rss",
    "schools",
    "top story",
}
ORGANIZATION_WORDS = {
    "administration",
    "agency",
    "association",
    "authority",
    "board",
    "bureau",
    "center",
    "commission",
    "committee",
    "company",
    "council",
    "court",
    "department",
    "district",
    "foundation",
    "group",
    "institute",
    "office",
    "parish",
    "school",
    "sheriff",
    "university",
}
FACILITY_WORDS = {"camp", "center", "facility", "jail", "penitentiary", "prison"}
TYPE_PRIORITY = {
    "author": 7,
    "legal_case": 6,
    "facility": 5,
    "organization": 4,
    "person": 3,
    "tag": 2,
    "category": 1,
    "named": 0,
}

KNOWN_ALIASES = {
    "nopd": "New Orleans Police Department",
    "ojc": "Orleans Justice Center",
    "opp": "Orleans Parish Prison",
    "opso": "Orleans Parish Sheriff's Office",
    "s&wb": "Sewerage and Water Board of New Orleans",
    "swbno": "Sewerage and Water Board of New Orleans",
}


@dataclass(frozen=True)
class GraphSearchHit:
    """One article reached through an entity or community path."""

    article_id: str
    score: float
    matched_entities: tuple[str, ...] = ()
    related_entities: tuple[str, ...] = ()
    communities: tuple[str, ...] = ()


@dataclass
class GraphSearchResult:
    """Ranked graph hits plus an inspectable traversal explanation."""

    hits: list[GraphSearchHit] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)
    related_entities: list[str] = field(default_factory=list)
    communities: list[dict[str, Any]] = field(default_factory=list)
    enabled: bool = False

    def diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "matched_entities": self.matched_entities,
            "related_entities": self.related_entities,
            "communities": self.communities,
            "candidate_count": len(self.hits),
        }


class KnowledgeGraphIndex:
    """Read-only traversal over the generated SQLite graph."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @property
    def available(self) -> bool:
        return self.path.exists()

    def metadata(self) -> dict[str, Any]:
        if not self.available:
            return {"available": False}
        try:
            with self._connect() as connection:
                rows = connection.execute("SELECT key, value FROM metadata").fetchall()
            payload = {str(row[0]): _json_value(str(row[1])) for row in rows}
            payload["available"] = True
            return payload
        except sqlite3.Error:
            return {"available": False}

    def search(self, query: str, limit: int = 12) -> GraphSearchResult:
        """Match entities, traverse weighted neighbors, then rank related articles."""

        tokens = _query_tokens(query)
        if not self.available:
            return GraphSearchResult(enabled=False)
        if not tokens:
            return GraphSearchResult(enabled=True)
        with self._connect() as connection:
            seeds = _matched_entities(connection, query, tokens)
            if not seeds:
                return GraphSearchResult(enabled=True)
            entity_scores = {row["entity_id"]: row["score"] for row in seeds}
            seed_ids = set(entity_scores)
            neighbor_rows = _neighbor_entities(connection, entity_scores)
            for row in neighbor_rows:
                entity_scores[row["entity_id"]] = max(
                    entity_scores.get(row["entity_id"], 0.0), row["score"]
                )

            community_rows = _matched_communities(
                connection,
                entity_scores,
                limit=4,
            )
            article_scores: dict[str, float] = defaultdict(float)
            article_entities: dict[str, list[tuple[int, str, bool]]] = defaultdict(list)
            entity_ids = list(entity_scores)[:48]
            placeholders = ",".join("?" for _ in entity_ids)
            rows = connection.execute(
                f"""
                SELECT ae.article_id, ae.entity_id, ae.mentions, e.canonical_name
                FROM article_entities ae
                JOIN entities e ON e.entity_id = ae.entity_id
                WHERE ae.entity_id IN ({placeholders})
                """,
                entity_ids,
            ).fetchall()
            for row in rows:
                article_id = str(row["article_id"])
                entity_id = int(row["entity_id"])
                direct = entity_id in seed_ids
                multiplier = 2.2 if direct else 0.8
                article_scores[article_id] += (
                    entity_scores[entity_id]
                    * multiplier
                    * (1.0 + 0.12 * math.log1p(int(row["mentions"])))
                )
                article_entities[article_id].append(
                    (entity_id, str(row["canonical_name"]), direct)
                )

            community_labels: dict[int, str] = {}
            for community in community_rows:
                community_id = int(community["community_id"])
                community_labels[community_id] = str(community["label"])
                related = connection.execute(
                    """
                    SELECT article_id, entity_count
                    FROM article_communities
                    WHERE community_id = ?
                    ORDER BY entity_count DESC, article_id DESC
                    LIMIT 20
                    """,
                    (community_id,),
                ).fetchall()
                for row in related:
                    article_scores[str(row["article_id"])] += (
                        float(community["score"])
                        * 0.18
                        * math.log1p(int(row["entity_count"]))
                    )

            ranked = sorted(
                article_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )[: max(1, limit)]
            hits: list[GraphSearchHit] = []
            for article_id, score in ranked:
                entities = article_entities.get(article_id, [])
                matched = tuple(
                    name for _, name, direct in entities if direct
                )[:6]
                related = tuple(
                    name for _, name, direct in entities if not direct
                )[:6]
                community_names = tuple(
                    str(row["label"])
                    for row in community_rows
                    if _article_in_community(
                        connection,
                        article_id,
                        int(row["community_id"]),
                    )
                )[:3]
                hits.append(
                    GraphSearchHit(
                        article_id=article_id,
                        score=round(score, 6),
                        matched_entities=matched,
                        related_entities=related,
                        communities=community_names,
                    )
                )

        return GraphSearchResult(
            hits=hits,
            matched_entities=[str(row["canonical_name"]) for row in seeds],
            related_entities=[str(row["canonical_name"]) for row in neighbor_rows[:12]],
            communities=[
                {
                    "label": str(row["label"]),
                    "summary": str(row["summary"]),
                    "score": round(float(row["score"]), 3),
                }
                for row in community_rows
            ],
            enabled=True,
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection


def build_knowledge_graph(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Build the local graph, Louvain communities, and extractive summaries."""

    data_dir = Path(data_dir)
    target = Path(output_path or data_dir / GRAPH_FILENAME)
    articles = _load_articles(data_dir / ARTICLES_DIRNAME)
    article_entities, names, entity_types, aliases = _extract_archive_entities(articles)
    document_counts = Counter(
        entity
        for entities in article_entities.values()
        for entity in entities
    )
    retained = {
        entity
        for entity, count in document_counts.items()
        if count >= MIN_NAMED_ENTITY_ARTICLES
        or entity_types.get(entity) in {
            "tag",
            "category",
            "author",
            "facility",
            "legal_case",
        }
        or entity in {normalize_entity(value) for value in KNOWN_ALIASES.values()}
    }
    article_entities = {
        article_id: {
            entity: mentions
            for entity, mentions in entities.items()
            if entity in retained
        }
        for article_id, entities in article_entities.items()
    }
    entity_norms = sorted(retained)
    entity_ids = {entity: index for index, entity in enumerate(entity_norms, start=1)}
    edges = _build_edges(article_entities, entity_ids, articles)
    communities = _louvain_communities(entity_ids, edges, document_counts)
    community_records, article_communities = _community_records(
        articles,
        article_entities,
        entity_ids,
        names,
        entity_types,
        document_counts,
        communities,
    )
    _write_graph(
        target,
        articles,
        article_entities,
        entity_ids,
        names,
        entity_types,
        document_counts,
        aliases,
        edges,
        communities,
        community_records,
        article_communities,
    )
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "article_count": len(articles),
        "entity_count": len(entity_ids),
        "edge_count": len(edges),
        "community_count": len(community_records),
        "article_community_count": sum(len(value) for value in article_communities.values()),
        "path": str(target),
    }


def _extract_archive_entities(
    articles: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, int]],
    dict[str, str],
    dict[str, str],
    dict[str, set[str]],
]:
    article_entities: dict[str, dict[str, int]] = {}
    display_counts: dict[str, Counter[str]] = defaultdict(Counter)
    type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    aliases: dict[str, set[str]] = defaultdict(set)
    for article in articles:
        article_id = str(article.get("id") or "").strip()
        if not article_id:
            continue
        found: Counter[str] = Counter()
        structured: list[tuple[str, str]] = []
        structured.extend((str(value), "tag") for value in article.get("tag_names") or [])
        structured.extend(
            (str(value), "category") for value in article.get("category_names") or []
        )
        author = str(article.get("author_name") or article.get("author") or "").strip()
        if author:
            structured.append((author, "author"))
        for display, entity_type in structured[:MAX_STRUCTURED_ENTITIES]:
            _record_entity(display, entity_type, found, display_counts, type_counts)

        title = str(article.get("title") or "")
        body = str(article.get("plain_text") or "")[:24_000]
        text = f"{title}. {body}"
        named: Counter[tuple[str, str]] = Counter()
        for left, right in CASE_RE.findall(text):
            display = f"{left} v. {right}"
            normalized = normalize_entity(display)
            if normalized:
                named[(display, "legal_case")] += 5
        for display in PROPER_RE.findall(text):
            normalized = normalize_entity(display)
            if _valid_named_entity(display, normalized):
                entity_type = (
                    "author"
                    if re.match(r"^By\s+", display)
                    else _classify_entity(display)
                )
                named[(display, entity_type)] += 1
        for display in ACRONYM_RE.findall(text):
            normalized = normalize_entity(display)
            if (
                normalized
                and normalized not in GENERIC_ENTITY_TOKENS
                and display not in GENERIC_ACRONYMS
                and not re.fullmatch(r"[IVXLCDM]+", display)
            ):
                named[(display, "organization")] += 1
        for long_name, acronym in LONG_ACRONYM_RE.findall(text):
            long_norm = normalize_entity(long_name)
            acronym_norm = normalize_entity(acronym)
            if _valid_named_entity(long_name, long_norm) and acronym_norm:
                named[(long_name, _classify_entity(long_name))] += 4
                aliases[long_norm].add(acronym)
        for alias, canonical in KNOWN_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", text, flags=re.IGNORECASE):
                named[(canonical, _classify_entity(canonical))] += 3
                aliases[normalize_entity(canonical)].add(alias)

        title_norm = normalize_entity(title)
        ranked_named = sorted(
            named.items(),
            key=lambda item: (
                -(item[1] + (3 if normalize_entity(item[0][0]) in title_norm else 0)),
                normalize_entity(item[0][0]),
            ),
        )[:MAX_NAMED_ENTITIES]
        for (display, entity_type), mentions in ranked_named:
            normalized = _record_entity(
                display,
                entity_type,
                found,
                display_counts,
                type_counts,
                mentions=mentions,
            )
            if normalized:
                aliases[normalized].add(display)
        article_entities[article_id] = dict(found)

    names = {
        normalized: counts.most_common(1)[0][0]
        for normalized, counts in display_counts.items()
    }
    entity_types = {
        normalized: max(
            counts,
            key=lambda value: (TYPE_PRIORITY.get(value, 0), counts[value]),
        )
        for normalized, counts in type_counts.items()
    }
    return article_entities, names, entity_types, aliases


def _record_entity(
    display: str,
    entity_type: str,
    found: Counter[str],
    display_counts: dict[str, Counter[str]],
    type_counts: dict[str, Counter[str]],
    mentions: int = 1,
) -> str:
    cleaned = " ".join(str(display).split()).strip(" .,:;–—-")
    cleaned = re.sub(r"^(?:By|From)\s+", "", cleaned)
    normalized = normalize_entity(cleaned)
    if (
        not normalized
        or len(normalized) < 2
        or cleaned.upper() in GENERIC_ACRONYMS
        or re.fullmatch(r"[IVXLCDM]+", cleaned)
    ):
        return ""
    found[normalized] += max(1, mentions)
    display_counts[normalized][cleaned] += max(1, mentions)
    type_counts[normalized][entity_type] += 1
    return normalized


def normalize_entity(value: str) -> str:
    text = str(value or "").replace("’", "'").lower().strip()
    text = re.sub(r"^the\s+", "", text)
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"[^a-z0-9&]+", " ", text)
    return " ".join(text.split())


def _valid_named_entity(display: str, normalized: str) -> bool:
    if not normalized or normalized in GENERIC_ENTITY_NAMES or len(normalized) < 3:
        return False
    tokens = normalized.split()
    if len(tokens) < 2 or len(tokens) > 7:
        return False
    meaningful = [token for token in tokens if token not in GENERIC_ENTITY_TOKENS]
    if len(meaningful) < 2:
        return False
    return not all(token.isdigit() for token in meaningful)


def _classify_entity(display: str) -> str:
    normalized = normalize_entity(display)
    tokens = set(normalized.split())
    if " v " in f" {normalized} ":
        return "legal_case"
    if tokens & FACILITY_WORDS:
        return "facility"
    if tokens & ORGANIZATION_WORDS or re.fullmatch(r"[A-Z][A-Z0-9&.-]{1,10}", display):
        return "organization"
    if 2 <= len(tokens) <= 4:
        return "person"
    return "named"


def _build_edges(
    article_entities: dict[str, dict[str, int]],
    entity_ids: dict[str, int],
    articles: list[dict[str, Any]],
) -> dict[tuple[int, int], tuple[int, str, str]]:
    article_dates = {
        str(article.get("id") or ""): str(article.get("date") or "")[:10]
        for article in articles
    }
    weights: Counter[tuple[int, int]] = Counter()
    first_dates: dict[tuple[int, int], str] = {}
    last_dates: dict[tuple[int, int], str] = {}
    for article_id, entities in article_entities.items():
        ranked = sorted(
            entities,
            key=lambda value: (-entities[value], value),
        )[:MAX_GRAPH_ENTITIES_PER_ARTICLE]
        date = article_dates.get(article_id, "")
        for left, right in combinations(ranked, 2):
            edge = tuple(sorted((entity_ids[left], entity_ids[right])))
            weights[edge] += 1
            if date:
                first_dates[edge] = min(first_dates.get(edge, date), date)
                last_dates[edge] = max(last_dates.get(edge, date), date)
    return {
        edge: (weight, first_dates.get(edge, ""), last_dates.get(edge, ""))
        for edge, weight in weights.items()
    }


def _louvain_communities(
    entity_ids: dict[str, int],
    edges: dict[tuple[int, int], tuple[int, str, str]],
    document_counts: Counter[str],
) -> dict[int, int]:
    import networkx as nx

    graph = nx.Graph()
    eligible = {
        entity_ids[normalized]
        for normalized, count in document_counts.items()
        if normalized in entity_ids and count >= MIN_NAMED_ENTITY_ARTICLES
    }
    graph.add_nodes_from(eligible)
    for (left, right), (weight, _, _) in edges.items():
        if weight >= MIN_COMMUNITY_EDGE_WEIGHT and left in eligible and right in eligible:
            graph.add_edge(left, right, weight=weight)
    if graph.number_of_edges() == 0:
        return {}
    partitions = nx.community.louvain_communities(
        graph,
        weight="weight",
        resolution=1.05,
        seed=42,
    )
    ordered = sorted(partitions, key=lambda group: (-len(group), min(group)))
    return {
        entity_id: community_id
        for community_id, group in enumerate(ordered, start=1)
        for entity_id in sorted(group)
        if len(group) >= 2
    }


def _community_records(
    articles: list[dict[str, Any]],
    article_entities: dict[str, dict[str, int]],
    entity_ids: dict[str, int],
    names: dict[str, str],
    entity_types: dict[str, str],
    document_counts: Counter[str],
    communities: dict[int, int],
) -> tuple[dict[int, dict[str, Any]], dict[str, dict[int, int]]]:
    id_to_normalized = {entity_id: normalized for normalized, entity_id in entity_ids.items()}
    community_entities: dict[int, list[int]] = defaultdict(list)
    for entity_id, community_id in communities.items():
        community_entities[community_id].append(entity_id)
    article_by_id = {str(article.get("id") or ""): article for article in articles}
    article_communities: dict[str, dict[int, int]] = {}
    community_articles: dict[int, Counter[str]] = defaultdict(Counter)
    for article_id, entities in article_entities.items():
        counts: Counter[int] = Counter()
        for normalized in entities:
            community_id = communities.get(entity_ids[normalized])
            if community_id:
                counts[community_id] += 1
                community_articles[community_id][article_id] += 1
        if counts:
            article_communities[article_id] = dict(counts)

    records: dict[int, dict[str, Any]] = {}
    for community_id, members in community_entities.items():
        descriptive_members = [
            member
            for member in members
            if entity_types.get(id_to_normalized[member]) != "author"
            and not _is_generic_relation(id_to_normalized[member])
        ] or members
        ranked_entities = sorted(
            descriptive_members,
            key=lambda entity_id: (
                -document_counts[id_to_normalized[entity_id]],
                names[id_to_normalized[entity_id]].lower(),
            ),
        )
        top_names = [names[id_to_normalized[item]] for item in ranked_entities[:8]]
        representative = sorted(
            community_articles.get(community_id, {}).items(),
            key=lambda item: (
                -item[1],
                str(article_by_id.get(item[0], {}).get("date") or ""),
                item[0],
            ),
        )[:6]
        representative_rows = [
            {
                "article_id": article_id,
                "title": str(article_by_id.get(article_id, {}).get("title") or ""),
                "date": str(article_by_id.get(article_id, {}).get("date") or "")[:10],
            }
            for article_id, _ in representative
        ]
        dates = [
            str(article_by_id.get(article_id, {}).get("date") or "")[:10]
            for article_id in community_articles.get(community_id, {})
            if str(article_by_id.get(article_id, {}).get("date") or "")
        ]
        label = " · ".join(top_names[:3])
        titles = "; ".join(
            f"{row['title']} ({row['date'][:4]})" if row["date"] else row["title"]
            for row in representative_rows[:4]
            if row["title"]
        )
        span = (
            f"{min(dates)[:4]}–{max(dates)[:4]}" if dates else "undated"
        )
        summary = (
            f"Archive community spanning {span}, connecting {', '.join(top_names[:6])}."
            + (f" Representative reporting: {titles}." if titles else "")
        )
        records[community_id] = {
            "label": label,
            "summary": summary,
            "entity_count": len(members),
            "article_count": len(community_articles.get(community_id, {})),
            "first_date": min(dates) if dates else "",
            "last_date": max(dates) if dates else "",
            "top_entities": top_names,
            "representative_articles": representative_rows,
        }
    return records, article_communities


def _write_graph(
    path: Path,
    articles: list[dict[str, Any]],
    article_entities: dict[str, dict[str, int]],
    entity_ids: dict[str, int],
    names: dict[str, str],
    entity_types: dict[str, str],
    document_counts: Counter[str],
    aliases: dict[str, set[str]],
    edges: dict[tuple[int, int], tuple[int, str, str]],
    communities: dict[int, int],
    community_records: dict[int, dict[str, Any]],
    article_communities: dict[str, dict[int, int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE articles (
                article_id TEXT PRIMARY KEY, title TEXT NOT NULL, date TEXT NOT NULL, url TEXT NOT NULL
            );
            CREATE TABLE entities (
                entity_id INTEGER PRIMARY KEY,
                canonical_name TEXT NOT NULL,
                normalized_name TEXT NOT NULL UNIQUE,
                entity_type TEXT NOT NULL,
                article_count INTEGER NOT NULL,
                community_id INTEGER
            );
            CREATE TABLE aliases (
                alias_norm TEXT NOT NULL, alias TEXT NOT NULL, entity_id INTEGER NOT NULL,
                PRIMARY KEY (alias_norm, entity_id)
            );
            CREATE TABLE article_entities (
                article_id TEXT NOT NULL, entity_id INTEGER NOT NULL, mentions INTEGER NOT NULL,
                PRIMARY KEY (article_id, entity_id)
            );
            CREATE TABLE edges (
                source_entity_id INTEGER NOT NULL, target_entity_id INTEGER NOT NULL,
                weight INTEGER NOT NULL, first_date TEXT NOT NULL, last_date TEXT NOT NULL,
                PRIMARY KEY (source_entity_id, target_entity_id)
            );
            CREATE TABLE communities (
                community_id INTEGER PRIMARY KEY, label TEXT NOT NULL, summary TEXT NOT NULL,
                entity_count INTEGER NOT NULL, article_count INTEGER NOT NULL,
                first_date TEXT NOT NULL, last_date TEXT NOT NULL,
                top_entities TEXT NOT NULL, representative_articles TEXT NOT NULL
            );
            CREATE TABLE article_communities (
                article_id TEXT NOT NULL, community_id INTEGER NOT NULL, entity_count INTEGER NOT NULL,
                PRIMARY KEY (article_id, community_id)
            );
            CREATE INDEX idx_article_entities_entity ON article_entities(entity_id, article_id);
            CREATE INDEX idx_edges_source ON edges(source_entity_id, weight DESC);
            CREATE INDEX idx_edges_target ON edges(target_entity_id, weight DESC);
            CREATE INDEX idx_article_communities_community
                ON article_communities(community_id, entity_count DESC);
            CREATE VIRTUAL TABLE entity_search USING fts5(
                entity_id UNINDEXED, name, aliases, tokenize='unicode61 remove_diacritics 2'
            );
            """
        )
        stats = {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "article_count": len(articles),
            "entity_count": len(entity_ids),
            "edge_count": len(edges),
            "community_count": len(community_records),
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [(key, json.dumps(value)) for key, value in stats.items()],
        )
        connection.executemany(
            "INSERT INTO articles(article_id, title, date, url) VALUES (?, ?, ?, ?)",
            [
                (
                    str(article.get("id") or ""),
                    str(article.get("title") or ""),
                    str(article.get("date") or ""),
                    str(article.get("url") or article.get("link") or ""),
                )
                for article in articles
                if str(article.get("id") or "")
            ],
        )
        entity_rows = []
        search_rows = []
        alias_rows = []
        for normalized, entity_id in entity_ids.items():
            entity_aliases = sorted(
                {value for value in aliases.get(normalized, set()) if normalize_entity(value)}
            )
            entity_rows.append(
                (
                    entity_id,
                    names[normalized],
                    normalized,
                    entity_types.get(normalized, "named"),
                    document_counts[normalized],
                    communities.get(entity_id),
                )
            )
            search_rows.append(
                (entity_id, names[normalized], " ".join(entity_aliases))
            )
            alias_rows.extend(
                (normalize_entity(alias), alias, entity_id)
                for alias in entity_aliases
                if normalize_entity(alias)
            )
        connection.executemany(
            """
            INSERT INTO entities(
                entity_id, canonical_name, normalized_name, entity_type,
                article_count, community_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            entity_rows,
        )
        connection.executemany(
            "INSERT OR IGNORE INTO aliases(alias_norm, alias, entity_id) VALUES (?, ?, ?)",
            alias_rows,
        )
        connection.executemany(
            "INSERT INTO entity_search(entity_id, name, aliases) VALUES (?, ?, ?)",
            search_rows,
        )
        connection.executemany(
            "INSERT INTO article_entities(article_id, entity_id, mentions) VALUES (?, ?, ?)",
            [
                (article_id, entity_ids[normalized], mentions)
                for article_id, entities in article_entities.items()
                for normalized, mentions in entities.items()
            ],
        )
        connection.executemany(
            """
            INSERT INTO edges(
                source_entity_id, target_entity_id, weight, first_date, last_date
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (left, right, weight, first_date, last_date)
                for (left, right), (weight, first_date, last_date) in edges.items()
            ],
        )
        connection.executemany(
            """
            INSERT INTO communities(
                community_id, label, summary, entity_count, article_count,
                first_date, last_date, top_entities, representative_articles
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    community_id,
                    record["label"],
                    record["summary"],
                    record["entity_count"],
                    record["article_count"],
                    record["first_date"],
                    record["last_date"],
                    json.dumps(record["top_entities"], ensure_ascii=False),
                    json.dumps(record["representative_articles"], ensure_ascii=False),
                )
                for community_id, record in community_records.items()
            ],
        )
        connection.executemany(
            """
            INSERT INTO article_communities(article_id, community_id, entity_count)
            VALUES (?, ?, ?)
            """,
            [
                (article_id, community_id, entity_count)
                for article_id, values in article_communities.items()
                for community_id, entity_count in values.items()
            ],
        )
        connection.commit()
    finally:
        connection.close()
    temporary.replace(path)


def _matched_entities(
    connection: sqlite3.Connection,
    query: str,
    tokens: list[str],
) -> list[dict[str, Any]]:
    expression = " OR ".join(f'"{token}"' for token in tokens[:12])
    rows = connection.execute(
        """
        SELECT e.entity_id, e.canonical_name, e.normalized_name, e.article_count,
               e.community_id, entity_search.aliases AS search_aliases,
               bm25(entity_search) AS text_rank
        FROM entity_search
        JOIN entities e ON e.entity_id = CAST(entity_search.entity_id AS INTEGER)
        WHERE entity_search MATCH ?
        ORDER BY text_rank
        LIMIT 40
        """,
        (expression,),
    ).fetchall()
    query_norm = normalize_entity(query)
    query_set = set(tokens)
    scored: list[dict[str, Any]] = []
    for row in rows:
        entity_tokens = set(_query_tokens(str(row["normalized_name"]), keep_generic=True))
        overlap = len(query_set & entity_tokens)
        exact_alias = connection.execute(
            f"""
            SELECT 1 FROM aliases
            WHERE entity_id = ? AND alias_norm IN ({','.join('?' for _ in tokens)})
            LIMIT 1
            """,
            (int(row["entity_id"]), *tokens),
        ).fetchone() is not None
        alias_overlap = 1 if exact_alias else 0
        exact_name = str(row["normalized_name"]) in query_norm
        if not overlap and not alias_overlap:
            continue
        coverage = overlap / max(1, len(entity_tokens))
        if (
            len(entity_tokens) > 1
            and not exact_name
            and not exact_alias
            and (overlap < 2 or coverage < 0.8)
        ):
            continue
        score = (2.0 * overlap) + (2.5 * coverage) + (2.0 * alias_overlap)
        if exact_name or exact_alias:
            score += 4.0
        score += min(1.2, math.log1p(int(row["article_count"])) / 5.0)
        scored.append({**dict(row), "score": score})
    scored.sort(key=lambda row: (-float(row["score"]), str(row["canonical_name"])))
    return scored[:8]


def _neighbor_entities(
    connection: sqlite3.Connection,
    seed_scores: dict[int, float],
) -> list[dict[str, Any]]:
    neighbors: dict[int, dict[str, Any]] = {}
    for seed_id, seed_score in seed_scores.items():
        rows = connection.execute(
            """
            SELECT
                CASE WHEN edge.source_entity_id = ? THEN edge.target_entity_id
                     ELSE edge.source_entity_id END AS entity_id,
                edge.weight,
                entity.canonical_name,
                entity.normalized_name,
                entity.entity_type,
                entity.community_id
            FROM edges edge
            JOIN entities entity ON entity.entity_id =
                CASE WHEN edge.source_entity_id = ? THEN edge.target_entity_id
                     ELSE edge.source_entity_id END
            WHERE edge.source_entity_id = ? OR edge.target_entity_id = ?
            ORDER BY edge.weight DESC
            LIMIT 18
            """,
            (seed_id, seed_id, seed_id, seed_id),
        ).fetchall()
        for row in rows:
            entity_id = int(row["entity_id"])
            if (
                entity_id in seed_scores
                or str(row["entity_type"]) == "author"
                or _is_generic_relation(str(row["normalized_name"]))
            ):
                continue
            score = seed_score * 0.34 * math.log1p(int(row["weight"]))
            current = neighbors.get(entity_id)
            if current is None or score > float(current["score"]):
                neighbors[entity_id] = {**dict(row), "score": score}
    return sorted(
        neighbors.values(),
        key=lambda row: (-float(row["score"]), str(row["canonical_name"])),
    )[:32]


def _matched_communities(
    connection: sqlite3.Connection,
    entity_scores: dict[int, float],
    limit: int,
) -> list[dict[str, Any]]:
    scores: Counter[int] = Counter()
    for entity_id, score in entity_scores.items():
        row = connection.execute(
            "SELECT community_id FROM entities WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()
        if row and row["community_id"]:
            scores[int(row["community_id"])] += score
    rows: list[dict[str, Any]] = []
    ranked = scores.most_common()
    minimum = (ranked[0][1] * 0.12) if ranked else 0.0
    for community_id, score in ranked:
        if score < minimum:
            continue
        record = connection.execute(
            "SELECT * FROM communities WHERE community_id = ?",
            (community_id,),
        ).fetchone()
        if record:
            rows.append({**dict(record), "score": float(score)})
        if len(rows) >= limit:
            break
    return rows


def _article_in_community(
    connection: sqlite3.Connection,
    article_id: str,
    community_id: int,
) -> bool:
    return connection.execute(
        """
        SELECT 1 FROM article_communities
        WHERE article_id = ? AND community_id = ?
        """,
        (article_id, community_id),
    ).fetchone() is not None


def _query_tokens(value: str, keep_generic: bool = False) -> list[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(str(value or ""))]
    if not keep_generic:
        tokens = [token for token in tokens if token not in GENERIC_ENTITY_TOKENS]
    return list(dict.fromkeys(token for token in tokens if len(token) > 1))


def _is_generic_relation(normalized: str) -> bool:
    simplified = " ".join(str(normalized).replace("&", " ").split())
    tokens = set(simplified.split())
    return (
        simplified in GENERIC_RELATION_NODES
        or "podcast" in tokens
        or simplified in {"u s", "united states"}
    )


def _load_articles(articles_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(articles_dir).glob("*.json"), key=_article_path_key):
        try:
            article = load_article(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if isinstance(article, dict):
            rows.append(article)
    return rows


def _article_path_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), path.name) if path.stem.isdigit() else (sys.maxsize, path.name)


def _json_value(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    stats = build_knowledge_graph(args.data_dir, args.output)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
