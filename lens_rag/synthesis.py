"""Research planning and evidence organization for investigative synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import re
from typing import Any, Iterable

from .config import ARTICLES_DIRNAME
from .embedder import article_to_chunks, load_article
from .query_engine import ArticleResult, ChunkMatch
from .query_engine import make_excerpt


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResearchFacet:
    """One explicit information need derived from a reporter's query."""

    key: str
    label: str
    query: str

    def to_dict(self) -> dict[str, str]:
        return {"key": self.key, "label": self.label, "query": self.query}


@dataclass
class ResearchPlan:
    """A bounded set of complementary retrieval facets."""

    query: str
    facets: list[ResearchFacet]
    contract: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "facets": [facet.to_dict() for facet in self.facets],
            "research_contract": list(self.contract),
        }


@dataclass(frozen=True)
class EvidenceItem:
    """One source-addressable excerpt in the synthesis evidence ledger."""

    evidence_id: str
    source_index: int
    article_id: str
    title: str
    publication_date: str
    categories: str
    tags: str
    excerpt: str
    chunk_index: int
    relevance: float
    facets: tuple[str, ...] = ()

    def render(self) -> str:
        facet_text = ", ".join(self.facets) if self.facets else "core"
        return "\n".join(
            [
                f"EVIDENCE {self.evidence_id} · SOURCE [{self.source_index}]",
                f"Article: {self.title}",
                f"Published: {self.publication_date or 'unknown'}",
                f"Facets: {facet_text}",
                f"Excerpt: {self.excerpt}",
            ]
        )


@dataclass
class EvidenceLedger:
    """Structured evidence plus coverage metadata for a synthesis request."""

    items: list[EvidenceItem]
    source_facets: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_results(
        cls,
        results: list[ArticleResult],
        source_facets: dict[str, tuple[str, ...]] | None = None,
        excerpts_per_article: int = 4,
    ) -> "EvidenceLedger":
        facet_map = source_facets or {}
        items: list[EvidenceItem] = []
        for source_index, result in enumerate(results, start=1):
            for match_number, match in enumerate(
                (result.matches or [])[:excerpts_per_article], start=1
            ):
                excerpt = str(match.excerpt or "").strip()
                if not excerpt:
                    continue
                items.append(
                    EvidenceItem(
                        evidence_id=f"S{source_index}-E{match_number}",
                        source_index=source_index,
                        article_id=str(result.article_id),
                        title=str(result.title),
                        publication_date=str(result.date or ""),
                        categories=str(result.categories or ""),
                        tags=str(result.tags or ""),
                        excerpt=excerpt,
                        chunk_index=int(match.chunk_index),
                        relevance=float(match.score),
                        facets=tuple(facet_map.get(str(result.article_id), ())),
                    )
                )
        items.sort(key=lambda item: (_date_key(item.publication_date), item.source_index))
        return cls(items=items, source_facets=facet_map)

    def batches(self, max_chars: int = 12_000) -> list[list[EvidenceItem]]:
        """Partition evidence without splitting an evidence item."""

        batches: list[list[EvidenceItem]] = []
        current: list[EvidenceItem] = []
        current_chars = 0
        for item in self.items:
            size = len(item.render())
            if current and current_chars + size > max_chars:
                batches.append(current)
                current = []
                current_chars = 0
            current.append(item)
            current_chars += size
        if current:
            batches.append(current)
        return batches

    def diagnostics(self) -> dict[str, Any]:
        dates = [item.publication_date[:10] for item in self.items if item.publication_date]
        source_ids = {item.article_id for item in self.items}
        facets = sorted({facet for item in self.items for facet in item.facets})
        return {
            "source_count": len(source_ids),
            "evidence_item_count": len(self.items),
            "facet_count": len(facets),
            "facets_covered": facets,
            "date_from": min(dates) if dates else None,
            "date_to": max(dates) if dates else None,
        }


@dataclass
class ResearchContext:
    """Coverage-aware results and diagnostics returned by research retrieval."""

    plan: ResearchPlan
    results: list[ArticleResult]
    source_facets: dict[str, tuple[str, ...]]
    facet_result_counts: dict[str, int]
    graph_diagnostics: dict[str, Any] = field(default_factory=dict)
    adjacent_context_count: int = 0

    def diagnostics(self) -> dict[str, Any]:
        ledger = EvidenceLedger.from_results(
            self.results,
            self.source_facets,
            excerpts_per_article=6,
        )
        diagnostics = {
            **ledger.diagnostics(),
            "facet_result_counts": self.facet_result_counts,
            "research_plan": self.plan.to_dict(),
            "adjacent_context_count": self.adjacent_context_count,
        }
        if self.graph_diagnostics:
            diagnostics["knowledge_graph"] = self.graph_diagnostics
        return diagnostics

    def prompt_plan(self) -> dict[str, Any]:
        """Return the compact plan and community context supplied to synthesis."""

        payload = self.plan.to_dict()
        communities = self.graph_diagnostics.get("communities", [])
        if communities:
            payload["knowledge_graph_communities"] = communities[:4]
        payload["retrieval_note"] = (
            "Knowledge-graph communities are extractive navigation aids; numbered "
            "article SOURCE blocks remain the only factual authority."
        )
        return payload


def build_research_plan(query: str) -> ResearchPlan:
    """Create an intent-aware investigative plan without another paid model call."""

    cleaned = " ".join(str(query).split()).strip()
    suffixes: list[tuple[str, str, str]] = [
        ("core", "Core findings", ""),
        (
            "historical_baseline",
            "Origins and historical baseline",
            "origin first reported background history earlier conditions",
        ),
        (
            "current_status",
            "Current status and change over time",
            "latest follow-up current status implementation changes over time",
        ),
        (
            "accountability",
            "Actors and accountability",
            "officials agencies organizations responsibility response oversight",
        ),
        (
            "legal_policy",
            "Legal and policy record",
            "lawsuit court ruling order policy compliance investigation",
        ),
        (
            "stakes_outcomes",
            "Stakes, consequences, and outcomes",
            "funding contracts consequences affected communities promises implementation outcomes",
        ),
    ]
    lowered = cleaned.lower()
    contract = [
        "Establish the supported core findings and necessary background.",
        "Reconstruct chronology without treating publication date as event date.",
        "Identify actors, institutions, responsibility, and accountability.",
        "Separate documented facts, attributed allegations, official claims, and rulings.",
        "Explain consequences, contradictions, uncertainty, and unanswered questions.",
    ]

    if _query_mentions(
        lowered,
        "change",
        "changed",
        "compare",
        "comparison",
        "before",
        "after",
        "versus",
        "vs",
        "over time",
    ):
        suffixes.append(
            (
                "comparison",
                "Comparison across periods or institutions",
                "compare before after difference changed persisted trajectory",
            )
        )
        contract.append("Compare relevant periods or institutions on like-for-like evidence.")

    if _query_mentions(
        lowered,
        "money",
        "fund",
        "funding",
        "budget",
        "contract",
        "spending",
        "cost",
        "paid",
        "dollar",
        "$",
    ):
        suffixes.append(
            (
                "money_mechanisms",
                "Money, contracts, and operational mechanisms",
                "budget spending funding contract vendor recipient stated purpose operational mechanism",
            )
        )
        contract.append("Trace money or policy mechanisms from stated purpose to reported result.")

    if _query_mentions(
        lowered,
        "promise",
        "promised",
        "reform",
        "implemented",
        "implementation",
        "outcome",
        "result",
        "worked",
        "effective",
    ):
        suffixes.append(
            (
                "promises_outcomes",
                "Promises versus implementation and outcomes",
                "promise commitment reform implementation follow-through measured outcome gap",
            )
        )
        contract.append("Compare stated commitments with documented implementation and outcomes.")

    if _query_mentions(
        lowered,
        "allege",
        "alleged",
        "claim",
        "claimed",
        "claims",
        "conflict",
        "controversy",
        "disagree",
        "dispute",
        "disputed",
    ):
        suffixes.append(
            (
                "conflicting_accounts",
                "Conflicting and corroborating accounts",
                "official response advocate resident witness dispute contradiction corroboration",
            )
        )
        contract.append("Represent material disagreements and the evidence available for each account.")

    facets = [
        ResearchFacet(
            key=key,
            label=label,
            query=cleaned if not suffix else f"{cleaned} {suffix}",
        )
        for key, label, suffix in suffixes
    ]
    return ResearchPlan(query=cleaned, facets=facets, contract=tuple(contract))


def build_research_context(
    engine: Any,
    query: str,
    initial_results: list[ArticleResult],
    *,
    category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    max_articles: int = 12,
    facet_top_k: int = 8,
) -> ResearchContext:
    """Retrieve per facet and select a relevant, diverse article portfolio."""

    plan = build_research_plan(query)
    facet_results: dict[str, list[ArticleResult]] = {"core": initial_results}
    graph_diagnostics: dict[str, Any] = {}
    for facet in plan.facets[1:]:
        try:
            retrieved = _search_facet(
                engine,
                facet.query,
                top_k=facet_top_k,
                category=category,
                date_from=date_from,
                date_to=date_to,
            )
            if facet.key == "historical_baseline":
                retrieved = _sort_temporal_results(retrieved, newest_first=False)
            elif facet.key == "current_status":
                retrieved = _sort_temporal_results(retrieved, newest_first=True)
            facet_results[facet.key] = retrieved
        except Exception as exc:  # noqa: BLE001 - preserve core retrieval
            logger.warning("Research facet %s failed: %s", facet.key, exc)
            facet_results[facet.key] = []

    if hasattr(engine, "graph_search_results"):
        try:
            graph_results, graph_diagnostics = engine.graph_search_results(
                query,
                top_k=facet_top_k,
            )
            if graph_results:
                facet_results["relational_graph"] = graph_results
        except Exception as exc:  # noqa: BLE001 - semantic retrieval remains available
            logger.warning("Knowledge graph retrieval failed: %s", exc)
            graph_diagnostics = {"enabled": False, "error": str(exc)}

    selected, source_facets = _coverage_select(facet_results, max_articles=max_articles)
    adjacent_context_count = _expand_adjacent_context(engine, selected)
    return ResearchContext(
        plan=plan,
        results=selected,
        source_facets=source_facets,
        facet_result_counts={key: len(value) for key, value in facet_results.items()},
        graph_diagnostics=graph_diagnostics,
        adjacent_context_count=adjacent_context_count,
    )


def _search_facet(
    engine: Any,
    query: str,
    *,
    top_k: int,
    category: str | None,
    date_from: str | None,
    date_to: str | None,
) -> list[ArticleResult]:
    if hasattr(engine, "search_results"):
        page = engine.search_results(
            query,
            page=1,
            page_size=top_k,
            category=category,
            date_from=date_from,
            date_to=date_to,
            nl_rewrite=False,
        )
        return list(page.results)
    return list(
        engine.search(
            query,
            top_k=top_k,
            category=category,
            date_from=date_from,
            date_to=date_to,
        )
    )


def _coverage_select(
    facet_results: dict[str, list[ArticleResult]],
    *,
    max_articles: int,
) -> tuple[list[ArticleResult], dict[str, tuple[str, ...]]]:
    """Select a relevant portfolio with facet, time, and topic diversity."""

    merged: dict[str, ArticleResult] = {}
    article_facets: dict[str, set[str]] = {}
    first_seen: dict[str, int] = {}
    next_seen = 0

    for facet, results in facet_results.items():
        for result in results:
            article_id = str(result.article_id)
            if article_id not in first_seen:
                first_seen[article_id] = next_seen
                next_seen += 1
            article_facets.setdefault(article_id, set()).add(facet)
            if article_id not in merged:
                merged[article_id] = _copy_result(result)
            else:
                _merge_result(merged[article_id], result)

    if not merged or max_articles <= 0:
        return [], {}

    selected_ids: list[str] = []
    core_ids = [
        str(result.article_id)
        for result in facet_results.get("core", [])
        if str(result.article_id) in merged
    ]
    if core_ids:
        selected_ids.append(core_ids[0])

    covered_facets = set(article_facets.get(selected_ids[0], set())) if selected_ids else set()
    date_buckets = {
        _date_bucket(merged[article_id].date) for article_id in selected_ids
    } - {""}
    metadata_tokens = set().union(
        *(_metadata_tokens(merged[article_id]) for article_id in selected_ids)
    ) if selected_ids else set()

    while len(selected_ids) < min(max_articles, len(merged)):
        candidates = [
            article_id for article_id in merged if article_id not in selected_ids
        ]
        if not candidates:
            break

        def portfolio_score(article_id: str) -> tuple[float, int]:
            result = merged[article_id]
            facets = article_facets.get(article_id, set())
            uncovered = len(facets - covered_facets)
            date_bucket = _date_bucket(result.date)
            date_novelty = 1.0 if date_bucket and date_bucket not in date_buckets else 0.0
            tokens = _metadata_tokens(result)
            topic_novelty = (
                len(tokens - metadata_tokens) / max(1, len(tokens)) if tokens else 0.0
            )
            redundancy = max(
                (
                    _result_similarity(result, merged[selected_id])
                    for selected_id in selected_ids
                ),
                default=0.0,
            )
            score = (
                5.0 * uncovered
                + 2.0 * max(0.0, min(1.0, float(result.score)))
                + 1.25 * date_novelty
                + 0.75 * topic_novelty
                - 2.0 * redundancy
            )
            return (score, -first_seen[article_id])

        selected_id = max(candidates, key=portfolio_score)
        selected_ids.append(selected_id)
        covered_facets.update(article_facets.get(selected_id, set()))
        bucket = _date_bucket(merged[selected_id].date)
        if bucket:
            date_buckets.add(bucket)
        metadata_tokens.update(_metadata_tokens(merged[selected_id]))

    selected = [merged[article_id] for article_id in selected_ids]
    source_facets = {
        article_id: tuple(sorted(article_facets.get(article_id, set())))
        for article_id in selected_ids
    }
    return selected, source_facets


def _expand_adjacent_context(engine: Any, results: list[ArticleResult]) -> int:
    """Add neighboring local chunks around selected semantic hits."""

    data_dir = getattr(engine, "data_dir", None)
    if data_dir is None:
        return 0
    articles_dir = Path(data_dir) / ARTICLES_DIRNAME
    added_count = 0
    for result in results:
        article_path = articles_dir / f"{result.article_id}.json"
        if not article_path.exists():
            continue
        try:
            chunks = article_to_chunks(load_article(article_path))
        except Exception as exc:  # noqa: BLE001 - retrieval remains usable
            logger.warning("Adjacent context failed for article %s: %s", result.article_id, exc)
            continue
        if not chunks:
            continue

        known_indices = {int(match.chunk_index) for match in result.matches}
        candidate_scores: dict[int, float] = {}
        for match in list(result.matches):
            for index in (int(match.chunk_index) - 1, int(match.chunk_index) + 1):
                if 0 <= index < len(chunks) and index not in known_indices:
                    candidate_scores[index] = max(
                        candidate_scores.get(index, 0.0),
                        max(0.0, float(match.score) * 0.96),
                    )

        for index, score in sorted(
            candidate_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:4]:
            result.matches.append(
                ChunkMatch(
                    score=score,
                    excerpt=make_excerpt(chunks[index].document),
                    chunk_index=index,
                )
            )
            known_indices.add(index)
            added_count += 1
        result.matches.sort(key=lambda match: match.score, reverse=True)
    return added_count


def _query_mentions(query: str, *terms: str) -> bool:
    return any(
        term in query if not re.fullmatch(r"[a-z0-9]+", term) else re.search(
            rf"\b{re.escape(term)}\b", query
        )
        for term in terms
    )


def _date_bucket(value: str) -> str:
    match = re.search(r"\b(19|20)(\d{2})\b", str(value or ""))
    if not match:
        return ""
    year = int(match.group(0))
    return f"{year - (year % 3)}-{year - (year % 3) + 2}"


def _metadata_tokens(result: ArticleResult) -> set[str]:
    return {
        token
        for token in re.findall(
            r"[a-z0-9]{3,}",
            f"{result.categories} {result.tags}".lower(),
        )
        if token not in {"and", "the", "news"}
    }


def _result_similarity(left: ArticleResult, right: ArticleResult) -> float:
    def tokens(result: ArticleResult) -> set[str]:
        excerpt = " ".join(match.excerpt for match in result.matches[:2])
        return set(re.findall(r"[a-z0-9]{4,}", f"{result.title} {excerpt}".lower()))

    left_tokens = tokens(left)
    right_tokens = tokens(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def _copy_result(result: ArticleResult) -> ArticleResult:
    return ArticleResult(
        article_id=str(result.article_id),
        title=str(result.title),
        date=str(result.date),
        url=str(result.url),
        categories=str(result.categories),
        tags=str(result.tags),
        score=float(result.score),
        score_info=result.score_info,
        matches=[
            ChunkMatch(
                score=float(match.score),
                excerpt=str(match.excerpt),
                chunk_index=int(match.chunk_index),
            )
            for match in result.matches
        ],
    )


def _merge_result(target: ArticleResult, incoming: ArticleResult) -> None:
    target.score = max(float(target.score), float(incoming.score))
    known = {(match.chunk_index, match.excerpt) for match in target.matches}
    for match in incoming.matches:
        key = (match.chunk_index, match.excerpt)
        if key in known:
            continue
        target.matches.append(
            ChunkMatch(
                score=float(match.score),
                excerpt=str(match.excerpt),
                chunk_index=int(match.chunk_index),
            )
        )
        known.add(key)
    target.matches.sort(key=lambda match: match.score, reverse=True)


def render_evidence_batch(items: Iterable[EvidenceItem]) -> str:
    return "\n\n".join(item.render() for item in items)


def _sort_temporal_results(
    results: list[ArticleResult],
    *,
    newest_first: bool,
) -> list[ArticleResult]:
    """Order dated candidates by publication time and keep undated items last."""

    dated = [result for result in results if str(result.date or "").strip()]
    undated = [result for result in results if not str(result.date or "").strip()]
    dated.sort(key=lambda result: str(result.date), reverse=newest_first)
    return dated + undated


def evaluate_synthesis(answer: str, source_count: int) -> dict[str, Any]:
    """Return transparent, no-credit structural quality checks for one brief."""

    citation_groups = re.findall(r"\[((?:\s*\d+\s*,?)+)\]", answer)
    references = [
        int(value)
        for group in citation_groups
        for value in re.findall(r"\d+", group)
    ]
    valid_references = [value for value in references if 1 <= value <= source_count]
    cited_sources = sorted(set(valid_references))
    paragraphs = [part.strip() for part in re.split(r"\n+", answer) if part.strip()]
    cited_paragraphs = [part for part in paragraphs if re.search(r"\[\s*\d+", part)]
    sentences = [
        sentence.strip()
        for paragraph in paragraphs
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph)
        if sentence.strip()
    ]
    citation_end = re.compile(
        r"(?:\[\s*\d+(?:\s*,\s*\d+)*\s*\]\s*)+[.!?][\"']?$"
    )
    cited_sentences = [sentence for sentence in sentences if citation_end.search(sentence)]
    paragraph_coverage = (
        len(cited_paragraphs) / len(paragraphs) if paragraphs else 0.0
    )
    source_utilization = len(cited_sources) / source_count if source_count else 0.0
    sentence_coverage = len(cited_sentences) / len(sentences) if sentences else 0.0
    prose = re.sub(r"\[((?:\s*\d+\s*,?)+)\]", "", answer)
    return {
        "answer_word_count": len(re.findall(r"\b[\w'-]+\b", prose)),
        "citation_marker_count": len(citation_groups),
        "citation_reference_count": len(valid_references),
        "cited_source_count": len(cited_sources),
        "cited_source_indices": cited_sources,
        "paragraph_citation_coverage": round(paragraph_coverage, 3),
        "sentence_citation_coverage": round(sentence_coverage, 3),
        "source_citation_utilization": round(source_utilization, 3),
        "invalid_citation_count": len(references) - len(valid_references),
    }


def _date_key(value: str) -> tuple[int, str]:
    text = str(value or "")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return (0, parsed.isoformat())
    except ValueError:
        return (1, text)
