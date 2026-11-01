"""Natural-language query helpers for Lens archive search."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

def load_local_env() -> None:
    """Load Lens CLI settings explicitly, never as a library import side effect."""

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                if key.strip():
                    os.environ.setdefault(key.strip(), value.strip())

logger = logging.getLogger(__name__)

DEFAULT_REWRITE_MODEL = "gpt-4o-mini"
DEFAULT_REWRITE_TIMEOUT_SECONDS = 8.0
DEFAULT_ANSWER_MODEL = "gpt-4o-mini"
DEFAULT_ANSWER_TIMEOUT_SECONDS = 30.0
DEFAULT_LOCAL_ANSWER_MODEL = "mlx-community/Qwen3-8B-4bit"
DEFAULT_LOCAL_LLM_BASE_URL = "http://127.0.0.1:5368/v1"
DEFAULT_LOCAL_ANSWER_TIMEOUT_SECONDS = 120.0
DEFAULT_DEEPSEEK_ANSWER_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_PRO_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_DEEPSEEK_ANSWER_TIMEOUT_SECONDS = 120.0
DEFAULT_OPUS_MODEL = "claude-opus-5"
DEFAULT_ANTHROPIC_API_BASE_URL = "https://api.anthropic.com"
DEFAULT_OPUS_ANSWER_TIMEOUT_SECONDS = 420.0
DEFAULT_ANSWER_MAX_TOKENS = 900
MAX_ANSWER_CONTEXT_CHARS = 18000
LOCAL_LLM_PROVIDERS = {"local", "mlx", "mlx-lm"}
DEEPSEEK_PROVIDERS = {"deepseek"}

STOPWORDS = {
    "about",
    "after",
    "all",
    "also",
    "and",
    "any",
    "are",
    "article",
    "articles",
    "before",
    "between",
    "both",
    "can",
    "coverage",
    "did",
    "discuss",
    "discussed",
    "discusses",
    "do",
    "does",
    "for",
    "from",
    "get",
    "has",
    "have",
    "how",
    "into",
    "lens",
    "new",
    "news",
    "nola",
    "orleans",
    "prior",
    "report",
    "reported",
    "reporting",
    "reports",
    "see",
    "show",
    "since",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "those",
    "through",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")

LOCAL_QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "airbnb": ("short-term rental", "vacation rental", "str"),
    "jail": ("prison", "detention", "opp", "orleans justice center"),
    "nopd": ("new orleans police department", "police"),
    "orleans parish prison": ("opp", "jail", "orleans justice center"),
    "sewerage and water board": ("s&wb", "swbno", "water board"),
    "short-term rental": ("airbnb", "vacation rental", "str"),
    "swbno": ("sewerage and water board", "s&wb", "water board"),
}

QUERY_BOILERPLATE_RE = re.compile(
    r"\b(?:what|which|where|when|who|how)\s+(?:has|have|did|does|is|are|was|were)\s+"
    r"(?:the\s+lens\s+)?(?:reported?|written|published|covered|said)\s+(?:on|about)\s+",
    re.IGNORECASE,
)
LEADING_SEARCH_RE = re.compile(
    r"^(?:(?:find|show me)\s+)?(?:stories|articles|coverage|reporting|reports?)\s+"
    r"(?:about|on|of|that\s+(?:mention|discuss|cover))\s+",
    re.IGNORECASE,
)
SEARCH_INTENT_WORD_RE = re.compile(
    r"\b(?:articles?|stories|coverage|reporting|reports?)\b",
    re.IGNORECASE,
)


def _env_float(name: str, default: float) -> float:
    """Read a positive float from the environment with a safe fallback."""

    try:
        return max(0.1, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    """Read a positive integer from the environment with a safe fallback."""

    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class RewriteResult:
    """Result of converting a natural-language query into search terms."""

    original_query: str
    rewritten_query: str
    expansion_terms: list[str] = field(default_factory=list)
    category: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    explanation: str = ""
    used_llm: bool = False

    @classmethod
    def unchanged(cls, query: str, explanation: str = "") -> "RewriteResult":
        """Return a no-op rewrite result."""

        return cls(
            original_query=query,
            rewritten_query=query,
            expansion_terms=keywords_from_query(query),
            explanation=explanation,
            used_llm=False,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this rewrite result for API responses."""

        return {
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "expansion_terms": self.expansion_terms,
            "category": self.category,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "explanation": self.explanation,
            "used_llm": self.used_llm,
        }


@dataclass(frozen=True)
class RelevanceScore:
    """Interpretable score metadata for one search result."""

    raw: float
    normalized: float
    tier: str
    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this score for API responses."""

        return {
            "raw": round(self.raw, 4),
            "normalized": round(self.normalized, 1),
            "tier": self.tier,
            "matched_terms": self.matched_terms,
        }


@dataclass(frozen=True)
class SearchResults:
    """Paginated search results with query metadata."""

    query: str
    effective_query: str
    results: list[Any]
    total: int
    page: int
    page_size: int
    has_more: bool
    category: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    rewrite: RewriteResult | None = None
    keywords: list[str] = field(default_factory=list)


class QueryRewriter:
    """Rewrite archive queries locally, with an explicitly enabled LLM option."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_REWRITE_TIMEOUT_SECONDS,
        cache_size: int = 256,
        client: Any | None = None,
        provider: str | None = None,
    ) -> None:
        self.model = model or os.getenv("LENS_RAG_LLM_MODEL", DEFAULT_REWRITE_MODEL)
        self.api_key = api_key if api_key is not None else os.getenv("LENS_RAG_LLM_API_KEY")
        self.timeout = timeout
        self.cache_size = cache_size
        self.client = client
        self.provider = (provider or os.getenv("LENS_RAG_LLM_PROVIDER", "local")).strip().lower()
        if client is not None and provider is None:
            self.provider = "openai"
        self._cache: OrderedDict[tuple[str, tuple[str, ...]], RewriteResult] = OrderedDict()

    def supports_rewriting(self) -> bool:
        """Return true because deterministic local rewriting is always available."""

        return True

    def rewrite(
        self,
        query: str,
        available_categories: list[str] | tuple[str, ...] | None = None,
    ) -> RewriteResult:
        """Rewrite a query, falling back to the original query on any failure."""

        cleaned_query = query.strip()
        categories = tuple(sorted({item.strip() for item in available_categories or [] if item.strip()}))
        if not cleaned_query:
            return RewriteResult.unchanged("")

        cache_key = (cleaned_query, categories)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached

        local_result = local_rewrite(cleaned_query, categories)
        use_llm = self.provider == "openai" and bool(self.client is not None or self.api_key)
        if not use_llm:
            result = local_result
            self._remember(cache_key, result)
            return result

        try:
            result = self._call_llm(cleaned_query, categories)
        except Exception as exc:  # noqa: BLE001 - search should gracefully degrade
            logger.warning("Query rewrite failed; using original query: %s", exc)
            result = local_result

        self._remember(cache_key, result)
        return result

    def _remember(
        self,
        cache_key: tuple[str, tuple[str, ...]],
        result: RewriteResult,
    ) -> None:
        self._cache[cache_key] = result
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _call_llm(self, query: str, categories: tuple[str, ...]) -> RewriteResult:
        client = self.client or self._openai_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a search query analyzer for a local news archive. "
                        "Convert the user's question into concise search terms and "
                        "extract only filters explicitly implied by the question. "
                        "Respond with valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": self._prompt(query, categories),
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
            timeout=self.timeout,
        )
        content = str(response.choices[0].message.content or "{}")
        payload = _json_object(content)
        rewritten_query = str(payload.get("rewritten_query") or query).strip() or query
        expansion_terms = _string_list(payload.get("expansion_terms"))
        if not expansion_terms:
            expansion_terms = keywords_from_query(rewritten_query)
        category = _match_category(payload.get("category"), categories)
        date_from = _clean_date(payload.get("date_from"))
        date_to = _clean_date(payload.get("date_to"))
        explanation = str(payload.get("explanation") or "").strip()
        return RewriteResult(
            original_query=query,
            rewritten_query=rewritten_query,
            expansion_terms=expansion_terms,
            category=category,
            date_from=date_from,
            date_to=date_to,
            explanation=explanation,
            used_llm=True,
        )

    def _openai_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency path
            raise RuntimeError("openai package is not installed") from exc
        return OpenAI(api_key=self.api_key)

    def _prompt(self, query: str, categories: tuple[str, ...]) -> str:
        category_text = ", ".join(categories) if categories else "(none provided)"
        return "\n".join(
            [
                f"Available categories: {category_text}",
                f'User query: "{query}"',
                "",
                "Respond with JSON:",
                "{",
                '  "rewritten_query": "optimized search terms",',
                '  "expansion_terms": ["term1", "term2"],',
                '  "category": "matching category or null",',
                '  "date_from": "YYYY-MM-DD or null",',
                '  "date_to": "YYYY-MM-DD or null",',
                '  "explanation": "brief explanation"',
                "}",
            ]
        )


class AnswerGenerator:
    """Generate a short cited research brief from returned search results."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        client: Any | None = None,
        max_context_chars: int = MAX_ANSWER_CONTEXT_CHARS,
        provider: str | None = None,
        base_url: str | None = None,
        max_tokens: int | None = None,
        audience: str = "reporters",
    ) -> None:
        self.audience = audience
        self.provider = (
            provider or os.getenv("LENS_RAG_LLM_PROVIDER", "local")
        ).strip().lower()
        default_model = (
            DEFAULT_LOCAL_ANSWER_MODEL
            if self.provider in LOCAL_LLM_PROVIDERS
            else DEFAULT_ANSWER_MODEL
        )
        self.model = (
            model
            or os.getenv("LENS_RAG_ANSWER_MODEL")
            or os.getenv("LENS_RAG_LLM_MODEL")
            or default_model
        )
        self.api_key = api_key if api_key is not None else os.getenv("LENS_RAG_LLM_API_KEY")
        configured_base_url = (
            base_url
            if base_url is not None
            else os.getenv("LENS_RAG_LLM_BASE_URL", "").strip()
        )
        self.base_url = configured_base_url or (
            DEFAULT_LOCAL_LLM_BASE_URL
            if self.provider in LOCAL_LLM_PROVIDERS
            else None
        )
        default_timeout = (
            DEFAULT_LOCAL_ANSWER_TIMEOUT_SECONDS
            if self.provider in LOCAL_LLM_PROVIDERS
            else DEFAULT_ANSWER_TIMEOUT_SECONDS
        )
        self.timeout = (
            timeout
            if timeout is not None
            else _env_float("LENS_RAG_ANSWER_TIMEOUT", default_timeout)
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else _env_int("LENS_RAG_ANSWER_MAX_TOKENS", DEFAULT_ANSWER_MAX_TOKENS)
        )
        self.client = client
        self.max_context_chars = max_context_chars

    def supports_answering(self) -> bool:
        """Return true when an LLM client can be used for answer generation."""

        if self.client is not None:
            return True
        if self.provider in LOCAL_LLM_PROVIDERS:
            return bool(self.base_url)
        return bool(self.api_key)

    def generate(self, query: str, results: list[Any]) -> str:
        """Generate a cited brief using only the supplied search results."""

        if not results:
            return "No matching articles were returned, so there is not enough source material to generate a brief."
        if not self.supports_answering():
            raise RuntimeError("answer generation requires a configured local or OpenAI backend")

        messages = [
                {
                    "role": "system",
                    "content": (
                        f"You write concise background briefs for {self.audience}. Use only facts "
                        "explicitly stated in the provided article context. Every factual "
                        "sentence must end with one or more bracketed source numbers like "
                        "[1]. Do not infer dates, chronology, motives, causation, job status, "
                        "or outcomes that the excerpts do not directly establish. Do not "
                        "cite sources outside the supplied list or invent details. Treat "
                        "each SOURCE block as separate: never cite one source for a fact "
                        "that appears only in another source. Prefer one source per factual "
                        "sentence; if a sentence combines sources, cite every contributing "
                        "source. When excerpts are incomplete or disagree, say so plainly."
                    ),
                },
                {
                    "role": "user",
                    "content": self._prompt(query, results),
                },
        ]
        content = self._complete(messages)
        brief = _sanitize_citations(content, len(results))
        if _uncited_sentences(brief):
            brief = _sanitize_citations(
                self._repair_citations(query, results, brief),
                len(results),
            )
        if _uncited_sentences(brief):
            # Qwen sometimes leaves broad transition sentences uncited even
            # after repair. Fail closed at sentence granularity instead of
            # discarding the directly supported portion of the whole brief.
            brief = _prune_uncited_sentences(brief)
        uncited = _uncited_sentences(brief)
        if uncited or not brief:
            raise RuntimeError(
                "answer model did not provide citations for every sentence"
            )
        return brief

    def _complete(self, messages: list[dict[str, str]]) -> str:
        """Run one bounded chat completion and clean model-control tokens."""

        client = self.client or self._openai_client()
        response = client.chat.completions.create(
            **self._completion_params(messages),
        )
        content = _strip_thinking(
            str(response.choices[0].message.content or "").strip()
        )
        if not content:
            return "The answer model returned an empty brief."
        return content

    def _completion_params(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Build provider-specific parameters for an OpenAI-compatible client."""

        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3 if self.provider in LOCAL_LLM_PROVIDERS else 0.2,
            "top_p": 0.8 if self.provider in LOCAL_LLM_PROVIDERS else 1.0,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        if self.provider in DEEPSEEK_PROVIDERS:
            # DeepSeek V4 defaults to thinking mode. Synthesis is an evidence-
            # bounded editing task, so disabling it reduces latency and output cost.
            params["extra_body"] = {"thinking": {"type": "disabled"}}
        return params

    def _repair_citations(
        self,
        query: str,
        results: list[Any],
        brief: str,
    ) -> str:
        """Ask the model once to repair missing or inaccurate inline citations."""

        prompt = "\n\n".join(
            [
                "/no_think" if self.provider in LOCAL_LLM_PROVIDERS else "",
                f"Question: {query}",
                "Edit the brief so every sentence ends with one or more accurate bracketed citations. Preserve supported wording, remove unsupported details, and return only the repaired brief.",
                f"Brief to repair:\n{brief}",
                f"Source context:\n{self._format_context(results)}",
            ]
        ).strip()
        return self._complete(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a citation editor. Use only the numbered SOURCE blocks. "
                        "Never assign a source to a claim it does not explicitly support. "
                        "Every sentence in the returned brief must end with citations."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
        )

    def _openai_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency path
            raise RuntimeError("openai package is not installed") from exc
        kwargs: dict[str, Any] = {
            "api_key": "local" if self.provider in LOCAL_LLM_PROVIDERS else self.api_key,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)

    def _prompt(self, query: str, results: list[Any]) -> str:
        sections = [
            f"Question: {query}",
            "Write 3-4 short paragraphs totaling 250-400 words. Include important caveats and disagreements if the sources show them.",
            "Use only details explicitly present in the excerpts. Omit a detail rather than infer or complete it from general knowledge.",
            "Every factual sentence must end with citations drawn from the bracket numbers shown in the context.",
            "Keep source attribution exact. Do not merge details from different SOURCE blocks unless the sentence cites all of them.",
            "Article context:",
            self._format_context(results),
        ]
        if self.provider in LOCAL_LLM_PROVIDERS:
            sections.insert(0, "/no_think")
        return "\n\n".join(sections)

    def _format_context(self, results: list[Any]) -> str:
        blocks: list[str] = []
        budget = self.max_context_chars
        for index, result in enumerate(results, start=1):
            block = _answer_context_block(index, result)
            if len(block) > budget and blocks:
                break
            if len(block) > budget:
                block = block[:budget].rsplit(" ", 1)[0].strip()
            blocks.append(block)
            budget -= len(block)
            if budget <= 0:
                break
        return "\n\n".join(blocks)


class DeepSeekAnswerGenerator(AnswerGenerator):
    """Generate a cited brief through DeepSeek's API."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        client: Any | None = None,
        max_context_chars: int = MAX_ANSWER_CONTEXT_CHARS,
        base_url: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(
            model=model
            or os.getenv("LENS_RAG_DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_ANSWER_MODEL),
            api_key=api_key if api_key is not None else _deepseek_api_key(),
            timeout=timeout
            if timeout is not None
            else _env_float(
                "LENS_RAG_DEEPSEEK_TIMEOUT",
                DEFAULT_DEEPSEEK_ANSWER_TIMEOUT_SECONDS,
            ),
            client=client,
            max_context_chars=max_context_chars,
            provider="deepseek",
            base_url=base_url
            if base_url is not None
            else os.getenv("LENS_RAG_DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
            max_tokens=max_tokens
            if max_tokens is not None
            else _env_int("LENS_RAG_DEEPSEEK_MAX_TOKENS", DEFAULT_ANSWER_MAX_TOKENS),
        )

    def generate(
        self,
        query: str,
        results: list[Any],
        draft_answer: str | None = None,
    ) -> str:
        """Synthesize from retrieved evidence, optionally refining draft A."""

        if not results:
            return "No matching articles were returned, so there is not enough source material to generate a brief."
        if not self.supports_answering():
            raise RuntimeError(
                "DeepSeek synthesis requires a configured DeepSeek API key"
            )

        draft = (draft_answer or "").strip()
        task = (
            "Audit Draft A against the SOURCE blocks, correct unsupported or "
            "misattributed claims, improve clarity, and return a standalone final brief."
            if draft
            else "Create a standalone research brief directly from the SOURCE blocks."
        )
        prompt_parts = [f"Reporter question: {query}", task]
        if draft:
            prompt_parts.append(f"Draft A:\n{draft}")
        prompt_parts.extend(
            [
                "Write 3-4 short paragraphs totaling 250-400 words. Use only explicitly supported details. Every factual sentence must end with accurate bracketed citations. Never cite a source for a claim that appears only in another source.",
                f"Source context:\n{self._format_context(results)}",
            ]
        )
        prompt = "\n\n".join(prompt_parts)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the second-pass evidence editor for an investigative news "
                    "archive. Treat any supplied draft as untrusted prose and the numbered "
                    "SOURCE blocks as the only authority. Remove unsupported claims rather than "
                    "repairing them with outside knowledge. Every factual sentence must "
                    "end with one or more accurate citations. Return only the final brief."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        brief = _sanitize_citations(self._complete(messages), len(results))
        if _uncited_sentences(brief):
            brief = _sanitize_citations(
                self._repair_citations(query, results, brief),
                len(results),
            )
        if _uncited_sentences(brief):
            brief = _prune_uncited_sentences(brief)
        if _uncited_sentences(brief) or not brief:
            raise RuntimeError(
                "DeepSeek answer model did not provide citations for every sentence"
            )
        return brief


class DeepSeekProAnswerGenerator(DeepSeekAnswerGenerator):
    """Create a deeper, on-demand evidence synthesis with DeepSeek V4 Pro."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        client: Any | None = None,
        max_context_chars: int = 48_000,
        base_url: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(
            model=model
            or os.getenv("LENS_RAG_DEEPSEEK_PRO_MODEL", DEFAULT_DEEPSEEK_PRO_MODEL),
            api_key=api_key,
            timeout=timeout
            if timeout is not None
            else _env_float("LENS_RAG_DEEPSEEK_PRO_TIMEOUT", 300.0),
            client=client,
            max_context_chars=max_context_chars,
            base_url=base_url,
            max_tokens=max_tokens
            if max_tokens is not None
            else _env_int("LENS_RAG_DEEPSEEK_PRO_MAX_TOKENS", 3000),
        )
        self.last_diagnostics: dict[str, Any] = {}
        self._verification_mode = "not run"
        self._claim_audit_diagnostics: dict[str, Any] = {}
        self._reduce_mode = "editorial reduce"

    def generate(
        self,
        query: str,
        results: list[Any],
        draft_answer: str | None = None,
    ) -> str:
        """Generate an in-depth brief through the structured research pipeline."""

        return self.generate_research(query, results)

    def generate_research(
        self,
        query: str,
        results: list[Any],
        *,
        source_facets: dict[str, tuple[str, ...]] | None = None,
        research_plan: dict[str, Any] | None = None,
    ) -> str:
        """Map evidence, plan an outline, reduce, and verify the final brief."""

        if not results:
            return "No matching articles were returned, so there is not enough source material to generate an in-depth brief."
        if not self.supports_answering():
            raise RuntimeError("In-depth synthesis requires a configured provider API key")

        from .synthesis import EvidenceLedger, evaluate_synthesis

        self._verification_mode = "not run"
        self._claim_audit_diagnostics = {}
        self._reduce_mode = "editorial reduce"
        ledger = EvidenceLedger.from_results(
            results,
            source_facets=source_facets,
            excerpts_per_article=6,
        )
        batches = ledger.batches(max_chars=10_000)[:4]
        mapped_notes = self._map_evidence_batches(query, batches)
        if not mapped_notes:
            mapped_notes = [self._format_context(results)]
        outline = self._build_evidence_outline(
            query,
            mapped_notes,
            research_plan=research_plan,
        )
        section_drafts = self._draft_evidence_sections(
            query,
            ledger,
            mapped_notes,
            outline,
        )
        try:
            draft = self._reduce_evidence(
                query,
                results,
                mapped_notes,
                outline,
                section_drafts=section_drafts,
            )
        except Exception as exc:
            # A single transient provider failure in the editorial merge should
            # not discard three independently completed, cited section drafts.
            if not section_drafts:
                raise
            logger.warning(
                "Editorial reduce failed; retaining cited section drafts: %s",
                exc,
            )
            draft = "\n\n".join(section_drafts)
            self._reduce_mode = "cited section-draft fallback"
        brief = self._verify_evidence_brief(query, results, draft)
        self.last_diagnostics = {
            **ledger.diagnostics(),
            **evaluate_synthesis(brief, len(results)),
            "map_batch_count": len(batches),
            "mapped_batch_count": len(mapped_notes),
            "section_draft_count": len(section_drafts),
            "reduce_mode": self._reduce_mode,
            "verification_mode": self._verification_mode,
            "claim_support_audit": self._claim_audit_diagnostics,
            "pipeline_stages": [
                "facet retrieval",
                "evidence ledger",
                "evidence map",
                "outline",
                "section-scoped drafting",
                "editorial reduce",
                "atomic claim and citation verification",
            ],
        }
        return brief

    def _map_evidence_batches(
        self,
        query: str,
        batches: list[list[Any]],
    ) -> list[str]:
        """Extract structured evidence notes from independent context batches."""

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from .synthesis import render_evidence_batch

        def map_one(batch: list[Any]) -> str:
            context = render_evidence_batch(batch)
            prompt = "\n\n".join(
                [
                    f"Reporter question: {query}",
                    "Extract a dense evidence memo from this batch. Organize atomic findings under: chronology; actors and institutions; legal or policy actions; money and operational mechanisms; reported consequences; conflicting accounts; and evidence gaps. Preserve dates, names, legal qualifiers, and uncertainty exactly. Do not write a narrative or use outside knowledge. Every bullet must cite its SOURCE number.",
                    f"Evidence batch:\n{context}",
                ]
            )
            return self._map_generator()._complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are an investigative evidence analyst. Convert excerpts "
                            "into compact, source-addressable notes. Distinguish documented "
                            "facts, quotations, allegations, rulings, and unresolved claims."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
            )

        if not batches:
            return []
        notes_by_index: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=min(3, len(batches))) as executor:
            futures = {
                executor.submit(map_one, batch): index
                for index, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    note = future.result().strip()
                except Exception as exc:  # noqa: BLE001 - degrade to remaining maps
                    logger.warning("Evidence map batch %s failed: %s", index, exc)
                    continue
                if note:
                    notes_by_index[index] = note
        return [notes_by_index[index] for index in sorted(notes_by_index)]

    def _build_evidence_outline(
        self,
        query: str,
        mapped_notes: list[str],
        *,
        research_plan: dict[str, Any] | None,
    ) -> str:
        """Plan a coherent answer whose sections are tied to available evidence."""

        plan_text = json.dumps(research_plan, ensure_ascii=False) if research_plan else ""
        prompt = "\n\n".join(
            part
            for part in [
                f"Reporter question: {query}",
                f"Research facets:\n{plan_text}" if plan_text else "",
                "Design a concise investigative outline from the evidence memos. Each section must state its analytical purpose, the SOURCE numbers it can use, the supported relationship or trajectory it will explain, and any caveat or unresolved question. Prioritize chronology, institutional accountability, mechanisms, consequences, contradictions, and changes over time. Do not draft prose or add facts.",
                "Evidence memos:\n" + "\n\n--- MEMO ---\n\n".join(mapped_notes),
            ]
            if part
        )
        return self._map_generator()._complete(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a senior investigative editor planning a source-grounded "
                        "brief. Build logical structure from the available evidence only."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
        )

    def _draft_evidence_sections(
        self,
        query: str,
        ledger: Any,
        mapped_notes: list[str],
        outline: str,
    ) -> list[str]:
        """Draft complementary sections from facet-scoped evidence cards."""

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from .synthesis import render_evidence_batch

        specs = [
            (
                "Findings and chronology",
                {
                    "core",
                    "historical_baseline",
                    "current_status",
                    "comparison",
                    "relational_graph",
                },
                "Establish what the archive supports, how events unfolded, and what changed or persisted.",
            ),
            (
                "Institutions, decisions, and mechanisms",
                {
                    "accountability",
                    "legal_policy",
                    "money_mechanisms",
                    "promises_outcomes",
                    "relational_graph",
                },
                "Connect actors, decisions, law or policy, money, implementation, and accountability.",
            ),
            (
                "Consequences, disputes, and open questions",
                {
                    "stakes_outcomes",
                    "conflicting_accounts",
                    "current_status",
                    "promises_outcomes",
                },
                "Explain reported consequences, competing accounts, evidentiary limits, and questions the archive leaves unresolved.",
            ),
        ]
        all_items = list(getattr(ledger, "items", []) or [])
        memo_text = "\n\n--- MEMO ---\n\n".join(mapped_notes)

        def draft_one(
            title: str,
            facets: set[str],
            purpose: str,
        ) -> str:
            scoped = [
                item
                for item in all_items
                if facets.intersection(set(getattr(item, "facets", ()) or ()))
            ]
            if not scoped:
                scoped = all_items
            context = render_evidence_batch(scoped[:24])
            prompt = "\n\n".join(
                [
                    f"Reporter question: {query}",
                    f"Evidence-backed outline:\n{outline}",
                    f"Draft only this section: {title}",
                    f"Section purpose: {purpose}",
                    "Write 250-450 rigorous words when the scoped evidence supports that depth. Build cross-source relationships rather than article-by-article summaries. Distinguish facts, attributed allegations, rulings, inference, contradiction, and uncertainty. Do not use facts from the outline or memos unless the evidence cards below support them. Every factual sentence must end with [n] or [n, n]. Return only the section, beginning with a short Markdown heading.",
                    f"Scoped evidence cards:\n{context}",
                    f"Extraction memos (organization aid only):\n{memo_text[:8_000]}",
                ]
            )
            return self._map_generator(max_tokens=1500)._complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are an investigative section editor. Work only from "
                            "the scoped, numbered evidence and retain legal, temporal, "
                            "and attribution qualifiers."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
            )

        if not all_items:
            return []
        drafts: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=len(specs)) as executor:
            futures = {
                executor.submit(draft_one, title, facets, purpose): index
                for index, (title, facets, purpose) in enumerate(specs)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    value = future.result().strip()
                except Exception as exc:  # noqa: BLE001 - reduce can use remaining drafts
                    logger.warning("Evidence section draft %s failed: %s", index, exc)
                    continue
                if value:
                    drafts[index] = value
        return [drafts[index] for index in sorted(drafts)]

    def _reduce_evidence(
        self,
        query: str,
        results: list[Any],
        mapped_notes: list[str],
        outline: str,
        *,
        section_drafts: list[str] | None = None,
    ) -> str:
        """Edit scoped drafts into one coherent, evidence-grounded brief."""

        drafts = list(section_drafts or [])
        prompt = "\n\n".join(
            [
                f"Reporter question: {query}",
                f"Evidence-backed outline:\n{outline}",
                "Evidence memos:\n" + "\n\n--- MEMO ---\n\n".join(mapped_notes),
                (
                    "Facet-scoped section drafts:\n"
                    + "\n\n--- SECTION DRAFT ---\n\n".join(drafts)
                    if drafts
                    else "No section drafts were available; use the evidence memos."
                ),
                "Edit the scoped drafts into a cohesive investigative research brief of roughly 900-1,400 words when evidence supports that depth. Preserve supported detail while removing repetition and improving transitions. Explain relationships across time and institutions rather than listing article summaries. Separate documented facts, attributed allegations, judicial or official findings, synthesis, and unresolved questions. Represent contradictions instead of silently resolving them. Never treat absence from this retrieved set as proof that something never occurred. Do not introduce a factual proposition that is absent from the original SOURCE blocks. Every factual sentence must end with citations in the exact form [1] or [1, 3]. Never write [Source 1], footnotes, or unnumbered links.",
                f"Original SOURCE blocks for verification:\n{self._format_context(results)}",
            ]
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior investigative research editor. The original numbered "
                    "SOURCE blocks are the only authority; evidence memos and the outline are "
                    "organizational aids, and section drafts are provisional. Build a rigorous "
                    "narrative, cross-check claims, surface contradictions and gaps, and never "
                    "add outside knowledge. Return only the finished cited brief."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return self._complete(messages)

    def _verify_evidence_brief(
        self,
        query: str,
        results: list[Any],
        brief: str,
    ) -> str:
        """Validate citations, then perform one atomic support-and-fact audit."""

        verified = self._enforce_sentence_citations(query, results, brief)
        draft_words = _prose_word_count(brief)
        depth_floor = _verification_depth_floor(draft_words)
        if verified and _prose_word_count(verified) >= depth_floor:
            self._verification_mode = "atomic claim audit"
            return self._audit_atomic_claim_support(query, results, verified)

        sectioned = self._verify_evidence_sections(query, results, brief)
        if sectioned and _prose_word_count(sectioned) >= depth_floor:
            self._verification_mode = "sectioned claim recovery"
            return self._audit_atomic_claim_support(query, results, sectioned)

        fallback = _sanitize_citations(brief, len(results))
        if _uncited_sentences(fallback):
            fallback = _prune_uncited_sentences(fallback)
        if fallback and _uncited_sentences(fallback):
            fallback = ""
        candidates = [candidate for candidate in (sectioned, verified, fallback) if candidate]
        if candidates:
            best = max(candidates, key=_prose_word_count)
            if best == sectioned:
                self._verification_mode = "sectioned audit below depth target"
            elif best == fallback:
                self._verification_mode = "validated Pro draft"
            else:
                self._verification_mode = "compact claim audit"
            return self._audit_atomic_claim_support(query, results, best)
        try:
            compact = self._map_generator(max_tokens=1800).generate(
                query,
                results,
                draft_answer=brief,
            )
        except Exception as exc:  # noqa: BLE001 - report final closed failure
            logger.warning("Compact cited recovery failed: %s", exc)
        else:
            compact = _sanitize_citations(compact, len(results))
            if compact:
                self._verification_mode = "compact cited recovery"
                return self._audit_atomic_claim_support(query, results, compact)
        raise RuntimeError("In-depth synthesis did not provide a supported cited analysis")

    def _audit_atomic_claim_support(
        self,
        query: str,
        results: list[Any],
        brief: str,
    ) -> str:
        """Check each cited claim against its cited excerpts and revise fail-closed."""

        claims = _atomic_claim_units(brief)
        if not claims:
            self._claim_audit_diagnostics = {
                "status": "skipped",
                "reason": "no cited factual claims detected",
                "claim_count": 0,
            }
            return brief

        cited_indices = sorted(
            {
                citation
                for claim in claims
                for citation in claim["citations"]
                if 1 <= citation <= len(results)
            }
        )
        source_context = "\n\n".join(
            _answer_context_block(index, results[index - 1])
            for index in cited_indices
        )
        claim_text = "\n".join(
            f'CLAIM {claim["id"]} (cites {claim["citations"]}): {claim["text"]}'
            for claim in claims
        )
        prompt = "\n\n".join(
            [
                f"Reporter question: {query}",
                "Audit every numbered claim only against the source excerpts it cites. A citation is not support merely because it is topically related. Mark each claim supported, partial, or unsupported; separately check names, dates, numbers, chronology, attribution, legal posture, and causal language. Then return a corrected complete brief that removes unsupported propositions and narrows partial ones. Keep supported depth. Do not add outside knowledge.",
                "Return exactly two tagged blocks. The first contains only a JSON array; the second contains raw Markdown so it does not need JSON escaping:\n"
                '<claim_audit>[{"id":1,"status":"supported|partial|unsupported","issue":"short explanation"}]</claim_audit>\n'
                "<revised_brief>complete cited Markdown brief</revised_brief>",
                f"Claims to audit:\n{claim_text}",
                f"Cited source excerpts:\n{source_context}",
                f"Original brief:\n{brief}",
            ]
        )
        try:
            response = self._map_generator(max_tokens=3000)._complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a legal-grade atomic claim verifier. Require direct "
                            "or clearly entailed support from the specific cited excerpts. "
                            "When support is partial or ambiguous, narrow or remove the claim."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            payload = _parse_claim_audit_response(response)
        except Exception as exc:  # noqa: BLE001 - retain prior verified brief
            logger.warning("Atomic claim support audit failed: %s", exc)
            self._claim_audit_diagnostics = {
                "status": "unavailable",
                "claim_count": len(claims),
            }
            return brief

        audits = payload.get("claims", [])
        statuses: dict[int, str] = {}
        if isinstance(audits, list):
            for item in audits:
                if not isinstance(item, dict):
                    continue
                try:
                    claim_id = int(item.get("id"))
                except (TypeError, ValueError):
                    continue
                status = str(item.get("status", "")).strip().lower()
                if status in {"supported", "partial", "unsupported"}:
                    statuses[claim_id] = status
        required_ids = {int(claim["id"]) for claim in claims}
        revised = _sanitize_citations(
            str(payload.get("revised_brief", "") or ""),
            len(results),
        )
        valid_revision = (
            required_ids.issubset(statuses)
            and bool(revised)
            and not _uncited_sentences(revised)
            and _prose_word_count(revised) >= max(
                1,
                int(_prose_word_count(brief) * 0.65),
            )
        )
        self._claim_audit_diagnostics = {
            "status": "applied" if valid_revision else "invalid response",
            "claim_count": len(claims),
            "supported_count": sum(
                status == "supported" for status in statuses.values()
            ),
            "partial_count": sum(status == "partial" for status in statuses.values()),
            "unsupported_count": sum(
                status == "unsupported" for status in statuses.values()
            ),
        }
        return revised if valid_revision else brief

    def _verify_evidence_sections(
        self,
        query: str,
        results: list[Any],
        brief: str,
    ) -> str:
        """Fact-check independent draft sections without collapsing overall depth."""

        from concurrent.futures import ThreadPoolExecutor, as_completed

        paragraphs = [part.strip() for part in re.split(r"\n+", brief) if part.strip()]
        if not paragraphs:
            return ""
        groups: list[list[str]] = []
        current: list[str] = []
        current_chars = 0
        for paragraph in paragraphs:
            if current and current_chars + len(paragraph) > 5_500:
                groups.append(current)
                current = []
                current_chars = 0
            current.append(paragraph)
            current_chars += len(paragraph)
        if current:
            groups.append(current)
        groups = groups[:3]
        source_context = self._format_context(results)

        def verify_one(group: list[str]) -> str:
            section = "\n\n".join(group)
            section_words = _prose_word_count(section)
            prompt = "\n\n".join(
                [
                    f"Reporter question: {query}",
                    f"Fact-check this section without summarizing it. Preserve every supported analytical relationship and at least {max(180, int(section_words * 0.75))} words when the sources support them. Correct or remove unsupported details, legal posture, dates, amounts, attribution, and causation. Every factual sentence must end with citations in the exact form [1] or [1, 3]. Return only the revised section.",
                    f"Section:\n{section}",
                    f"Source context:\n{source_context}",
                ]
            )
            output = self._map_generator(max_tokens=1800)._complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a meticulous investigative fact-checker. Edit the "
                            "section in place; do not condense supported reporting. The "
                            "numbered SOURCE blocks are the only authority."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            cleaned = _sanitize_citations(output, len(results))
            if _uncited_sentences(cleaned):
                cleaned = _prune_uncited_sentences(cleaned)
            return cleaned if cleaned and not _uncited_sentences(cleaned) else ""

        verified: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=min(3, len(groups))) as executor:
            futures = {
                executor.submit(verify_one, group): index
                for index, group in enumerate(groups)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    value = future.result().strip()
                except Exception as exc:  # noqa: BLE001 - other sections remain usable
                    logger.warning("Section verification %s failed: %s", index, exc)
                    continue
                if value:
                    verified[index] = value
        if len(verified) != len(groups):
            return ""
        return "\n\n".join(verified[index] for index in range(len(groups)))

    def _enforce_sentence_citations(
        self,
        query: str,
        results: list[Any],
        brief: str,
    ) -> str:
        """Repair once, then fail closed by retaining only cited sentences."""

        cleaned = _sanitize_citations(brief, len(results))
        if cleaned and _uncited_sentences(cleaned):
            try:
                cleaned = _sanitize_citations(
                    self._map_generator(max_tokens=2200)._repair_citations(
                        query,
                        results,
                        cleaned,
                    ),
                    len(results),
                )
            except Exception as exc:  # noqa: BLE001 - prune below
                logger.warning("Sentence-level citation repair failed: %s", exc)
        if _uncited_sentences(cleaned):
            cleaned = _prune_uncited_sentences(cleaned)
        return cleaned if cleaned and not _uncited_sentences(cleaned) else ""

    def _map_generator(self, max_tokens: int = 1100) -> DeepSeekAnswerGenerator:
        """Return the economical mapper/verifier used around the Pro reduce pass."""

        return DeepSeekAnswerGenerator(
            model=os.getenv("LENS_RAG_DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_ANSWER_MODEL),
            api_key=self.api_key,
            timeout=min(self.timeout, 180.0),
            client=self.client,
            max_context_chars=self.max_context_chars,
            base_url=self.base_url,
            max_tokens=max_tokens,
        )

    def _repair_citations(
        self,
        query: str,
        results: list[Any],
        brief: str,
    ) -> str:
        """Compatibility wrapper for direct callers of the older Pro pipeline."""

        return self._verify_evidence_brief(query, results, brief)

    def _completion_params(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        params = super()._completion_params(messages)
        params["extra_body"] = {"thinking": {"type": "enabled"}}
        params["reasoning_effort"] = "high"
        return params


class OpenAIAnswerGenerator(DeepSeekProAnswerGenerator):
    """Run the independent research and citation-audit pipeline with GPT-6.1 Sol."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        client: Any | None = None,
        max_context_chars: int = 48_000,
        max_tokens: int | None = None,
        effort: str | None = None,
    ) -> None:
        super().__init__(
            model=model or os.getenv("LENS_RAG_OPENAI_MODEL", "gpt-6.1-sol"),
            api_key=api_key if api_key is not None else (
                os.getenv("LENS_RAG_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
            ),
            timeout=timeout if timeout is not None else _env_float("LENS_RAG_OPENAI_TIMEOUT", 240.0),
            client=client,
            max_context_chars=max_context_chars,
            base_url="https://api.openai.com/v1",
            max_tokens=max_tokens if max_tokens is not None else _env_int("LENS_RAG_OPENAI_MAX_TOKENS", 7000),
        )
        self.provider = "openai"
        self.effort = effort or os.getenv("LENS_RAG_OPENAI_EFFORT", "medium")

    def _complete(self, messages: list[dict[str, str]]) -> str:
        return self._responses_generator(self.max_tokens, self.effort)._complete(messages)

    def _map_generator(self, max_tokens: int = 1400) -> _OpenAIResponsesAnswerGenerator:
        # Reasoning shares the output budget, so leave room before the cited text.
        return self._responses_generator(
            max(max_tokens, _env_int("LENS_RAG_OPENAI_MAP_MAX_TOKENS", 5000)),
            os.getenv("LENS_RAG_OPENAI_MAP_EFFORT", "low"),
        )

    def _responses_generator(
        self, max_tokens: int, effort: str,
    ) -> _OpenAIResponsesAnswerGenerator:
        generator = _OpenAIResponsesAnswerGenerator(
            model=self.model,
            api_key=self.api_key,
            timeout=self.timeout,
            client=self.client,
            max_context_chars=self.max_context_chars,
            base_url=self.base_url,
            max_tokens=max_tokens,
        )
        generator.provider = "openai"
        generator.effort = effort
        return generator


class _OpenAIResponsesAnswerGenerator(DeepSeekAnswerGenerator):
    """Use Responses for bounded evidence mapping, auditing, and brief recovery."""

    effort = "low"

    def _complete(self, messages: list[dict[str, str]]) -> str:
        from openai import OpenAIError

        client = self.client or self._openai_client()
        try:
            response = client.responses.create(
                model=self.model,
                input=messages,
                reasoning={"effort": self.effort},
                max_output_tokens=self.max_tokens,
                timeout=self.timeout,
                store=False,
            )
        except OpenAIError as exc:
            # SDK errors can include credential fragments; expose only the status.
            status = getattr(exc, "status_code", None)
            raise RuntimeError(
                f"OpenAI synthesis request failed (HTTP {status})"
                if status else "OpenAI synthesis request failed"
            ) from exc
        if response.status != "completed":
            raise RuntimeError("OpenAI synthesis did not complete within its output budget")
        content = response.output_text.strip()
        if not content:
            raise RuntimeError("OpenAI returned an empty synthesis")
        return content



class _AnthropicDirectAnswerGenerator(DeepSeekAnswerGenerator):
    """Call Anthropic's Messages API directly."""

    def __init__(
        self,
        model: str,
        api_key: str | None,
        *,
        timeout: float,
        client: Any | None = None,
        max_context_chars: int = 48_000,
        base_url: str = DEFAULT_ANTHROPIC_API_BASE_URL,
        max_tokens: int = 1400,
        effort: str | None = None,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            timeout=timeout,
            client=client,
            max_context_chars=max_context_chars,
            base_url=base_url,
            max_tokens=max_tokens,
        )
        self.provider = "anthropic-direct"
        self.effort = effort

    def supports_answering(self) -> bool:
        return bool(self.api_key and self.base_url)

    def _complete(self, messages: list[dict[str, str]]) -> str:
        """Run one direct Anthropic Messages request."""

        import httpx

        system = "\n\n".join(
            str(message["content"])
            for message in messages
            if message["role"] == "system"
        )
        conversation = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
            if message["role"] in {"user", "assistant"}
        ]
        base_url = str(self.base_url or "").rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        url = f"{base_url}/v1/messages"
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": conversation,
        }
        if system:
            payload["system"] = system
        if self.effort:
            payload["output_config"] = {"effort": self.effort}
        headers = {
            "x-api-key": str(self.api_key or ""),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        response = None
        for attempt in range(3):
            if self.client is not None:
                response = self.client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
            else:
                response = httpx.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
            try:
                response.raise_for_status()
                break
            except Exception as exc:
                # Anthropic's current adaptive-thinking models reject non-default
                # sampling parameters and may also return actionable workspace/model
                # access errors. Preserve the safe provider message so a streamed API
                # response does not collapse every 4xx into an opaque generic failure.
                provider_message = ""
                try:
                    error_data = response.json()
                    error = error_data.get("error", {})
                    if isinstance(error, dict):
                        provider_message = str(error.get("message", "")).strip()
                except Exception:  # noqa: BLE001 - error body may not be JSON
                    pass
                status = getattr(response, "status_code", None)
                retryable = status in {408, 409, 429, 500, 502, 503, 504, 529}
                if retryable and attempt < 2:
                    logger.warning(
                        "Transient Anthropic status %s; retrying Opus request",
                        status,
                    )
                    time.sleep(1.5 * (2**attempt))
                    continue
                detail = (
                    provider_message or "Anthropic rejected the Messages request"
                )
                raise RuntimeError(
                    f"Claude Opus 5 request failed: {detail}"
                ) from exc
        assert response is not None
        data = response.json()
        if data.get("stop_reason") == "refusal":
            stop_details = data.get("stop_details", {})
            category = (
                str(stop_details.get("category", "")).strip()
                if isinstance(stop_details, dict)
                else ""
            )
            suffix = f" ({category})" if category else ""
            raise RuntimeError(f"Claude Opus 5 declined this synthesis{suffix}")
        content = "\n".join(
            str(block.get("text", "")).strip()
            for block in data.get("content", [])
            if isinstance(block, dict)
            and block.get("type") == "text"
            and str(block.get("text", "")).strip()
        )
        content = _strip_thinking(content)
        if not content:
            raise RuntimeError("Claude Opus 5 returned an empty response")
        return content


class OpusAnswerGenerator(DeepSeekProAnswerGenerator):
    """Run the structured in-depth research pipeline with Claude Opus 5."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        client: Any | None = None,
        max_context_chars: int = 48_000,
        base_url: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        resolved_api_key = api_key if api_key is not None else _anthropic_api_key()
        resolved_base_url = (
            base_url
            if base_url is not None
            else os.getenv(
                "LENS_RAG_OPUS_BASE_URL",
                DEFAULT_ANTHROPIC_API_BASE_URL,
            )
        )
        super().__init__(
            model=model or os.getenv("LENS_RAG_OPUS_MODEL", DEFAULT_OPUS_MODEL),
            api_key=resolved_api_key,
            timeout=timeout
            if timeout is not None
            else _env_float(
                "LENS_RAG_OPUS_TIMEOUT",
                DEFAULT_OPUS_ANSWER_TIMEOUT_SECONDS,
            ),
            client=client,
            max_context_chars=max_context_chars,
            base_url=resolved_base_url,
            max_tokens=max_tokens
            if max_tokens is not None
            else _env_int("LENS_RAG_OPUS_MAX_TOKENS", 7000),
        )
        self.provider = "anthropic-direct"

    def supports_answering(self) -> bool:
        return bool(self.api_key and self.base_url)

    def generate_research(
        self,
        query: str,
        results: list[Any],
        *,
        source_facets: dict[str, tuple[str, ...]] | None = None,
        research_plan: dict[str, Any] | None = None,
    ) -> str:
        if not self.supports_answering():
            raise RuntimeError(
                "Claude Opus 5 requires ANTHROPIC_API_KEY for direct Anthropic access"
            )
        brief = super().generate_research(
            query,
            results,
            source_facets=source_facets,
            research_plan=research_plan,
        )
        verification = str(self.last_diagnostics.get("verification_mode", ""))
        self.last_diagnostics["verification_mode"] = verification.replace(
            "flash", "Opus"
        )
        self.last_diagnostics["provider_route"] = "direct"
        self.last_diagnostics["effort"] = os.getenv("LENS_RAG_OPUS_EFFORT", "high")
        self.last_diagnostics["intermediate_effort"] = os.getenv(
            "LENS_RAG_OPUS_MAP_EFFORT", os.getenv("LENS_RAG_OPUS_EFFORT", "high")
        )
        return brief

    def _complete(self, messages: list[dict[str, str]]) -> str:
        return self._anthropic_generator(max_tokens=self.max_tokens)._complete(messages)

    def _map_generator(
        self,
        max_tokens: int = 1400,
    ) -> _AnthropicDirectAnswerGenerator:
        # Opus 5's always-on adaptive thinking counts against max_tokens. Small
        # caps that are adequate for non-thinking models can exhaust the budget
        # before any text block is emitted, especially during citation repair and
        # atomic claim verification.
        token_floor = _env_int("LENS_RAG_OPUS_MAP_MAX_TOKENS", 5000)
        return self._anthropic_generator(
            max_tokens=max(max_tokens, token_floor),
            effort=os.getenv("LENS_RAG_OPUS_MAP_EFFORT"),
        )

    def _anthropic_generator(
        self,
        *,
        max_tokens: int,
        effort: str | None = None,
    ) -> _AnthropicDirectAnswerGenerator:
        return _AnthropicDirectAnswerGenerator(
            model=self.model,
            api_key=self.api_key,
            timeout=self.timeout,
            client=self.client,
            max_context_chars=self.max_context_chars,
            base_url=str(self.base_url or DEFAULT_ANTHROPIC_API_BASE_URL),
            max_tokens=max_tokens,
            effort=effort or os.getenv("LENS_RAG_OPUS_EFFORT"),
        )


def _anthropic_api_key() -> str | None:
    """Resolve the Anthropic key without copying it into the repository."""

    direct = os.getenv("LENS_RAG_OPUS_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if direct:
        return direct.strip()

    env_path = Path(
        os.path.expanduser(
            os.getenv("LENS_RAG_OPUS_ENV_FILE", "~/.hermes/.env")
        )
    )
    if not env_path.exists():
        return None
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            if key.removeprefix("export ").strip() == "ANTHROPIC_API_KEY":
                return value.strip().strip('"').strip("'") or None
    except OSError:
        logger.warning("Could not read Anthropic credential file: %s", env_path)
    return None


def _deepseek_api_key() -> str | None:
    """Resolve the DeepSeek key without copying Hermes secrets into this repo."""

    direct = (
        os.getenv("LENS_RAG_DEEPSEEK_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
    )
    if direct:
        return direct.strip()

    env_path = Path(
        os.path.expanduser(
            os.getenv("LENS_RAG_DEEPSEEK_ENV_FILE", "~/.hermes/.env")
        )
    )
    if not env_path.exists():
        return None
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            if key.removeprefix("export ").strip() == "DEEPSEEK_API_KEY":
                return value.strip().strip('"').strip("'") or None
    except OSError:
        logger.warning("Could not read DeepSeek credential file: %s", env_path)
    return None


def local_rewrite(query: str, categories: tuple[str, ...] = ()) -> RewriteResult:
    """Parse common archive filters and aliases without a network model call."""

    today = date.today()
    date_from, date_to, temporal_spans = _local_date_filters(query, today)
    category = _local_category(query, categories)

    rewritten = query
    for start, end in reversed(temporal_spans):
        rewritten = f"{rewritten[:start]} {rewritten[end:]}"
    rewritten = QUERY_BOILERPLATE_RE.sub("", rewritten).strip(" ,?.")
    rewritten = LEADING_SEARCH_RE.sub("", rewritten).strip(" ,?.")
    rewritten = SEARCH_INTENT_WORD_RE.sub(" ", rewritten)
    rewritten = re.sub(
        r"(?:\s+\b(?:in|during|from|after|before|prior|through|until|between|and|to|the)\b)+$",
        "",
        rewritten,
        flags=re.IGNORECASE,
    )
    rewritten = re.sub(r"\s+", " ", rewritten).strip(" ,?.") or query

    terms = keywords_from_query(rewritten)
    lowered = query.lower()
    for phrase, aliases in LOCAL_QUERY_ALIASES.items():
        if phrase in lowered:
            terms.extend(aliases)
    terms = _dedupe_terms(terms)

    details: list[str] = ["Local query analysis"]
    if category:
        details.append(f"category {category}")
    if date_from or date_to:
        details.append(f"dates {date_from or 'start'} to {date_to or 'present'}")
    if len(terms) > len(keywords_from_query(rewritten)):
        details.append("archive aliases expanded")
    return RewriteResult(
        original_query=query,
        rewritten_query=rewritten,
        expansion_terms=terms,
        category=category,
        date_from=date_from,
        date_to=date_to,
        explanation="; ".join(details) + ".",
        used_llm=False,
    )


def _local_date_filters(
    query: str,
    today: date,
) -> tuple[str | None, str | None, list[tuple[int, int]]]:
    """Extract explicit, unambiguous date ranges from natural language."""

    lowered = query.lower()
    spans: list[tuple[int, int]] = []

    between = re.search(
        r"\b(?:between|from)\s+(20\d{2}|19\d{2})\s+(?:and|to|through|until)\s+"
        r"(20\d{2}|19\d{2})\b",
        lowered,
    )
    if between:
        start_year, end_year = sorted((int(between.group(1)), int(between.group(2))))
        spans.append(between.span())
        return f"{start_year:04d}-01-01", f"{end_year:04d}-12-31", spans

    exact_year = re.search(r"\b(?:in|during)\s+(20\d{2}|19\d{2})\b", lowered)
    if exact_year:
        year = int(exact_year.group(1))
        spans.append(exact_year.span())
        return f"{year:04d}-01-01", f"{year:04d}-12-31", spans

    relative = re.search(r"\b(?:past|last)\s+(\d{1,2})\s+years?\b", lowered)
    if relative:
        years = max(1, int(relative.group(1)))
        spans.append(relative.span())
        return f"{today.year - years:04d}-{today.month:02d}-{today.day:02d}", None, spans

    named_relative = re.search(r"\b(this year|last year|past year)\b", lowered)
    if named_relative:
        phrase = named_relative.group(1)
        spans.append(named_relative.span())
        if phrase == "this year":
            return f"{today.year:04d}-01-01", None, spans
        if phrase == "last year":
            return f"{today.year - 1:04d}-01-01", f"{today.year - 1:04d}-12-31", spans
        prior = today - timedelta(days=365)
        return prior.isoformat(), None, spans

    date_from: str | None = None
    date_to: str | None = None
    since = re.search(r"\b(?:since|from|after)\s+(20\d{2}|19\d{2})(?!\s+(?:to|and))\b", lowered)
    if since:
        year = int(since.group(1))
        if since.group(0).startswith("after"):
            year += 1
        date_from = f"{year:04d}-01-01"
        spans.append(since.span())
    before = re.search(
        r"\b(?:before|prior(?:\s+to)?|through|until|to)\s+(20\d{2}|19\d{2})\b",
        lowered,
    )
    if before:
        year = int(before.group(1))
        if before.group(0).startswith(("before", "prior")):
            year -= 1
        date_to = f"{year:04d}-12-31"
        spans.append(before.span())
    return date_from, date_to, spans


def _local_category(query: str, categories: tuple[str, ...]) -> str | None:
    lowered = query.lower()
    matches = [category for category in categories if category.lower() in lowered]
    return max(matches, key=len) if matches else None


class KeywordBooster:
    """Blend vector scores with exact keyword matches."""

    def __init__(self, terms: list[str] | tuple[str, ...]) -> None:
        self.terms = _dedupe_terms(terms)
        self._stemmed_terms = [_stem_phrase(term) for term in self.terms]

    @classmethod
    def from_query(
        cls,
        query: str,
        expansion_terms: list[str] | tuple[str, ...] | None = None,
    ) -> "KeywordBooster":
        """Build a booster from expansion terms plus useful words from the query."""

        return cls(list(expansion_terms or []) + keywords_from_query(query))

    def boost(self, text: str, vector_score: float) -> float:
        """Return a score boosted by keyword overlap without lowering vector score."""

        keyword_score = self.keyword_score(text)
        if keyword_score <= 0:
            return round(max(0.0, min(1.0, vector_score)), 4)
        blended = (0.7 * vector_score) + (0.3 * keyword_score)
        return round(max(vector_score, min(1.0, blended)), 4)

    def keyword_score(self, text: str) -> float:
        """Return a 0-1 score for how many configured terms appear in text."""

        if not self.terms:
            return 0.0
        matched = len(self.matched_terms(text))
        return min(1.0, (matched / max(1, len(self.terms))) * 1.2)

    def matched_terms(self, text: str) -> list[str]:
        """Return display terms that appear in text."""

        if not text or not self.terms:
            return []
        haystack = f" {' '.join(_stem_tokens(text))} "
        matches: list[str] = []
        for display, stemmed in zip(self.terms, self._stemmed_terms):
            if not stemmed:
                continue
            needle = f" {stemmed} "
            if needle in haystack:
                matches.append(display)
        return matches


class RelevanceScorer:
    """Attach normalized score labels and matched terms to article results."""

    def __init__(self, terms: list[str] | tuple[str, ...]) -> None:
        self.booster = KeywordBooster(terms)

    def apply(self, results: list[Any]) -> list[Any]:
        """Attach score_info to each result in place and return the results."""

        if not results:
            return results
        raw_scores = [float(getattr(result, "score", 0.0) or 0.0) for result in results]
        low = min(raw_scores)
        high = max(raw_scores)
        for result in results:
            raw = float(getattr(result, "score", 0.0) or 0.0)
            if high > low:
                normalized = ((raw - low) / (high - low)) * 100.0
            else:
                normalized = 100.0 if raw > 0 else 0.0
            matched = self._matched_terms_for_result(result)
            setattr(
                result,
                "score_info",
                RelevanceScore(
                    raw=raw,
                    normalized=normalized,
                    tier=_tier(raw, normalized),
                    matched_terms=matched,
                ),
            )
        return results

    def _matched_terms_for_result(self, result: Any) -> list[str]:
        parts = [
            str(getattr(result, "title", "") or ""),
            str(getattr(result, "categories", "") or ""),
            str(getattr(result, "tags", "") or ""),
        ]
        for match in getattr(result, "matches", []) or []:
            parts.append(str(getattr(match, "excerpt", "") or ""))
        return self.booster.matched_terms(" ".join(parts))


def keywords_from_query(query: str, limit: int = 12) -> list[str]:
    """Extract useful keyword stems from a raw query."""

    terms: list[str] = []
    for token in TOKEN_RE.findall(query.lower()):
        stemmed = _stem_token(token.strip("'"))
        if len(stemmed) <= 2 or stemmed in STOPWORDS:
            continue
        terms.append(stemmed)
    return _dedupe_terms(terms)[:limit]


def _tier(raw: float, normalized: float) -> str:
    if normalized >= 75.0 and raw >= 0.60:
        return "Strong match"
    if normalized >= 40.0 or raw >= 0.50:
        return "Moderate match"
    return "Weak match"


def _json_object(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content, flags=re.IGNORECASE).strip()
        content = re.sub(r"```$", "", content).strip()
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("rewrite response was not a JSON object")
    return payload


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def _clean_date(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    return text if DATE_RE.match(text) else None


def _match_category(value: Any, categories: tuple[str, ...]) -> str | None:
    if value is None:
        return None
    needle = str(value).strip().lower()
    if not needle or needle == "null":
        return None
    for category in categories:
        candidate = category.lower()
        if needle == candidate or needle in candidate or candidate in needle:
            return category
    return None


def _answer_context_block(index: int, result: Any) -> str:
    article_id = str(getattr(result, "article_id", "") or "").strip()
    title = str(getattr(result, "title", "") or "").strip()
    date = str(getattr(result, "date", "") or "").strip()
    url = str(getattr(result, "url", "") or "").strip()
    categories = str(getattr(result, "categories", "") or "").strip()
    tags = str(getattr(result, "tags", "") or "").strip()
    excerpts = []
    for match in getattr(result, "matches", []) or []:
        excerpt = str(getattr(match, "excerpt", "") or "").strip()
        if excerpt:
            excerpts.append(excerpt)
    excerpt_text = "\n".join(f"- {excerpt}" for excerpt in excerpts[:6])
    content = "\n".join(
        part
        for part in [
            f"[{index}] Article ID: {article_id}",
            f"Title: {title}",
            f"Date: {date}",
            f"URL: {url}",
            f"Categories: {categories}" if categories else "",
            f"Tags: {tags}" if tags else "",
            "Relevant excerpts:",
            excerpt_text or "- No excerpt available.",
        ]
        if part
    )
    return f"=== SOURCE [{index}] START ===\n{content}\n=== SOURCE [{index}] END ==="


def _atomic_claim_units(brief: str) -> list[dict[str, Any]]:
    """Return cited factual sentences as addressable claim units."""

    units: list[dict[str, Any]] = []
    for paragraph in re.split(r"\n+", str(brief or "")):
        paragraph = paragraph.strip()
        if not paragraph or paragraph.startswith("#"):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            sentence = sentence.strip()
            citations = [
                int(value)
                for group in re.findall(r"\[((?:\s*\d+\s*,?)+)\]", sentence)
                for value in re.findall(r"\d+", group)
            ]
            if not citations:
                continue
            units.append(
                {
                    "id": len(units) + 1,
                    "text": sentence,
                    "citations": sorted(set(citations)),
                }
            )
    return units[:60]


def _parse_json_object(value: str) -> dict[str, Any]:
    """Parse a JSON object from a plain or fenced model response."""

    text = str(value or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model response did not contain a JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("model response JSON was not an object")
    return payload


def _parse_claim_audit_response(value: str) -> dict[str, Any]:
    """Parse the tagged audit format, retaining legacy JSON compatibility."""

    text = str(value or "").strip()
    audit_match = re.search(
        r"<claim_audit>\s*(.*?)\s*</claim_audit>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    brief_match = re.search(
        r"<revised_brief>\s*(.*?)\s*</revised_brief>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if audit_match and brief_match:
        claims = json.loads(audit_match.group(1))
        if not isinstance(claims, list):
            raise ValueError("claim audit block was not a JSON array")
        return {
            "claims": claims,
            "revised_brief": brief_match.group(1).strip(),
        }
    return _parse_json_object(text)


def _sanitize_citations(answer: str, source_count: int) -> str:
    """Keep only bracketed numeric citations that refer to supplied sources."""

    def replace(match: re.Match[str]) -> str:
        raw_numbers = [part.strip() for part in match.group(1).split(",")]
        valid_numbers: list[str] = []
        seen: set[str] = set()
        for raw in raw_numbers:
            if not raw.isdigit():
                continue
            value = int(raw)
            if 1 <= value <= source_count and raw not in seen:
                seen.add(raw)
                valid_numbers.append(raw)
        return f"[{', '.join(valid_numbers)}]" if valid_numbers else ""

    cleaned = re.sub(
        r"\[\s*sources?\s*:?#?\s*((?:\d+\s*,?\s*)+)\]",
        r"[\1]",
        answer,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"【\s*((?:\d+\s*,?\s*)+)】",
        r"[\1]",
        cleaned,
    )
    cleaned = re.sub(r"\[((?:\s*\d+\s*,?)+)\]", replace, cleaned)
    # Some models place citations immediately after sentence punctuation
    # ("Fact. [1]"). Normalize that common style to the validator's canonical
    # form ("Fact [1].") without changing the cited source numbers.
    cleaned = re.sub(
        r"([.!?])\s*((?:\[\s*\d+(?:\s*,\s*\d+)*\s*\]\s*)+)(?=\s|$)",
        r" \2\1",
        cleaned,
    )
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return cleaned.strip()


def _prose_word_count(answer: str) -> int:
    prose = re.sub(r"\[((?:\s*\d+\s*,?)+)\]", "", answer)
    return len(re.findall(r"\b[\w'-]+\b", prose))


def _verification_depth_floor(draft_words: int) -> int:
    if draft_words >= 800:
        return max(650, min(800, int(draft_words * 0.65)))
    return max(250, int(draft_words * 0.65))


def _strip_thinking(answer: str) -> str:
    """Remove Qwen reasoning blocks and leaked chat-template sentinels."""

    cleaned = re.sub(r"<think>.*?</think>", "", answer, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<\|(?:im_end|endoftext)\|>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _uncited_sentences(answer: str) -> list[str]:
    """Return prose sentences that do not end in a numeric source citation."""

    citation_end = re.compile(
        r"(?:\[\s*\d+(?:\s*,\s*\d+)*\s*\]\s*)+[.!?][\"']?$"
    )
    sentences: list[str] = []
    for paragraph in re.split(r"\n+", answer):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            sentence = sentence.strip()
            if sentence and not citation_end.search(sentence):
                sentences.append(sentence)
    return sentences


def _prune_uncited_sentences(answer: str) -> str:
    """Remove uncited prose while preserving paragraph structure and cited facts."""

    citation_end = re.compile(
        r"(?:\[\s*\d+(?:\s*,\s*\d+)*\s*\]\s*)+[.!?][\"']?$"
    )
    paragraphs: list[str] = []
    for paragraph in re.split(r"\n+", answer):
        cited = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", paragraph.strip())
            if sentence.strip() and citation_end.search(sentence.strip())
        ]
        if cited:
            paragraphs.append(" ".join(cited))
    return "\n\n".join(paragraphs)


def _prune_uncited_paragraphs(answer: str) -> str:
    """Preserve long-form paragraphs containing validated numeric citations."""

    citation = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
    return "\n\n".join(
        paragraph.strip()
        for paragraph in re.split(r"\n+", answer)
        if paragraph.strip() and citation.search(paragraph)
    )


def _uncited_paragraphs(answer: str) -> list[str]:
    """Return long-form paragraphs that contain no numeric source citation."""

    citation = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n+", answer)
        if paragraph.strip() and not citation.search(paragraph)
    ]


def _dedupe_terms(terms: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for term in terms:
        normalized = " ".join(_stem_tokens(str(term)))
        if len(normalized) <= 2 or normalized in STOPWORDS or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _stem_phrase(value: str) -> str:
    return " ".join(_stem_tokens(value))


def _stem_tokens(value: str) -> list[str]:
    return [_stem_token(token.strip("'")) for token in TOKEN_RE.findall(value.lower())]


def _stem_token(token: str) -> str:
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token
