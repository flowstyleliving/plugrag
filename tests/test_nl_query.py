from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import pytest

from lens_rag.nl_query import (
    DEFAULT_LOCAL_LLM_BASE_URL,
    AnswerGenerator,
    DeepSeekAnswerGenerator,
    DeepSeekProAnswerGenerator,
    OpusAnswerGenerator,
    OpenAIAnswerGenerator,
    QueryRewriter,
    _sanitize_citations,
)
from lens_rag.query_engine import ArticleResult, ChunkMatch


class FakeCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type(
            "Response",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type(
                                "Message",
                                (),
                                {
                                    "content": (
                                        "<think>private reasoning</think>"
                                        "Schools were discussed using valid source [1] "
                                        "and not invalid source [3], as documented [1]."
                                        "<|im_end|>"
                                    )
                                },
                            )()
                        },
                    )()
                ]
            },
        )()


class FakeClient:
    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = type(
            "Chat",
            (),
            {"completions": self.completions},
        )()


def test_openai_uses_responses_and_keeps_mapping_on_openai() -> None:
    import httpx
    from openai import OpenAI

    calls = []

    def respond(request):
        calls.append((request.url, json.loads(request.content)))
        return httpx.Response(200, json={
            "id": "resp_test", "object": "response", "created_at": 0,
            "status": "completed", "model": "gpt-6.1-sol",
            "output": [{"id": "msg_test", "type": "message", "role": "assistant",
                        "status": "completed", "content": [{"type": "output_text",
                        "text": "The board discussed school funding [1].", "annotations": []}]}],
        })

    client = OpenAI(api_key="test-openai", http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    generator = OpenAIAnswerGenerator(api_key="test-openai", client=client)
    assert generator._complete([{"role": "user", "content": "Use the sources"}]).endswith("[1].")
    mapper = generator._map_generator(max_tokens=1100)
    mapper._complete([{"role": "user", "content": "Audit citations"}])
    assert mapper.provider == "openai"
    assert mapper.api_key == "test-openai"
    for url, body in calls:
        assert str(url) == "https://api.openai.com/v1/responses"
        assert body["model"] == "gpt-6.1-sol"
        assert body["store"] is False
        assert "temperature" not in body
        assert "max_tokens" not in body
        assert "extra_body" not in body
    assert calls[0][1]["reasoning"] == {"effort": "medium"}
    assert calls[1][1]["reasoning"] == {"effort": "low"}
    assert calls[1][1]["max_output_tokens"] >= 5000
    # Citation recovery must produce one bounded brief, not restart research.
    result = ArticleResult(
        article_id="1", title="School funding", date="2024-01-15",
        url="https://example.test/1", categories="Education", tags="Schools",
        score=0.9, matches=[ChunkMatch(score=0.9, excerpt="The board discussed school funding.", chunk_index=0)],
    )
    before = len(calls)
    assert mapper.generate("school funding", [result], draft_answer="A draft").endswith("[1].")
    assert len(calls) == before + 1


@pytest.mark.parametrize("status,text", [("incomplete", "Partial claim [1]."), ("completed", "")])
def test_openai_rejects_incomplete_or_empty_output(status, text) -> None:
    from types import SimpleNamespace

    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(status=status, output_text=text)))
    with pytest.raises(RuntimeError):
        OpenAIAnswerGenerator(client=client)._complete([])


def test_openai_error_does_not_expose_credentials() -> None:
    import httpx
    from openai import OpenAI

    def reject(request):
        return httpx.Response(401, json={"error": {"message": "Invalid key test-secret", "type": "authentication_error"}})

    client = OpenAI(api_key="test-secret", http_client=httpx.Client(transport=httpx.MockTransport(reject)))
    with pytest.raises(RuntimeError, match="HTTP 401") as error:
        OpenAIAnswerGenerator(client=client)._complete([])
    assert "test-secret" not in str(error.value)


def test_openai_does_not_inherit_other_provider_credentials(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LENS_RAG_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LENS_RAG_DEEPSEEK_API_KEY", "deepseek-only")
    monkeypatch.setenv("LENS_RAG_LLM_API_KEY", "legacy-only")
    generator = OpenAIAnswerGenerator()
    assert generator.api_key == ""
    assert not generator.supports_answering()


def test_sanitize_citations_normalizes_common_model_formats() -> None:
    answer = "One claim [Source 1]. Another claim 【2】. Invalid [Source 9]."

    assert _sanitize_citations(answer, 2) == (
        "One claim [1]. Another claim [2]. Invalid."
    )


def test_answer_generator_uses_only_returned_citation_indices() -> None:
    result = ArticleResult(
        article_id="1",
        title="School Board budget",
        date="2024-01-15T10:00:00",
        url="https://example.test/1",
        categories="Education",
        tags="Schools",
        score=0.91,
        matches=[
            ChunkMatch(
                score=0.91,
                excerpt="The school board discussed education funding.",
                chunk_index=0,
            )
        ],
    )
    client = FakeClient()
    generator = AnswerGenerator(client=client, provider="local")

    answer = generator.generate("school budget", [result])

    assert "[1]" in answer
    assert "[3]" not in answer
    assert "private reasoning" not in answer
    assert "im_end" not in answer
    request = client.completions.calls[0]
    assert request["max_tokens"] == 900
    assert request["temperature"] == 0.3
    assert request["top_p"] == 0.8
    assert request["messages"][1]["content"].startswith("/no_think")
    assert "=== SOURCE [1] START ===" in request["messages"][1]["content"]
    assert "=== SOURCE [1] END ===" in request["messages"][1]["content"]


def test_local_answer_generator_defaults_to_dedicated_mlx_endpoint(monkeypatch) -> None:
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.delenv("LENS_RAG_LLM_BASE_URL", raising=False)
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    generator = AnswerGenerator(provider="mlx", api_key=None)

    assert generator.supports_answering() is True
    assert generator.base_url == DEFAULT_LOCAL_LLM_BASE_URL
    generator._openai_client()
    assert captured == {
        "api_key": "local",
        "base_url": "http://127.0.0.1:5368/v1",
    }


def test_mlx_launcher_port_matches_client_default() -> None:
    launcher = Path(__file__).resolve().parents[1] / "scripts" / "start_mlx.sh"
    default_port = urlparse(DEFAULT_LOCAL_LLM_BASE_URL).port

    assert default_port == 5368
    assert f'MLX_PORT="${{MLX_PORT:-{default_port}}}"' in launcher.read_text(
        encoding="utf-8"
    )


def test_openai_answer_generator_still_requires_api_key() -> None:
    generator = AnswerGenerator(provider="openai", api_key="")

    assert generator.supports_answering() is False


def test_answer_generator_repairs_uncited_sentences() -> None:
    responses = iter(
        [
            "The jail remained understaffed. Conditions regressed [1].",
            "The jail remained understaffed [1]. Conditions regressed [1].",
        ]
    )

    class RepairCompletions:
        def __init__(self) -> None:
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            content = next(responses)
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {"message": type("Message", (), {"content": content})()},
                        )()
                    ]
                },
            )()

    completions = RepairCompletions()
    client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": completions})()},
    )()
    result = ArticleResult(
        article_id="1",
        title="Jail oversight",
        date="2023-01-01",
        url="https://example.test/1",
        categories="Criminal Justice",
        tags="jail",
        score=0.9,
        matches=[ChunkMatch(score=0.9, excerpt="The jail remained understaffed and conditions regressed.", chunk_index=0)],
    )

    answer = AnswerGenerator(client=client, provider="local").generate(
        "What changed at the jail?",
        [result],
    )

    assert answer == "The jail remained understaffed [1]. Conditions regressed [1]."
    assert len(completions.calls) == 2
    assert "Brief to repair" in completions.calls[1]["messages"][1]["content"]


def test_answer_generator_prunes_uncited_sentences_after_failed_repair() -> None:
    responses = iter(
        [
            "Broad unsupported transition. The jail remained understaffed [1].",
            "Broad unsupported transition. The jail remained understaffed [1].",
        ]
    )

    class UnrepairedCompletions:
        def create(self, **kwargs):
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message", (), {"content": next(responses)}
                                )()
                            },
                        )()
                    ]
                },
            )()

    client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": UnrepairedCompletions()})()},
    )()
    result = ArticleResult(
        article_id="1",
        title="Jail oversight",
        date="2023-01-01",
        url="https://example.test/1",
        categories="Criminal Justice",
        tags="jail",
        score=0.9,
        matches=[
            ChunkMatch(
                score=0.9,
                excerpt="The jail remained understaffed.",
                chunk_index=0,
            )
        ],
    )

    answer = AnswerGenerator(client=client, provider="local").generate(
        "What changed at the jail?",
        [result],
    )

    assert answer == "The jail remained understaffed [1]."


def test_deepseek_refines_local_draft_through_direct_api() -> None:
    class DeepSeekCompletions:
        def __init__(self) -> None:
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message",
                                    (),
                                    {"content": "The board reviewed the budget [1]."},
                                )()
                            },
                        )()
                    ]
                },
            )()

    completions = DeepSeekCompletions()
    client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": completions})()},
    )()
    result = ArticleResult(
        article_id="1",
        title="School Board budget",
        date="2024-01-15T10:00:00",
        url="https://example.test/1",
        categories="Education",
        tags="Schools",
        score=0.91,
        matches=[
            ChunkMatch(
                score=0.91,
                excerpt="The school board reviewed the proposed budget.",
                chunk_index=0,
            )
        ],
    )
    generator = DeepSeekAnswerGenerator(
        api_key="test-key",
        base_url="https://api.deepseek.com/v1",
        client=client,
    )

    answer = generator.generate(
        "What did the school board review?",
        [result],
        "The board approved the budget [1].",
    )

    assert answer == "The board reviewed the budget [1]."
    request = completions.calls[0]
    assert request["model"] == "deepseek-v4-flash"
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "Draft A:\nThe board approved the budget [1]." in request["messages"][1]["content"]
    assert "=== SOURCE [1] START ===" in request["messages"][1]["content"]


def test_deepseek_synthesizes_without_local_draft() -> None:
    class DeepSeekCompletions:
        def __init__(self) -> None:
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            message = type("Message", (), {"content": "The board reviewed the budget. [1]"})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    completions = DeepSeekCompletions()
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    result = ArticleResult(
        article_id="1", title="School Board budget", date="2024-01-15T10:00:00",
        url="https://example.test/1", categories="Education", tags="Schools", score=0.91,
        matches=[ChunkMatch(score=0.91, excerpt="The school board reviewed the proposed budget.", chunk_index=0)],
    )
    generator = DeepSeekAnswerGenerator(api_key="test-key", client=client)

    answer = generator.generate("What did the board review?", [result])

    assert answer == "The board reviewed the budget [1]."
    prompt = completions.calls[0]["messages"][1]["content"]
    assert "Create a standalone research brief directly" in prompt
    assert "Draft A:" not in prompt


def test_deepseek_pro_uses_thinking_and_deeper_prompt() -> None:
    class Completions:
        def __init__(self) -> None:
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            message = type("Message", (), {"content": "The board reviewed the budget [1]."})()
            return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()

    completions = Completions()
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    result = ArticleResult(
        article_id="1", title="School Board budget", date="2024-01-15T10:00:00",
        url="https://example.test/1", categories="Education", tags="Schools", score=0.91,
        matches=[ChunkMatch(score=0.91, excerpt="The school board reviewed the proposed budget.", chunk_index=0)],
    )

    generator = DeepSeekProAnswerGenerator(api_key="test-key", client=client)
    answer = generator.generate(
        "What did the board review?", [result]
    )

    assert answer == "The board reviewed the budget [1]."
    pro_requests = [
        request
        for request in completions.calls
        if request["model"] == "deepseek-v4-pro"
    ]
    assert len(pro_requests) == 1
    request = pro_requests[0]
    assert request["model"] == "deepseek-v4-pro"
    assert request["extra_body"] == {"thinking": {"type": "enabled"}}
    assert request["reasoning_effort"] == "high"
    assert "900-1,400 words" in request["messages"][1]["content"]
    assert "Contradictions" in request["messages"][1]["content"] or "contradictions" in request["messages"][1]["content"]
    assert any(
        "chronology" in call["messages"][1]["content"]
        for call in completions.calls
        if call["model"] == "deepseek-v4-flash"
    )
    scoped_requests = [
        call
        for call in completions.calls
        if "Scoped evidence cards:" in call["messages"][1]["content"]
    ]
    assert len(scoped_requests) == 3
    assert "Facet-scoped section drafts:" in request["messages"][1]["content"]
    assert generator.last_diagnostics["section_draft_count"] == 3
    assert "section-scoped drafting" in generator.last_diagnostics["pipeline_stages"]


def test_in_depth_pipeline_retains_cited_sections_when_reduce_fails() -> None:
    result = ArticleResult(
        article_id="1",
        title="School Board budget",
        date="2024-01-15T10:00:00",
        url="https://example.test/1",
        categories="Education",
        tags="Schools",
        score=0.91,
        matches=[
            ChunkMatch(
                score=0.91,
                excerpt="The school board reviewed the proposed budget.",
                chunk_index=0,
            )
        ],
    )
    generator = DeepSeekProAnswerGenerator(api_key="test-key", client=object())
    generator._map_evidence_batches = lambda query, batches: ["Evidence memo [1]."]
    generator._build_evidence_outline = (
        lambda query, notes, research_plan=None: "Evidence outline"
    )
    generator._draft_evidence_sections = (
        lambda query, ledger, notes, outline: [
            "## Findings\n\nThe board reviewed the budget [1].",
            "## Accountability\n\nOfficials discussed the proposal [1].",
        ]
    )
    generator._reduce_evidence = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("temporary provider error")
    )
    generator._verify_evidence_brief = lambda query, results, brief: brief

    answer = generator.generate_research("What did the board review?", [result])

    assert "## Findings" in answer
    assert "## Accountability" in answer
    assert generator.last_diagnostics["reduce_mode"] == "cited section-draft fallback"


def test_atomic_claim_audit_applies_supported_json_revision() -> None:
    result = ArticleResult(
        article_id="1",
        title="School Board budget",
        date="2024-01-15T10:00:00",
        url="https://example.test/1",
        categories="Education",
        tags="Schools",
        score=0.91,
        matches=[
            ChunkMatch(
                score=0.91,
                excerpt="The school board reviewed the proposed budget.",
                chunk_index=0,
            )
        ],
    )
    revised = "The school board reviewed the proposed budget [1]."

    class Auditor:
        def _complete(self, messages):
            assert "Audit every numbered claim" in messages[1]["content"]
            audit = json.dumps(
                [
                    {
                        "id": 1,
                        "status": "partial",
                        "issue": "The source says proposed budget.",
                    }
                ]
            )
            return (
                f"<claim_audit>{audit}</claim_audit>\n"
                f"<revised_brief>{revised}</revised_brief>"
            )

    generator = DeepSeekProAnswerGenerator(api_key="test-key", client=object())
    generator._map_generator = lambda max_tokens=1100: Auditor()

    answer = generator._audit_atomic_claim_support(
        "What did the board review?",
        [result],
        "The school board approved the budget [1].",
    )

    assert answer == revised
    assert generator._claim_audit_diagnostics == {
        "status": "applied",
        "claim_count": 1,
        "supported_count": 0,
        "partial_count": 1,
        "unsupported_count": 0,
    }


@pytest.mark.parametrize("effort", [None, "medium"])
def test_opus_calls_anthropic_directly(monkeypatch, effort) -> None:
    monkeypatch.delenv("LENS_RAG_OPUS_EFFORT", raising=False)
    if effort:
        monkeypatch.setenv("LENS_RAG_OPUS_EFFORT", effort)
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "The board reviewed the budget [1].",
                    }
                ]
            }

    class Transport:
        def __init__(self) -> None:
            self.calls = []

        def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return Response()

    transport = Transport()
    generator = OpusAnswerGenerator(
        api_key="test-anthropic-key",
        client=transport,
    )

    answer = generator._complete(
        [
            {"role": "system", "content": "Use only supplied evidence."},
            {"role": "user", "content": "Synthesize the reporting."},
        ]
    )

    assert answer == "The board reviewed the budget [1]."
    url, request = transport.calls[0]
    assert url == "https://api.anthropic.com/v1/messages"
    assert request["headers"]["x-api-key"] == "test-anthropic-key"
    assert set(request["headers"]) == {
        "x-api-key",
        "anthropic-version",
        "content-type",
    }
    assert request["json"]["model"] == "claude-opus-5"
    if effort:
        assert request["json"]["output_config"] == {"effort": effort}
    else:
        assert "output_config" not in request["json"]
    assert "temperature" not in request["json"]
    assert request["json"]["system"] == "Use only supplied evidence."
    assert request["json"]["messages"] == [
        {"role": "user", "content": "Synthesize the reporting."}
    ]


def test_opus_map_budget_leaves_room_for_adaptive_thinking(
    monkeypatch,
) -> None:
    monkeypatch.delenv("LENS_RAG_OPUS_MAP_MAX_TOKENS", raising=False)
    monkeypatch.setenv("LENS_RAG_OPUS_EFFORT", "medium")
    monkeypatch.setenv("LENS_RAG_OPUS_MAP_EFFORT", "low")
    generator = OpusAnswerGenerator(
        api_key="test-anthropic-key",
        client=object(),
    )

    mapper = generator._map_generator(max_tokens=1800)

    assert mapper.max_tokens == 5000
    assert mapper.effort == "low"
    assert generator._anthropic_generator(max_tokens=7000).effort == "medium"


def test_opus_surfaces_anthropic_request_error_without_exposing_credentials() -> None:
    class Response:
        def raise_for_status(self) -> None:
            raise RuntimeError("400 Bad Request")

        def json(self) -> dict:
            return {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "Enable 30-day data retention for this workspace.",
                },
            }

    class Transport:
        def post(self, url, **kwargs):
            return Response()

    generator = OpusAnswerGenerator(
        api_key="private-test-key",
        client=Transport(),
    )

    with pytest.raises(
        RuntimeError,
        match="Enable 30-day data retention for this workspace",
    ) as exc:
        generator._complete(
            [{"role": "user", "content": "Synthesize the reporting."}]
        )

    assert "private-test-key" not in str(exc.value)


def test_opus_retries_transient_anthropic_failures(monkeypatch) -> None:
    class Response:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"{self.status_code} upstream error")

        def json(self) -> dict:
            if self.status_code >= 400:
                return {
                    "error": {
                        "message": "An error occurred while processing your request."
                    }
                }
            return {
                "content": [
                    {"type": "text", "text": "The cited synthesis recovered [1]."}
                ]
            }

    class Transport:
        def __init__(self) -> None:
            self.calls = 0

        def post(self, url, **kwargs):
            self.calls += 1
            return Response(529 if self.calls < 3 else 200)

    transport = Transport()
    monkeypatch.setattr("lens_rag.nl_query.time.sleep", lambda seconds: None)
    generator = OpusAnswerGenerator(
        api_key="test-anthropic-key",
        client=transport,
    )

    answer = generator._complete(
        [{"role": "user", "content": "Synthesize the reporting."}]
    )

    assert answer == "The cited synthesis recovered [1]."
    assert transport.calls == 3


def test_local_rewriter_extracts_dates_aliases_without_api_call() -> None:
    rewriter = QueryRewriter(api_key="present-but-local-by-default")

    result = rewriter.rewrite(
        "what has The Lens reported about jail conditions since 2020?",
        ["Criminal Justice", "Environment"],
    )

    assert result.used_llm is False
    assert result.rewritten_query == "jail conditions"
    assert result.date_from == "2020-01-01"
    assert result.date_to is None
    assert "detention" in result.expansion_terms


def test_local_rewriter_extracts_category_and_bounded_years() -> None:
    result = QueryRewriter().rewrite(
        "stories about Criminal Justice between 2019 and 2022",
        ["Criminal Justice", "Schools"],
    )

    assert result.rewritten_query == "Criminal Justice"
    assert result.category == "Criminal Justice"
    assert result.date_from == "2019-01-01"
    assert result.date_to == "2022-12-31"


def test_local_rewriter_handles_prior_year_language() -> None:
    result = QueryRewriter().rewrite("jail oversight articles prior 2019")

    assert result.rewritten_query == "jail oversight"
    assert result.date_from is None
    assert result.date_to == "2018-12-31"
