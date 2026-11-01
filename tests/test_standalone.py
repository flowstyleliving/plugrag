"""Portable ingestion and integration checks, isolated from the production archive."""

from dataclasses import replace
from pathlib import Path
import json
import subprocess
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from plugrag import Document, RAG, Settings
from plugrag.cli import main, read_documents
from plugrag.generation import CitedGenerator
from plugrag.server import create_app


@pytest.fixture
def rag(tmp_path, fake_model):
    pytest.importorskip("chromadb")
    return RAG(Settings(data_dir=tmp_path), model=fake_model)


def test_ingest_search_update_and_restart(rag, fake_model):
    documents = [
        Document(
            id="1",
            text="The school board approved an education budget.",
            title="School budget",
            categories=["Education"],
            date="2024-01-01",
        ),
        Document(
            id="jail",
            text="The sheriff reported on prison oversight.",
            categories=["Justice"],
            date="2025-01-01",
        ),
    ]
    assert rag.ingest(documents)["articles_embedded"] == 2
    initial_count = rag.collection.count()
    assert rag.ingest(documents)["articles_skipped"] == 2
    assert rag.collection.count() == initial_count
    assert rag.search("school board budget")[0]["id"] == "1"
    assert rag.search("school", category="Justice")[0]["id"] == "jail"
    assert rag.search("budget", date_to="2023-12-31") == []
    # Same ID and chunk count: remove old lexical content and update metadata.
    rag.ingest(
        [
            replace(
                documents[0],
                text="Storm drainage budget repairs.",
                title="Storm repairs",
                categories=["Infrastructure"],
            )
        ]
    )
    assert rag.collection.count() == initial_count
    restarted = RAG(rag.settings, model=fake_model)
    hits = restarted.search("storm drainage", category="Infrastructure")
    assert hits[0]["id"] == "1"
    assert hits[0]["title"] == "Storm repairs"
    assert "school" not in hits[0]["matches"][0]["excerpt"].lower()
    assert (rag.data_dir / "search_fts.sqlite3").is_file()


def test_incompatible_stores_are_rejected(rag):
    rag.ingest([Document(id="a", text="School budget")])
    with pytest.raises(ValueError, match="settings differ"):
        RAG(replace(rag.settings, embedding_model="different-model"))
    with pytest.raises(ValueError, match="settings differ"):
        RAG(replace(rag.settings, chunk_size=100))
    manifest_path = rag.data_dir / "manifest.json"
    data = json.loads(manifest_path.read_text())
    del data["rag_settings"]
    manifest_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="existing archive"):
        RAG(rag.settings)


@pytest.mark.parametrize(
    "values",
    [
        {"id": "../escape", "text": "hello"},
        {"id": "..", "text": "hello"},
        {"id": "a", "text": " "},
        {"id": "a", "text": 123},
        {"id": "a", "text": "hello", "date": "yesterday"},
        {"id": "a", "text": "hello", "categories": "Education"},
    ],
)
def test_document_validation(values):
    with pytest.raises(ValueError):
        Document(**values)


def test_duplicate_ids_fail_before_writing(rag):
    with pytest.raises(ValueError, match="duplicate"):
        rag.ingest([Document(id="a", text="one"), Document(id="a", text="two")])
    assert not (rag.data_dir / "articles").exists()


def test_failed_embedding_is_retryable(rag):
    class FailingModel:
        def encode_texts(self, texts):
            raise RuntimeError("injected model failure")

    working_model = rag.model
    rag.model = FailingModel()
    with pytest.raises(RuntimeError, match="failed embedding"):
        rag.ingest([Document(id="a", text="School board budget")])
    rag.model = working_model
    assert (
        rag.ingest([Document(id="a", text="School board budget")])["articles_embedded"]
        == 1
    )
    assert rag.search("school")[0]["id"] == "a"


def test_cli_loaders_and_errors(tmp_path, capsys):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "policy.md").write_text("School budget policy")
    (folder / "ignore.bin").write_bytes(b"\x00")
    first = read_documents(folder)
    assert len(first) == 1
    assert first[0].id == read_documents(folder)[0].id
    source = tmp_path / "input.jsonl"
    source.write_text('{"id":"a","text":"hello"}\n\n')
    assert read_documents(source)[0].text == "hello"
    assert main(["ingest", str(tmp_path / "missing.json")]) == 1
    assert "plugrag:" in capsys.readouterr().err


def test_api_mount_auth_validation_and_answers(rag):
    rag.ingest([Document(id="a", text="School budget approved.")])
    parent = FastAPI()
    parent.mount("/rag", create_app(rag, api_key="secret"))
    with TestClient(parent) as client:
        assert client.get("/rag/health").json()["status"] == "ok"
        assert client.post("/rag/search", json={"query": "school"}).status_code == 401
        headers = {"Authorization": "Bearer secret"}
        assert (
            client.post(
                "/rag/search", headers=headers, json={"query": "school"}
            ).json()["results"][0]["id"]
            == "a"
        )
        for payload in (
            {"query": " "},
            {"query": "school", "date_from": "oops"},
            {"query": "school", "top_k": 101},
            {"query": "school", "date_from": "2025-01-01", "date_to": "2024-01-01"},
        ):
            assert (
                client.post("/rag/search", headers=headers, json=payload).status_code
                == 422
            )
        assert (
            client.post(
                "/rag/answer", headers=headers, json={"query": "school"}
            ).status_code
            == 501
        )

        class Generator:
            def generate(self, query, sources):
                assert sources[0]["index"] == 1
                return "The school budget was approved [1]."

        rag.generator = Generator()
        response = client.post("/rag/answer", headers=headers, json={"query": "school"})
        assert response.status_code == 200
        assert response.json()["sources"][0]["id"] == "a"

        # Internal/provider exception text must not reach API responses.
        class BrokenGenerator:
            def generate(self, query, sources):
                raise RuntimeError("private-provider-key")

        rag.generator = BrokenGenerator()
        response = client.post("/rag/answer", headers=headers, json={"query": "school"})
        assert response.status_code == 503
        assert "private-provider-key" not in response.text


def test_generator_keeps_citation_safeguards():
    from types import SimpleNamespace

    class Completions:
        def create(self, **kwargs):
            assert "for the user" in kwargs["messages"][0]["content"]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="The school budget was approved [1]."
                        )
                    )
                ]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    generator = CitedGenerator(
        model="test", base_url="http://localhost/v1", client=client
    )
    sources = [
        {
            "id": "a",
            "title": "Budget",
            "date": "",
            "url": "",
            "categories": "",
            "tags": "",
            "score": 1.0,
            "matches": [
                {
                    "score": 1.0,
                    "excerpt": "The school budget was approved.",
                    "chunk_index": 0,
                }
            ],
        }
    ]
    assert (
        generator.generate("school?", sources) == "The school budget was approved [1]."
    )


def test_settings_ignore_lens_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LENS_RAG_DATA_DIR", "/private/lens/archive")
    monkeypatch.setenv("LENS_RAG_EMBEDDING_MODEL", "private-model")
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    assert Settings.from_env().data_dir == tmp_path
    assert Settings.from_env().embedding_model == "all-MiniLM-L6-v2"


def test_library_import_does_not_load_dotenv(tmp_path):
    # Check in a fresh interpreter, so previous legacy API imports cannot mask it.
    code = "import os; before = dict(os.environ); import plugrag; assert dict(os.environ) == before"
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
