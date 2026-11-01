# PlugRAG

A standalone RAG system extracted from Lens RAG. Clone this repository, ingest your
own documents, and use it as a Python library, a CLI, or an HTTP service. Retrieval
runs locally with sentence-transformer embeddings, persistent Chroma vectors, and
SQLite FTS5/BM25 hybrid search. No Lens archive, WordPress site, or paid API is needed.
Answer generation is optional and supports OpenAI-compatible local or remote servers.

## Quick start

Use Python 3.12 on macOS or Linux (package supports Python 3.10–3.12). For Windows,
use the Docker instructions below; the existing manifest writer uses POSIX locking.

Clone the repository and install:

```bash
git clone https://github.com/flowstyleliving/plugrag.git
cd plugrag
```

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[server,llm]'
plugrag ingest examples/documents.json
plugrag search 'How many vacation days do employees receive?'
plugrag serve
```

The first ingestion downloads the embedding model (about 90 MB). Later runs can use
the cached model offline. The default store is `.rag-data/`, separate from the Lens
archive's `data/`. Everything needed for retrieval persists there across restarts.

Open [interactive API docs](http://127.0.0.1:8000/docs), or call:

```bash
curl http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"vacation days", "top_k":5}'
```

A second identical ingestion skips unchanged documents. A changed document with the
same ID replaces its old chunks. Ingestion is an upsert: removing a document from an
input file does not delete it from the store. Use a new data directory to replace an
entire collection.

## Embed in your application

The core install is `python -m pip install .`. It does not require FastAPI or an LLM SDK.

```python
from pathlib import Path
from plugrag import Document, RAG, Settings

rag = RAG(Settings(data_dir=Path("./my-search-data")))
rag.ingest([
    Document(
        id="refund-policy",
        title="Refund policy",
        text="Customers may request a refund within 30 days of purchase.",
        url="https://example.com/refunds",
        categories=["Support"],
    ),
])
hits = rag.search("When can I request a refund?", top_k=5)
print(hits[0]["id"], hits[0]["matches"][0]["excerpt"])
```

`search()` returns JSON-compatible dictionaries with `id`, `title`, `date`, `url`,
`categories`, `tags`, `score`, `score_info`, and `matches`. Each match includes its
excerpt, chunk index and score. Scores rank results; they are not probabilities.
Optional filters are `category`, `date_from`, and `date_to`.

Adapt any database, document loader or scraper to `Document` objects. You may inject
`RAG(..., model=your_model, collection=your_collection, generator=your_generator)`.
The embedding model implements `encode_texts(list[str]) -> list[list[float]]` and may
provide `encode_query(str) -> list[float]`. Collection injection expects the Chroma
collection interface. Use the same embedding implementation for ingestion and query.
A custom generator implements `generate(query, sources) -> str`; `sources` contain
numbered search results. Custom generators own their grounding/validation policy.

To mount the HTTP API in an existing FastAPI app (install the `server` extra):

```python
from fastapi import FastAPI
from plugrag import RAG
from plugrag.server import create_app

app = FastAPI()
app.mount("/rag", create_app(RAG(), api_key="your-server-token"))
```

The service exposes `GET /health`, `POST /search`, and `POST /answer`. Answer requests
use the search request fields. `/answer` returns 501 until generation is configured.
Ingestion stays in the CLI/SDK so the public service has no write endpoint.

## Bring your own data

```bash
plugrag ingest path/to/documents.json
plugrag ingest path/to/documents.jsonl
plugrag ingest path/to/text-and-markdown-directory
plugrag --data-dir ./another-store search 'your question'
```

JSON input is an array of objects; JSONL is one object per line. Only `id` and `text`
are required. Optional fields are `title`, `url`, `date`, `categories`, and `tags`.
Dates accept `YYYY-MM-DD`, ISO timestamps, `YYYY/MM/DD`, or `MM/DD/YYYY`.
IDs must start with a letter or digit and contain only letters, digits, `.`, `_`, or
`-` (128 characters maximum). Categories and tags are lists of strings.

Directory ingestion recursively reads UTF-8 `.txt` and `.md` files. IDs derive from
relative file paths, so repeat imports update the same files. JSON/JSONL lets you
supply stable IDs across renames. PDF/Office parsing belongs in your source adapter;
pass extracted plain text to `Document`.

## Optional cited answers

Install `.[llm]` and configure a running OpenAI-compatible server:

```bash
export RAG_LLM_BASE_URL=http://127.0.0.1:8001/v1
export RAG_LLM_MODEL=your-served-model-name
# For a paid remote endpoint, also set RAG_LLM_API_KEY.
plugrag answer 'How much vacation do employees get?'
```

For Docker, use a reachable server address such as `host.docker.internal` on Docker
Desktop; container `127.0.0.1` points at the container itself.

For the SDK, opt in explicitly:

```python
from plugrag import RAG
from plugrag.generation import CitedGenerator

rag = RAG(generator=CitedGenerator(
    model="your-served-model-name",
    base_url="http://127.0.0.1:8001/v1",
))
response = rag.answer("How much vacation do employees get?")
print(response["answer"], response["sources"])
```

The built-in generator retains the shared citation repair and validation logic:
numbered citations, removal of out-of-range references, and repair or pruning of
uncited sentences. These are structural checks, not proof that a claim is true.
The deeper Lens research, evidence-ledger and atomic claim audit workflows remain in
the original application; the generic answer path produces a short cited brief.

## Configuration and operations

[Example environment settings](examples/plugrag.env.example) list the supported
`RAG_*` variables. Export them yourself or pass an env file to Docker. PlugRAG never
automatically reads repository `.env` files or personal home-directory credentials,
and ignores `LENS_RAG_*` settings.

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_DATA_DIR` | `.rag-data` | Persistent store |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Model identity |
| `RAG_MODEL_PATH` | empty | Local model weights, preserving model identity |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `200` / `50` | Token chunking limits |
| `RAG_HYBRID` | `true` | Combine dense and lexical retrieval |
| `RAG_RERANKER_MODEL` | empty | Optional local cross-encoder |
| `RAG_API_KEY` | empty | Bearer authentication for search and answer |
| `RAG_LLM_BASE_URL` / `RAG_LLM_MODEL` | empty | Explicit answer backend |
| `RAG_LLM_API_KEY` | `local` | Server-side answer credential |
| `RAG_LLM_MAX_TOKENS` / `RAG_LLM_TIMEOUT` | `900` / `120` | Output ceiling / timeout seconds |

Use **one writer per store**. Stop the service before ingestion, then restart readers.
Source files and manifests are written atomically; an ingestion batch is not a
transaction. An embedding failure exits with an error; rerun the same input to resume.
Do not mix embedding models or chunk settings in a store: mismatches are rejected.
Reingest into a new directory to change those settings. Default chunks reserve room
for document headers under MiniLM's 256-token sequence limit; long titles/tags or
custom models may require smaller chunks and model-specific evaluation.

The service binds to loopback by default. For public hosting, set `RAG_API_KEY` and
send `Authorization: Bearer YOUR_TOKEN`; use your existing gateway for HTTPS, access
policy, rate limiting and request budgets. Health and generated API documentation
remain public. The store has one shared corpus, so give separate tenants separate
stores/services. No CORS origins are enabled by default.

## Docker

```bash
docker build -t plugrag .
docker volume create plugrag-data
docker volume create plugrag-models
docker run --rm -v plugrag-data:/data -v plugrag-models:/models \
  plugrag ingest /examples/documents.json
docker run --rm -p 127.0.0.1:8000:8000 \
  -v plugrag-data:/data -v plugrag-models:/models plugrag
```

To import your own files, mount them read-only at `/input` and run `ingest /input`.
The image runs as a non-root user and excludes archive data, model weights and secrets
from its build context. Named volumes persist the store and downloaded models.

## Components and development

- `plugrag/core.py`: public document contract, configuration, ingestion and retrieval.
- `plugrag/generation.py`: optional compatible LLM adapter with citation safeguards.
- `plugrag/server.py`: mountable HTTP service with optional bearer authentication.
- `plugrag/cli.py`: ingestion, search, answers and server commands.
- `lens_rag/`: shared retrieval internals and the existing Lens-specific application.

The installable wheel includes both packages, so consumers need no separate engine
checkout. The distribution excludes the archive, frontend and deployment credentials.

```bash
python -m pip install -e '.[server,llm,legacy,dev]'
python -m pytest tests/
python -m build
```

The `lens_rag` package contains the shared engine and its original archive adapters.
The standalone interface is `plugrag`; this repository contains no Lens archive data
or personal deployment configuration.
