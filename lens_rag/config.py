"""Project-wide defaults for the Lens RAG pipeline."""

from pathlib import Path

WP_API_BASE = "https://thelensnola.org/wp-json/wp/v2"
DEFAULT_DATA_DIR = Path("data")
ARTICLES_DIRNAME = "articles"
MANIFEST_FILENAME = "manifest.json"
CHROMA_DIRNAME = "chroma_db"
COLLECTION_NAME = "lens_articles"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_RATE_LIMIT_SECONDS = 0.5
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 3

MINILM_MAX_SEQUENCE_LENGTH = 256
DEFAULT_CHUNK_SIZE = 200  # all-MiniLM-L6-v2 margin for document header
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_BATCH_SIZE = 64
