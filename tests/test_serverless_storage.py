from concurrent.futures import ThreadPoolExecutor
from lens_rag.serverless_storage import writable_chroma_directory


def test_local_index_is_not_copied(tmp_path, monkeypatch):
    monkeypatch.delenv("LENS_RAG_IMMUTABLE_ARCHIVE", raising=False)
    assert writable_chroma_directory(tmp_path) == tmp_path


def test_concurrent_staging_keeps_source_immutable(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "chroma.sqlite3").write_bytes(b"database")
    (source / "vectors").mkdir()
    (source / "vectors/index.bin").write_bytes(b"vectors")
    monkeypatch.setenv("LENS_RAG_IMMUTABLE_ARCHIVE", "1")
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "runtime"))
    with ThreadPoolExecutor(max_workers=4) as executor:
        targets = list(executor.map(writable_chroma_directory, [source] * 4))
    assert len(set(targets)) == 1
    target = targets[0]
    assert target != source
    assert (target / "vectors/index.bin").read_bytes() == b"vectors"
    (target / "chroma.sqlite3").write_bytes(b"runtime writes")
    assert (source / "chroma.sqlite3").read_bytes() == b"database"
    assert writable_chroma_directory(source) == target
    assert (target / "chroma.sqlite3").read_bytes() == b"runtime writes"
