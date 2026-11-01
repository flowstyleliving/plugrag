"""Stage an immutable Chroma snapshot in writable serverless storage."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path


def writable_chroma_directory(source: Path) -> Path:
    """Copy once per snapshot, with a process-safe lock and atomic publication."""

    if os.getenv("LENS_RAG_IMMUTABLE_ARCHIVE") != "1":
        return source

    import fcntl

    source = source.resolve()
    stat = (source / "chroma.sqlite3").stat()
    identity = f"{source}:{stat.st_size}:{stat.st_mtime_ns}"
    revision = hashlib.sha256(identity.encode()).hexdigest()[:20]
    root = Path(tempfile.gettempdir()) / "lens-rag"
    root.mkdir(parents=True, exist_ok=True)
    target = root / revision
    with (root / f"{revision}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.is_dir():
            return target
        staging = root / f"{revision}.staging"
        shutil.rmtree(staging, ignore_errors=True)
        try:
            shutil.copytree(source, staging)
            os.replace(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return target
