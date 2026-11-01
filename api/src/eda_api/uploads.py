"""Upload handling with Privacy-by-Design: the raw file never outlives its run.

The upload is streamed to a private file (so a 50 MB file isn't held twice in
memory and the size limit is enforced while reading, not after). It waits there
until a worker picks up the job; the worker deletes it as soon as the data is in
memory, and again in `finally`. Rejected uploads are deleted immediately.
Leftovers from a hard crash (power loss, kill -9) are purged at startup.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import BinaryIO

import pandas as pd

CHUNK = 1024 * 1024


class UploadRejected(ValueError):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


def save_upload(source: BinaryIO, directory: Path, max_bytes: int, name: str | None = None) -> tuple[Path, int]:
    """Stream `source` into a private file (size limit enforced while reading). Returns (path, size).

    On any rejection the partial file is deleted before the error propagates. On success the
    caller owns the file and must delete it (the worker does, in `finally`)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name or uuid.uuid4().hex}.upload"
    size = 0
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)   # readable by this user only
        with os.fdopen(fd, "wb") as out:
            while chunk := source.read(CHUNK):
                size += len(chunk)
                if size > max_bytes:
                    raise UploadRejected(f"file is larger than the {max_bytes // (1024 * 1024)} MB limit", 413)
                out.write(chunk)
        if size == 0:
            raise UploadRejected("file is empty")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path, size


def read_csv(path: Path, max_rows: int, max_columns: int) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, nrows=max_rows + 1, low_memory=False, encoding_errors="replace")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        raise UploadRejected(f"could not read the file as CSV ({type(exc).__name__})") from None
    if df.empty or df.shape[1] < 2:
        raise UploadRejected("the CSV needs at least 2 columns and 1 row")
    if len(df) > max_rows:
        raise UploadRejected(f"the CSV has more than {max_rows:,} rows")
    if df.shape[1] > max_columns:
        raise UploadRejected(f"the CSV has {df.shape[1]} columns; the limit is {max_columns}")
    return df


def purge_stale_uploads(directory: Path, older_than_minutes: int) -> int:
    if not directory.exists():
        return 0
    cutoff = time.time() - older_than_minutes * 60
    removed = 0
    for path in directory.glob("*.upload"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except FileNotFoundError:
            pass
    return removed
