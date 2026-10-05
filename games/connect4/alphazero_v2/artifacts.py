"""Atomic, no-overwrite artifact publication: temporary file -> validation -> link."""
import hashlib
import os
from pathlib import Path
import tempfile

import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(path, payload, validate):
    """Save to a sibling temporary file, validate by reloading, then publish.

    ``validate(temp_path)`` must raise on any defect. Publication uses a hard
    link, which atomically refuses to replace an existing artifact. Returns the
    published file's SHA-256. Nothing is published if any step fails.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        with torch.random.fork_rng(devices=[]):
            validate(temporary)
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)
    return file_sha256(path)
