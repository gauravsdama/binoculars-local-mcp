"""Complete, cache-safe verification of pinned local model snapshots."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .profiles import ModelFile


@dataclass(frozen=True)
class FileIdentity:
    path: Path
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class ModelVerification:
    verified: bool
    verified_files: int
    required_files: int
    error: str | None


def _identity(path: Path) -> FileIdentity:
    stat = path.stat()
    return FileIdentity(
        path=path.resolve(),
        device=stat.st_dev,
        inode=stat.st_ino,
        size=stat.st_size,
        modified_ns=stat.st_mtime_ns,
        changed_ns=stat.st_ctime_ns,
    )


@lru_cache(maxsize=32)
def _sha256(identity: FileIdentity) -> str:
    digest = hashlib.sha256()
    with identity.path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model_directory(
    path: Path | None, required_files: tuple[ModelFile, ...]
) -> ModelVerification:
    """Verify every required snapshot file, invalidating cache entries on file changes."""
    if path is None:
        return ModelVerification(False, 0, len(required_files), "model path is not configured")
    if not path.is_dir():
        return ModelVerification(False, 0, len(required_files), "model path is not a directory")
    verified_files = 0
    for expected in required_files:
        candidate = path / expected.name
        if not candidate.is_file():
            return ModelVerification(
                False,
                verified_files,
                len(required_files),
                f"missing required file: {expected.name}",
            )
        if _sha256(_identity(candidate)) != expected.sha256:
            return ModelVerification(
                False,
                verified_files,
                len(required_files),
                f"hash mismatch: {expected.name}",
            )
        verified_files += 1
    return ModelVerification(True, verified_files, len(required_files), None)
