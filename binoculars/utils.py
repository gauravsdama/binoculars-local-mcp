"""Validation helpers for local model assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def require_local_model_path(value: str | Path, *, label: str) -> Path:
    """Resolve and validate a local model directory without accepting model IDs."""
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"{label} must be a local model directory: {path}")
    return path


def assert_tokenizer_consistency(tokenizer_1: Any, tokenizer_2: Any) -> None:
    """Require the observer and performer to use the same token vocabulary."""
    if tokenizer_1.get_vocab() != tokenizer_2.get_vocab():
        raise ValueError("Observer and performer tokenizers are not identical")
