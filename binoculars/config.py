"""Immutable configuration for one MCP server process."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    """Environment-derived settings captured once by the detector runtime."""

    profile_name: str
    observer_path: Path | None
    performer_path: Path | None
    device: str
    lock_timeout_seconds: float

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> RuntimeConfig:
        values = os.environ if environment is None else environment
        timeout_text = values.get("BINOCULARS_LOCK_TIMEOUT_SECONDS", "120")
        try:
            lock_timeout_seconds = float(timeout_text)
        except ValueError as exc:
            raise ValueError("BINOCULARS_LOCK_TIMEOUT_SECONDS must be a number") from exc
        if not 0.1 <= lock_timeout_seconds <= 3600:
            raise ValueError("BINOCULARS_LOCK_TIMEOUT_SECONDS must be between 0.1 and 3600")

        def optional_path(name: str) -> Path | None:
            value = values.get(name)
            return Path(value).expanduser() if value else None

        device = values.get("BINOCULARS_DEVICE", "auto")
        valid_device = device in {"auto", "cpu", "mps", "cuda"} or (
            device.startswith("cuda:") and device.removeprefix("cuda:").isdigit()
        )
        if not valid_device:
            raise ValueError(
                "BINOCULARS_DEVICE must be 'auto', 'cpu', 'mps', 'cuda', or 'cuda:<index>'"
            )

        return cls(
            profile_name=values.get("BINOCULARS_PROFILE", "qwen2.5-0.5b"),
            observer_path=optional_path("BINOCULARS_OBSERVER_MODEL"),
            performer_path=optional_path("BINOCULARS_PERFORMER_MODEL"),
            device=device,
            lock_timeout_seconds=lock_timeout_seconds,
        )
