"""Local-only stdio MCP server for Binoculars."""

from __future__ import annotations

import hashlib
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Literal

from .offline import install_network_guard, network_guard_probe

install_network_guard()

from mcp.server.fastmcp import FastMCP  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from .detector import Binoculars  # noqa: E402
from .profiles import ModelProfile, get_profile  # noqa: E402

SERVER_INSTRUCTIONS = (
    "This is a local-only, read-only AI-text detection server. Call binoculars_status first. "
    "Analysis requires compatible observer and performer models already stored in local "
    "directories. Treat predictions as signals, never proof of authorship or misconduct. "
    "Network access is blocked."
)

mcp = FastMCP("binoculars_mcp", instructions=SERVER_INSTRUCTIONS)


class AnalyzeTextInput(BaseModel):
    """Validated text-analysis request."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        max_length=100_000,
        description="Text to assess; 200-300 words generally produces a more useful signal.",
    )
    mode: Literal["low-fpr", "accuracy"] = Field(
        default="low-fpr",
        description="Threshold mode. low-fpr is more conservative about AI-generated labels.",
    )


_DETECTOR: Binoculars | None = None
_DETECTOR_LOCK = threading.RLock()


def _configured_paths() -> tuple[Path | None, Path | None]:
    observer = os.environ.get("BINOCULARS_OBSERVER_MODEL")
    performer = os.environ.get("BINOCULARS_PERFORMER_MODEL")
    return (
        Path(observer).expanduser() if observer else None,
        Path(performer).expanduser() if performer else None,
    )


def _configured_profile() -> ModelProfile:
    return get_profile(os.environ.get("BINOCULARS_PROFILE", "qwen2.5-0.5b"))


def _model_directory_ready(path: Path | None) -> bool:
    if path is None or not path.is_dir() or not (path / "config.json").is_file():
        return False
    has_tokenizer = any(
        (path / name).is_file()
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json")
    )
    has_weights = any(path.glob("*.safetensors"))
    return has_tokenizer and has_weights


@lru_cache(maxsize=4)
def _weight_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as weights:
        for block in iter(lambda: weights.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _weight_hash_matches(path: Path | None, expected: str) -> bool:
    if path is None or not _model_directory_ready(path):
        return False
    weights = path / "model.safetensors"
    return weights.is_file() and _weight_sha256(weights) == expected


def _get_detector() -> Binoculars:
    global _DETECTOR
    if _DETECTOR is not None:
        return _DETECTOR
    with _DETECTOR_LOCK:
        if _DETECTOR is None:
            observer, performer = _configured_paths()
            if observer is None or performer is None:
                raise RuntimeError(
                    "Local models are not configured. Set BINOCULARS_OBSERVER_MODEL and "
                    "BINOCULARS_PERFORMER_MODEL to compatible local model directories."
                )
            profile = _configured_profile()
            if not _weight_hash_matches(observer, profile.observer_sha256):
                raise RuntimeError("Observer model weights do not match the configured profile")
            if not _weight_hash_matches(performer, profile.performer_sha256):
                raise RuntimeError("Performer model weights do not match the configured profile")
            _DETECTOR = Binoculars(
                observer,
                performer,
                accuracy_threshold=profile.accuracy_threshold,
                low_fpr_threshold=profile.low_fpr_threshold,
                max_token_observed=profile.max_tokens,
            )
    return _DETECTOR


@mcp.tool(
    name="binoculars_status",
    annotations={
        "title": "Check Binoculars Local Readiness",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def binoculars_status() -> dict[str, object]:
    """Check local model configuration and confirm the network guard is active."""
    observer, performer = _configured_paths()
    observer_ready = _model_directory_ready(observer)
    performer_ready = _model_directory_ready(performer)
    profile = _configured_profile()
    observer_hash_verified = _weight_hash_matches(observer, profile.observer_sha256)
    performer_hash_verified = _weight_hash_matches(performer, profile.performer_sha256)
    ready = (
        observer_ready and performer_ready and observer_hash_verified and performer_hash_verified
    )
    return {
        "ready": ready,
        "transport": "stdio",
        "network_access": "blocked",
        "network_guard_verified": network_guard_probe(),
        "observer_model": str(observer) if observer else None,
        "observer_model_ready": observer_ready,
        "observer_hash_verified": observer_hash_verified,
        "performer_model": str(performer) if performer else None,
        "performer_model_ready": performer_ready,
        "performer_hash_verified": performer_hash_verified,
        "profile": profile.name,
        "max_tokens": profile.max_tokens,
        "calibration": profile.calibration_summary,
        "guidance": (
            None if ready else "Configure the exact pinned local model files for this profile."
        ),
    }


@mcp.tool(
    name="binoculars_analyze_text",
    annotations={
        "title": "Analyze Text with Local Binoculars Models",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def binoculars_analyze_text(params: AnalyzeTextInput) -> dict[str, object]:
    """Assess text using pre-existing local models; never proof of authorship."""
    with _DETECTOR_LOCK:
        detector = _get_detector()
        detector.change_mode(params.mode)
        score = float(detector.compute_score(params.text))
        threshold = detector.threshold
    label = "Most likely AI-generated" if score < threshold else "Most likely human-generated"
    return {
        "score": score,
        "threshold": threshold,
        "mode": params.mode,
        "label": label,
        "characters_analyzed": len(params.text),
        "profile": _configured_profile().name,
        "calibration": _configured_profile().calibration_summary,
        "caution": "This detector is imperfect; do not use the result as proof of authorship.",
    }


def main() -> None:
    """Run the local MCP server over standard input/output."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
