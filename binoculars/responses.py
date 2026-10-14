"""Pure MCP response formatting."""

from __future__ import annotations

from .config import RuntimeConfig
from .offline import network_guard_probe
from .profiles import ModelProfile
from .runtime import RuntimeAnalysis, RuntimeStatus


def format_status(
    config: RuntimeConfig, profile: ModelProfile, status: RuntimeStatus
) -> dict[str, object]:
    return {
        "ready": status.ready,
        "transport": "stdio",
        "network_access": "blocked",
        "network_guard_scope": "common Python socket egress in the MCP process",
        "network_guard_verified": network_guard_probe(),
        "observer_model": str(config.observer_path) if config.observer_path else None,
        "observer_model_ready": status.observer.verified,
        "observer_hash_verified": status.observer.verified,
        "observer_manifest_verified": status.observer.verified,
        "observer_files_verified": status.observer.verified_files,
        "observer_verification_error": status.observer.error,
        "performer_model": str(config.performer_path) if config.performer_path else None,
        "performer_model_ready": status.performer.verified,
        "performer_hash_verified": status.performer.verified,
        "performer_manifest_verified": status.performer.verified,
        "performer_files_verified": status.performer.verified_files,
        "performer_verification_error": status.performer.error,
        "profile": profile.name,
        "max_tokens": profile.max_tokens,
        "device_requested": config.device,
        "device_active": status.active_device,
        "detector_loaded": status.detector_loaded,
        "lock_timeout_seconds": config.lock_timeout_seconds,
        "calibration": profile.calibration_summary,
        "guidance": None if status.ready else "Configure the exact pinned local model snapshot.",
    }


def format_analysis(result: RuntimeAnalysis, mode: str) -> dict[str, object]:
    analysis = result.analysis
    label = (
        "Most likely AI-generated"
        if analysis.score < result.threshold
        else "Most likely human-generated"
    )
    return {
        "score": analysis.score,
        "threshold": result.threshold,
        "mode": mode,
        "label": label,
        "input_tokens": analysis.input_tokens,
        "tokens_analyzed": analysis.tokens_analyzed,
        "truncated": analysis.truncated,
        "characters_analyzed": result.characters_analyzed,
        "device": result.device,
        "profile": result.profile.name,
        "calibration": result.profile.calibration_summary,
        "caution": "This detector is imperfect; do not use the result as proof of authorship.",
    }
