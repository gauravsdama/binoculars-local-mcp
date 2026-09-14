"""Calibrated local model profiles for Binoculars."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    """Thresholds and runtime limits tied to one compatible model pair."""

    name: str
    observer_repository: str
    observer_revision: str
    observer_sha256: str
    performer_repository: str
    performer_revision: str
    performer_sha256: str
    accuracy_threshold: float
    low_fpr_threshold: float
    max_tokens: int
    calibration_summary: str


QWEN_05B = ModelProfile(
    name="qwen2.5-0.5b",
    observer_repository="Qwen/Qwen2.5-0.5B",
    observer_revision="060db6499f32faf8b98477b0a26969ef7d8b9987",
    observer_sha256="88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
    performer_repository="Qwen/Qwen2.5-0.5B-Instruct",
    performer_revision="7ae557604adf67be50417f59c2c2f167def9a775",
    performer_sha256="fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    accuracy_threshold=0.9477803409099579,
    low_fpr_threshold=0.905291736125946,
    max_tokens=256,
    calibration_summary=(
        "Calibrated on 360 local texts and evaluated on a disjoint 240-text holdout "
        "derived from CC-News, CNN, and PubMed human/Falcon samples. The calibration "
        "corpus is not distributed. Holdout accuracy was 88.3%; conservative-mode "
        "human FPR was 2.5% and machine TPR was 60.0%."
    ),
)

PROFILES = {QWEN_05B.name: QWEN_05B}


def get_profile(name: str) -> ModelProfile:
    """Return a known profile or fail closed rather than using mismatched cutoffs."""
    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown BINOCULARS_PROFILE {name!r}; available: {choices}") from exc
