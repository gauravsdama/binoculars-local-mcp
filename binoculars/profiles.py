"""Calibrated local model profiles for Binoculars."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelFile:
    """One required file in a pinned model snapshot."""

    name: str
    sha256: str


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
    observer_files: tuple[ModelFile, ...]
    performer_files: tuple[ModelFile, ...]
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
    observer_files=(
        ModelFile(
            "config.json", "479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b"
        ),
        ModelFile(
            "generation_config.json",
            "8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8",
        ),
        ModelFile("merges.txt", "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
        ModelFile(
            "model.safetensors",
            "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
        ),
        ModelFile(
            "tokenizer.json",
            "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        ),
        ModelFile(
            "tokenizer_config.json",
            "c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9",
        ),
        ModelFile("vocab.json", "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    ),
    performer_files=(
        ModelFile(
            "config.json", "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45"
        ),
        ModelFile(
            "generation_config.json",
            "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
        ),
        ModelFile("merges.txt", "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
        ModelFile(
            "model.safetensors",
            "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
        ),
        ModelFile(
            "tokenizer.json",
            "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        ),
        ModelFile(
            "tokenizer_config.json",
            "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
        ),
        ModelFile("vocab.json", "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    ),
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
