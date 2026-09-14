#!/usr/bin/env python3
"""Download and verify the pinned local Binoculars model pair."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download

from binoculars.profiles import QWEN_05B

MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(repository: str, revision: str, expected: str, destination: Path) -> None:
    weights = destination / "model.safetensors"
    if weights.is_file() and sha256(weights) == expected:
        print(f"Verified existing model: {destination}")
        return

    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(
            f"Refusing to replace an incomplete or mismatched directory: {destination}\n"
            "Move it aside and run this command again."
        )

    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repository}@{revision} to {destination}")
    snapshot_download(
        repo_id=repository,
        revision=revision,
        local_dir=destination,
        allow_patterns=MODEL_FILES,
    )
    if not weights.is_file() or sha256(weights) != expected:
        raise SystemExit(f"Weight verification failed for {destination}")
    print(f"Verified model weights: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path(".models"))
    args = parser.parse_args()
    root = args.destination.expanduser().resolve()
    fetch(
        QWEN_05B.observer_repository,
        QWEN_05B.observer_revision,
        QWEN_05B.observer_sha256,
        root / "qwen2.5-0.5b-base",
    )
    fetch(
        QWEN_05B.performer_repository,
        QWEN_05B.performer_revision,
        QWEN_05B.performer_sha256,
        root / "qwen2.5-0.5b-instruct",
    )

    env_file = root / "env.zsh"
    env_file.write_text(
        "\n".join(
            [
                "export BINOCULARS_PROFILE=qwen2.5-0.5b",
                f"export BINOCULARS_OBSERVER_MODEL={root / 'qwen2.5-0.5b-base'}",
                f"export BINOCULARS_PERFORMER_MODEL={root / 'qwen2.5-0.5b-instruct'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    env_file.chmod(0o600)
    print(f"Local configuration: source {env_file}")


if __name__ == "__main__":
    main()
