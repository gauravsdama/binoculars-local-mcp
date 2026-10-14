"""Opt-in golden and performance checks for the pinned CPU/MPS profile."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXPECTED_SCORES = {"cpu": 1.027764, "mps": 1.019900}
SCORE_TOLERANCE = {"cpu": 0.000005, "mps": 0.000500}


def _model_paths() -> tuple[Path, Path]:
    if os.environ.get("BINOCULARS_RUN_DEVICE_TESTS") != "1":
        pytest.skip("set BINOCULARS_RUN_DEVICE_TESTS=1 to run pinned-model device checks")
    observer = os.environ.get("BINOCULARS_OBSERVER_MODEL")
    performer = os.environ.get("BINOCULARS_PERFORMER_MODEL")
    if not observer or not performer:
        pytest.fail("device checks require both BINOCULARS_*_MODEL paths")
    return Path(observer), Path(performer)


@pytest.mark.device
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_pinned_profile_device_golden_and_performance(device: str) -> None:
    if device == "mps":
        import torch

        if not torch.backends.mps.is_available():
            pytest.skip("MPS is not available")

    observer, performer = _model_paths()
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(  # noqa: S603 -- executable is the active test interpreter.
        [
            sys.executable,
            str(repository / "scripts" / "benchmark_local.py"),
            "--device",
            device,
            "--iterations",
            "2",
            "--observer",
            str(observer),
            "--performer",
            str(performer),
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = json.loads(completed.stdout)
    low_score, high_score = result["score_range"]

    assert low_score == pytest.approx(EXPECTED_SCORES[device], abs=SCORE_TOLERANCE[device])
    assert high_score == pytest.approx(EXPECTED_SCORES[device], abs=SCORE_TOLERANCE[device])
    assert result["model_load_seconds"] < 10
    assert result["first_inference_seconds"] < 10
    assert result["repeated_inference_p95_seconds"] < 10
