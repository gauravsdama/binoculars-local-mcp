"""Measure local Binoculars model loading and repeated inference."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import time
from pathlib import Path

from binoculars import Binoculars

SAMPLE = """
The maintenance team reviewed the cooling system after several operators noticed
that the equipment room felt warmer late in the afternoon. Temperature records
showed a gradual rise rather than a sudden failure, so the technicians inspected
airflow, filters, fan speed, and the placement of nearby storage boxes. A filter
had accumulated more dust than expected, and two boxes were partly blocking a
return vent. After replacing the filter and clearing the vent, the team monitored
the room through another full workday. The temperature remained within the normal
range, and the fans no longer stayed at their highest setting. The incident did
not interrupt production, but it exposed a weakness in the inspection checklist.
The revised checklist now includes a visual check around each return vent and a
monthly comparison of temperature trends. Operators were also asked to report
small changes in noise or airflow instead of waiting for an alarm. These steps
are intentionally simple because the goal is to catch ordinary maintenance
problems before they become equipment failures. The team will review the records
again after three months to decide whether the new checks are frequent enough.
""".strip()


def peak_rss_mb() -> float:
    """Return this process's peak resident memory in MiB."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
    return value / divisor


def synchronize(device: str) -> None:
    """Wait for pending accelerator work before recording elapsed time."""
    if device == "mps":
        import torch

        torch.mps.synchronize()
    elif device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def accelerator_memory_mb(device: str) -> dict[str, float]:
    """Return allocator memory reported by the selected accelerator."""
    if device != "mps":
        return {}
    import torch

    return {
        "mps_current_allocated_mb": round(torch.mps.current_allocated_memory() / 1024**2, 1),
        "mps_driver_allocated_mb": round(torch.mps.driver_allocated_memory() / 1024**2, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--observer",
        type=Path,
        default=os.environ.get("BINOCULARS_OBSERVER_MODEL"),
        required="BINOCULARS_OBSERVER_MODEL" not in os.environ,
    )
    parser.add_argument(
        "--performer",
        type=Path,
        default=os.environ.get("BINOCULARS_PERFORMER_MODEL"),
        required="BINOCULARS_PERFORMER_MODEL" not in os.environ,
    )
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("--iterations must be at least 2")

    started = time.perf_counter()
    detector = Binoculars(args.observer, args.performer, device=args.device)
    synchronize(args.device)
    model_load_seconds = time.perf_counter() - started

    timings: list[float] = []
    scores: list[float] = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        scores.append(float(detector.compute_score(SAMPLE)))
        synchronize(args.device)
        timings.append(time.perf_counter() - started)

    ordered = sorted(timings[1:])
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    result = {
        "device": args.device,
        "model_load_seconds": round(model_load_seconds, 3),
        "first_inference_seconds": round(timings[0], 3),
        "repeated_inference_median_seconds": round(statistics.median(timings[1:]), 3),
        "repeated_inference_p95_seconds": round(ordered[p95_index], 3),
        "iterations": args.iterations,
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "sample_characters": len(SAMPLE),
        "score_range": [round(min(scores), 6), round(max(scores), 6)],
        **accelerator_memory_mb(args.device),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
