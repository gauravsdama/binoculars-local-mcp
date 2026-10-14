# Release readiness

Last verified: 2026-10-14

## Verdict

Ready for public source review as package version `v0.1.1`. The published
`v0.1.0` tag remains on its existing commit. No model weights, calibration
corpus, or upstream datasets are part of the release.

## Audience and project story

The intended user is a developer who needs an offline, machine-local
AI-text-detection signal through MCP. The upstream authors created the
Binoculars method and original implementation. Gaurav Dama added the MCP stdio
surface, enforced offline/network boundary, compact profile and calibration,
model-hash checks, macOS setup, packaging, tests, and CI. `NOTICE.md` records
that boundary and the preserved `upstream-base` commit.

## Release surface

- BSD-3-Clause text is preserved in `LICENSE.md`; `NOTICE.md` and
  `THIRD_PARTY_NOTICES.md` cover provenance, dependencies, optional models, and
  excluded datasets.
- `uv.lock` pins the Python graph. Wheel and source-archive checks require the
  implementation and legal notices, and reject models, datasets, samples,
  environment files, and macOS metadata.
- `scripts/setup_macos.sh` checks macOS, CPU, memory, disk, Python, and `uv`
  before any optional install or model fetch. Downloads require confirmation,
  use pinned model revisions and file allowlists, and verify complete manifests.
- The ignored local model pair occupies about 1.9 GB. It is not tracked. Each
  profile now pins a complete seven-file manifest for each model, rather than
  trusting only the weight file.

## Verification evidence

```text
uv lock --check
uv run ruff format --check binoculars tests scripts
uv run ruff check binoculars tests scripts
uv run pytest -q                         # 14 passed, 2 opt-in device checks skipped
BINOCULARS_RUN_DEVICE_TESTS=1 \
BINOCULARS_OBSERVER_MODEL=.models/qwen2.5-0.5b-base \
BINOCULARS_PERFORMER_MODEL=.models/qwen2.5-0.5b-instruct \
uv run pytest -q tests/test_device_regression.py
                                            # 2 passed (CPU and MPS)
uv run bandit -q -c pyproject.toml -r binoculars
uv run pip-audit                         # no known vulnerabilities; PyJWT 2.15.1, urllib3 2.8.0
uv run python scripts/release_guard.py   # passed, 30 release files
uv build
uv run python scripts/verify_distribution.py
```

The verified wheel was 20 KiB and the source archive was 24 KiB. Build outputs
were created in a temporary directory and removed after inspection.

The live MCP status check reported `ready=true`, `transport=stdio`,
`network_guard_verified=true`, and all seven manifest files verified for both
Qwen2.5-0.5B model directories under the ignored `.models/` path.

## Remaining boundaries

- The detector is an imperfect statistical signal, not proof of authorship.
- The compact calibration is a limited local benchmark and its corpus is not
  distributable from this repository.
- Model acquisition contacts Hugging Face and remains subject to the model
  publisher's terms. Those weights must not be added to a source release.
- The Python socket guard is defense in depth, not an OS sandbox. Native code,
  subprocesses, and compromised dependencies require an external sandbox.
- The lock wait is bounded, but native inference cannot be safely cancelled
  in-process after it begins. A hard execution deadline requires worker-process
  isolation.
- The `v0.1.1` tag must be created only after the committed tree passes CI.
  Moving or replacing the existing `v0.1.0` tag would rewrite published
  history and is not permitted.
- PyPI publishing is not part of this release. The repository has no existing
  PyPI workflow or documented PyPI project ownership, so trusted publishing
  should be configured only if the owner later establishes that release path.
- Eight analyzed tokens is a numerical safety floor, not a statistically
  calibrated minimum. The owner should retain it or raise it only after
  reviewing token-length coverage in the private calibration corpus.
- The tag and release are created only after a clean-clone verification and the
  committed tree pass the remote CI workflow.
