# Release readiness

Last verified: 2026-09-15

## Verdict

Ready for public source review and a `v0.1.0` Python package/tag as a local
MCP/CLI derivative. No model weights, calibration corpus, or upstream datasets
are part of the release.

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
  use pinned model revisions and file allowlists, and verify weight hashes.
- The ignored local model pair occupies about 1.9 GB. It is not tracked.

## Verification evidence

```text
uv lock --check
uv run ruff format --check binoculars tests scripts
uv run ruff check binoculars tests scripts
uv run pytest -q                         # 7 passed
uv run bandit -q -c pyproject.toml -r binoculars
uv run pip-audit                         # no known vulnerabilities
uv run python scripts/release_guard.py   # passed, 24 release files
uv build
uv run python scripts/verify_distribution.py
```

The live MCP status check reported `ready=true`, `transport=stdio`,
`network_access=blocked`, `network_guard_verified=true`, and verified hashes
for both Qwen2.5-0.5B model directories under the ignored `.models/` path.

## Remaining boundaries

- The detector is an imperfect statistical signal, not proof of authorship.
- The compact calibration is a limited local benchmark and its corpus is not
  distributable from this repository.
- Model acquisition contacts Hugging Face and remains subject to the model
  publisher's terms. Those weights must not be added to a source release.
- A clean-clone CI run and the remote repository settings still need to pass
  after the owner commits these local changes.
