# Binoculars Local MCP

This repository is a local, security-focused derivative of [Binoculars](https://github.com/ahans30/Binoculars), the zero-shot AI-text detection method introduced in [*Spotting LLMs With Binoculars*](https://arxiv.org/abs/2401.12070). The upstream authors created the scoring method and original implementation. This project adds a read-only MCP interface, offline model loading, process-level network blocking, a compact local model profile, and profile-specific calibration.

The detector returns a statistical signal, not proof that a person or model wrote a passage. It should not be the sole basis for grading, discipline, employment decisions, accusations, or other consequential action.

## Local security boundary

- MCP transport is `stdio`; the server does not listen on a port.
- Before MCP, model, or HTTP-capable dependencies import, the MCP process forces Hugging Face and Transformers offline modes and installs a guard for common Python TCP/UDP socket egress. Local Unix-domain sockets remain available to libraries.
- Model arguments must resolve to existing local directories; remote model IDs are rejected.
- Loads require `local_files_only=True`, `trust_remote_code=False`, and SafeTensors weights.
- The compact profile verifies SHA-256 digests for every required config, tokenizer, vocabulary, merge, generation-config, and SafeTensors file before reporting readiness or loading the detector. Cached digests are invalidated when file identity or metadata changes.
- Both tools are read-only and declare `openWorldHint=false`.
- Input is limited to 100,000 characters, and the compact profile analyzes at most 256 tokens.

Installing Python packages may contact a package index. Model acquisition is a separate explicit step. Once configured, model inference uses only verified local files. The in-process guard covers common Python socket APIs; it is defense in depth, not an operating-system sandbox or a guarantee against native code, subprocesses, or a compromised dependency. Use an OS-level network sandbox when that stronger boundary is required.

## Install

Python 3.11 through 3.13 is supported. This checkout uses Python 3.12 with `uv`:

On a Mac, begin with the optional setup helper. Its default mode only reports macOS,
processor, memory, disk, Python, and `uv` readiness:

```bash
./scripts/setup_macos.sh
```

Dependency installation and model acquisition remain separate choices:

```bash
./scripts/setup_macos.sh --install
./scripts/setup_macos.sh --download-models
source .models/env.zsh
```

The second command explains the download before it starts, fetches the two pinned model
revisions, verifies every file in each pinned model manifest, and writes an ignored local
environment file. Use `--yes` only after reviewing the requested operation.

For manual setup:

```bash
uv sync --python python3.12 --extra dev --locked
```

Model weights stay outside Git under `.models/`. Point the server at compatible local directories:

```bash
export BINOCULARS_PROFILE=qwen2.5-0.5b
export BINOCULARS_OBSERVER_MODEL=/path/to/qwen2.5-0.5b-base
export BINOCULARS_PERFORMER_MODEL=/path/to/qwen2.5-0.5b-instruct
export BINOCULARS_DEVICE=cpu  # or mps, cuda, cuda:N, or auto
```

Without both model directories, the MCP server can still start, but `binoculars_status` reports that analysis is not ready.
Configuration is captured once when the MCP process starts; restart it after changing these environment variables.

## Compact model profile

The `qwen2.5-0.5b` profile pairs Qwen2.5-0.5B with Qwen2.5-0.5B-Instruct. Their SafeTensors weights use about 1.9 GB together, substantially less than the two 7B models in the original configuration. Repository revisions, file hashes, thresholds, token limit, and the calibration summary are pinned in `binoculars/profiles.py`.

This compact profile was calibrated on 360 local texts and evaluated on a separate 240-text holdout derived from CC-News, CNN, and PubMed human/Falcon samples. The calibration corpus is not distributed with this repository. On that holdout, accuracy mode reached 88.3%. Conservative mode measured a 2.5% human false-positive rate and a 60.0% machine true-positive rate. This is a limited local benchmark, not a general performance guarantee and not the original paper's evaluation.

## MCP tools

### `binoculars_status`

Reports model readiness, selected profile, weight-hash checks, stdio transport, the active network-guard probe, and calibration context. Call it before analysis.

### `binoculars_analyze_text`

Accepts `text` and an optional threshold mode:

- `low-fpr` is the default and is more conservative about AI-generated labels.
- `accuracy` provides the profile's broader screening threshold.

The result includes the score, threshold, mode, label, model profile, device, character count, original tokenizer count, analyzed tokenizer count, and whether truncation occurred. Lower scores indicate a stronger AI-generated signal under the configured threshold. Inputs with fewer than eight analyzed tokens are rejected instead of labeled, as are non-finite numerical results. Samples of roughly 200-300 words are generally more useful than very short passages.

## Run and verify

Start the stdio server:

```bash
.venv/bin/binoculars-mcp
```

Available project checks are:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check binoculars tests
.venv/bin/bandit -q -c pyproject.toml -r binoculars
.venv/bin/pip-audit
```

Requests wait at most 120 seconds for another analysis to release the detector lock. Set `BINOCULARS_LOCK_TIMEOUT_SECONDS` between `0.1` and `3600` to change that queueing bound. Once native model inference starts, MCP cancellation cannot safely interrupt it in-process; process isolation is required for a hard execution deadline.

The default test suite uses tiny local models and runs without downloads. With the pinned models already present, run the CPU/MPS golden and performance checks explicitly:

```bash
BINOCULARS_RUN_DEVICE_TESTS=1 \
BINOCULARS_OBSERVER_MODEL=/path/to/qwen2.5-0.5b-base \
BINOCULARS_PERFORMER_MODEL=/path/to/qwen2.5-0.5b-instruct \
uv run pytest -q tests/test_device_regression.py
```

Inspect a Codex registration with:

```bash
codex mcp get binoculars-local
codex mcp list
```

Inference for a local stdio server uses the machine's MPS or CPU resources. It does not use hosted Codex inference compute.

## Local performance

Measurements below were taken on a 24 GB M4 Pro MacBook Pro with macOS 26.6.2. Each run loaded both pinned Qwen2.5-0.5B models and analyzed the same 1,196-character passage six times.

| Device | Model load | First call | Repeated median | Repeated p95 | Memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| MPS | 1.54 s | 0.31 s | 0.159 s | 0.159 s | 538 MiB peak RSS; 3,123 MiB MPS driver |
| CPU | 1.73 s | 1.25 s | 0.346 s | 0.425 s | 5,234 MiB peak RSS |

The first uncached MPS process took 2.85 seconds to load the models and 2.21 seconds for its first call. Later runs benefited from operating-system file caches. Reproduce the measurement with:

```bash
uv run python scripts/benchmark_local.py --device mps --iterations 6
uv run python scripts/benchmark_local.py --device cpu --iterations 6
```

## Limitations

Binoculars signals are weaker on short, non-English, heavily quoted, or memorized text. Results also depend on the selected observer/performer models and calibration data. Report the score, threshold, mode, and profile together; do not translate one label into certainty about authorship.

## Attribution

The Binoculars score and original implementation are by Abhimanyu Hans, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi, Aniruddha Saha, Micah Goldblum, Jonas Geiping, and Tom Goldstein. See the [paper](https://arxiv.org/abs/2401.12070), [upstream repository](https://github.com/ahans30/Binoculars), and preserved [license](LICENSE.md).

The local MCP runtime, offline and network guards, compact profile configuration, weight verification, security tests, and local calibration are Gaurav Dama's derivative work. The underlying detection method remains the upstream authors' work.
