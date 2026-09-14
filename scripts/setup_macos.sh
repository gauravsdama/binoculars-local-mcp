#!/bin/zsh
set -euo pipefail

script_dir=${0:A:h}
project_dir=${script_dir:h}
install=false
download_models=false
assume_yes=false

usage() {
  printf '%s\n' \
    'Usage: ./scripts/setup_macos.sh [--check] [--install] [--download-models] [--yes]' \
    '' \
    '  --check            Inspect this Mac without changing it (default).' \
    '  --install          Create the locked Python environment.' \
    '  --download-models  Download and verify the pinned model pair (about 1.9 GB).' \
    '  --yes              Accept prompts; intended for an already-reviewed command.'
}

for arg in "$@"; do
  case "$arg" in
    --check) ;;
    --install) install=true ;;
    --download-models) download_models=true ;;
    --yes) assume_yes=true ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$arg" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ $(uname -s) != Darwin ]]; then
  printf '%s\n' 'This setup helper supports macOS only.' >&2
  exit 2
fi

arch=$(uname -m)
macos_version=$(sw_vers -productVersion)
memory_bytes=$(sysctl -n hw.memsize)
memory_gib=$((memory_bytes / 1024 / 1024 / 1024))
free_kib=$(df -Pk "$project_dir" | awk 'NR == 2 {print $4}')
free_gib=$((free_kib / 1024 / 1024))

printf 'macOS %s, %s, %s GiB memory, %s GiB free disk\n' \
  "$macos_version" "$arch" "$memory_gib" "$free_gib"

if [[ "$arch" == arm64 ]]; then
  printf '%s\n' 'Apple Silicon detected; Binoculars can use MPS acceleration.'
else
  printf '%s\n' 'Intel Mac detected; inference will use CPU and may be considerably slower.'
fi

if (( memory_gib < 8 )); then
  printf '%s\n' 'Warning: less than 8 GiB memory is not recommended for this model pair.' >&2
elif (( memory_gib < 16 )); then
  printf '%s\n' 'Caution: 8-15 GiB may work, but close memory-heavy applications first.'
else
  printf '%s\n' 'Memory check passed (16 GiB or more).'
fi

if (( free_gib < 6 )); then
  printf '%s\n' 'Warning: keep at least 6 GiB free for dependencies, models, and working space.' >&2
else
  printf '%s\n' 'Disk-space check passed.'
fi

python_command=''
for candidate in python3.13 python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c \
    'import sys; raise SystemExit(not ((3, 11) <= sys.version_info[:2] < (3, 14)))'; then
    python_command=$candidate
    break
  fi
done

if [[ -n "$python_command" ]]; then
  printf 'Python: %s (%s)\n' "$python_command" "$($python_command --version 2>&1)"
else
  printf '%s\n' 'Python 3.11, 3.12, or 3.13 is required.' >&2
fi

if command -v uv >/dev/null 2>&1; then
  printf 'uv: %s\n' "$(uv --version)"
else
  printf '%s\n' 'uv is not installed. Install it with: brew install uv'
fi

if ! $install && ! $download_models; then
  printf '%s\n' 'Check complete. No files or packages were changed.'
  exit 0
fi

if [[ -z "$python_command" ]] || ! command -v uv >/dev/null 2>&1; then
  printf '%s\n' 'Install the missing prerequisites above, then rerun this command.' >&2
  exit 1
fi

confirm() {
  local prompt=$1
  if $assume_yes; then
    return 0
  fi
  printf '%s [y/N] ' "$prompt"
  read -r reply
  [[ "$reply" == [Yy] || "$reply" == [Yy][Ee][Ss] ]]
}

cd "$project_dir"
if $install; then
  confirm 'Create or update .venv from the locked dependency file?' || exit 0
  uv sync --python "$python_command" --extra dev --locked
fi

if $download_models; then
  if (( free_gib < 4 )); then
    printf '%s\n' 'Model download stopped: at least 4 GiB free disk is required.' >&2
    exit 1
  fi
  confirm 'Download the pinned Qwen model pair from Hugging Face now?' || exit 0
  uv run python scripts/download_models.py --destination .models
  source "$project_dir/.models/env.zsh"
  uv run python -c 'from binoculars.mcp_server import binoculars_status; print(binoculars_status())'
fi

printf '%s\n' 'Setup complete. Model files and local paths remain ignored by Git.'
