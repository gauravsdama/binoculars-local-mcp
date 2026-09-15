#!/usr/bin/env python3
"""Check the tracked release surface without reading ignored model data."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
git = shutil.which("git")
if git is None:
    raise SystemExit("Release guard requires Git.")
tracked = (
    subprocess.run(  # noqa: S603 -- executable is resolved locally; arguments are static.
        [git, "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    .stdout.decode()
    .split("\0")
)
tracked = [path for path in tracked if path and (root / path).exists()]
blocked_parts = {".DS_Store", ".env", ".models", "datasets", "samples"}
violations = [path for path in tracked if blocked_parts.intersection(Path(path).parts)]
patterns = [
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
]
for relative in tracked:
    path = root / relative
    if path.is_file() and path.stat().st_size <= 1_000_000:
        data = path.read_bytes()
        if any(pattern.search(data) for pattern in patterns):
            violations.append(f"secret-shaped content: {relative}")

if violations:
    raise SystemExit("Release guard failed:\n- " + "\n- ".join(sorted(set(violations))))
print(f"Release guard passed for {len(tracked)} tracked files.")
