#!/usr/bin/env python3
"""Verify that built archives contain source and exclude local/private data."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

dist = Path(__file__).resolve().parents[1] / "dist"
archives = sorted(dist.glob("*"))
if not archives:
    raise SystemExit("No distribution archives found; run `uv build` first.")

blocked_parts = {".models", "datasets", "samples", ".env", ".DS_Store"}

for archive in archives:
    if archive.suffix == ".whl":
        with zipfile.ZipFile(archive) as package:
            names = package.namelist()
    elif archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as package:
            names = package.getnames()
    else:
        continue
    required_suffixes = {"binoculars/mcp_server.py", "LICENSE.md", "NOTICE.md"}
    if archive.name.endswith(".tar.gz"):
        required_suffixes.add("README.md")
    normalized = {name.split("/", 1)[-1] for name in names}
    missing = {
        suffix for suffix in required_suffixes if not any(name.endswith(suffix) for name in names)
    }
    blocked = [name for name in names if blocked_parts.intersection(Path(name).parts)]
    if missing or blocked:
        raise SystemExit(f"{archive.name}: missing={sorted(missing)} blocked={blocked}")
    print(f"Verified {archive.name}: {len(normalized)} members")
