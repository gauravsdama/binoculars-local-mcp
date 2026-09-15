# Third-party software and models

The source retains and modifies the BSD-3-Clause-licensed Binoculars work named
in `NOTICE.md`. Runtime dependencies are separately licensed projects,
including MCP, NumPy, Pydantic, SentencePiece, PyTorch, and Transformers. Their
exact resolved versions are recorded in `uv.lock`; installing the project
downloads those packages from their configured package index.

Optional Qwen weights are not bundled. `scripts/download_models.py` records the
publisher repositories, exact revisions, allowed file list, and expected
SafeTensors hashes used by the compact profile. Model licenses remain the
publisher's terms and are not replaced by this repository's BSD license.
