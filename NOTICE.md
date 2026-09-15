# Notices and source provenance

This repository preserves the Git history of the upstream Binoculars project
at <https://github.com/ahans30/Binoculars>. The upstream project introduced the
Binoculars scoring method and its original Python implementation. Its BSD
3-Clause license is preserved in `LICENSE.md`, and the upstream baseline is
retained locally as the `upstream-base` branch at commit `b4a51ef`.

Gaurav Dama's derivative work begins after that baseline and includes the MCP
stdio interface, enforced offline model loading, process-level network guard,
compact-model profile and calibration, weight verification, macOS setup,
packaging, tests, CI, and security documentation.

The optional Qwen model files are not part of this repository. The setup tool
downloads pinned revisions of Qwen2.5-0.5B and Qwen2.5-0.5B-Instruct from their
publisher only after explicit confirmation. Users must review the model cards
and licenses before downloading or redistributing those files.

The local calibration corpus and upstream evaluation datasets are not
distributed here.
