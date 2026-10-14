"""Local-only stdio MCP server for Binoculars."""

from __future__ import annotations

from typing import Literal

from .offline import configure_offline_environment, install_network_guard

configure_offline_environment()
install_network_guard()

from mcp.server.fastmcp import FastMCP  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from .config import RuntimeConfig  # noqa: E402
from .responses import format_analysis, format_status  # noqa: E402
from .runtime import DetectorRuntime  # noqa: E402

SERVER_INSTRUCTIONS = (
    "This is a local-only, read-only AI-text detection server. Call binoculars_status first. "
    "Analysis requires compatible observer and performer models already stored in local "
    "directories. Treat predictions as signals, never proof of authorship or misconduct. "
    "The MCP process blocks common network egress through Python socket APIs."
)

mcp = FastMCP("binoculars_mcp", instructions=SERVER_INSTRUCTIONS)


class AnalyzeTextInput(BaseModel):
    """Validated text-analysis request."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        max_length=100_000,
        description="Text to assess; 200-300 words generally produces a more useful signal.",
    )
    mode: Literal["low-fpr", "accuracy"] = Field(
        default="low-fpr",
        description="Threshold mode. low-fpr is more conservative about AI-generated labels.",
    )


_RUNTIME = DetectorRuntime(RuntimeConfig.from_environment())


@mcp.tool(
    name="binoculars_status",
    annotations={
        "title": "Check Binoculars Local Readiness",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def binoculars_status() -> dict[str, object]:
    """Check local model configuration and the scoped Python socket guard."""
    return format_status(_RUNTIME.config, _RUNTIME.profile, _RUNTIME.status())


@mcp.tool(
    name="binoculars_analyze_text",
    annotations={
        "title": "Analyze Text with Local Binoculars Models",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def binoculars_analyze_text(params: AnalyzeTextInput) -> dict[str, object]:
    """Assess text using pre-existing local models; never proof of authorship."""
    return format_analysis(_RUNTIME.analyze(params.text, params.mode), params.mode)


def main() -> None:
    """Run the local MCP server over standard input/output."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
