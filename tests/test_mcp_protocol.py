from __future__ import annotations

from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_stdio_server_lists_and_calls_tools() -> None:
    repository = Path(__file__).resolve().parents[1]
    parameters = StdioServerParameters(
        command=str(repository / ".venv" / "bin" / "python"),
        args=["-m", "binoculars.mcp_server"],
        cwd=repository,
    )

    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            assert names == {"binoculars_status", "binoculars_analyze_text"}

            result = await session.call_tool("binoculars_status")
            assert result.isError is False
            assert result.structuredContent is not None
            assert result.structuredContent["network_guard_verified"] is True
            assert result.structuredContent["transport"] == "stdio"
            assert result.structuredContent["profile"] == "qwen2.5-0.5b"

            unavailable = await session.call_tool(
                "binoculars_analyze_text",
                {"params": {"text": "local test", "mode": "low-fpr"}},
            )
            assert unavailable.isError is True
            assert "Local models are not configured" in unavailable.content[0].text
