"""Argument-header mirroring across discovery and invocation HTTP connections."""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from fastmcp import Client, FastMCP
from fastmcp.client.group import ClientGroup
from fastmcp.utilities.tests import run_server_in_process
from pydantic import Field

from langchain.mcp import MCPAdapter


def _run_header_server(host: str, port: int) -> None:
    server: FastMCP[None] = FastMCP("header-server", cache_ttl=60)

    @server.tool
    def read_repository(
        owner: Annotated[str, Field(json_schema_extra={"x-mcp-header": "owner"})],
        repo: Annotated[str, Field(json_schema_extra={"x-mcp-header": "repo"})],
    ) -> str:
        """Return the repository after the transport validates its argument headers."""
        return f"{owner}/{repo}"

    server.run(transport="http", host=host, port=port, json_response=True, show_banner=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("interrupts", [False, True])
async def test_detached_tool_preserves_param_headers(*, grouped: bool, interrupts: bool) -> None:
    with run_server_in_process(_run_header_server) as url:

        async def own_handler(*_: Any) -> Any:
            return None

        client: Client[Any] = Client(
            f"{url}/mcp", elicitation_handler=None if interrupts else own_handler, cache=True
        )
        target = ClientGroup({"github": client}) if grouped else client
        adapter = MCPAdapter(target)
        [tool] = await adapter.list_tools(cache_mode="use")
        arguments = {"owner": "example", "repo": "repository"}

        for _ in range(2):
            [block] = await tool.ainvoke(arguments)
            assert block["text"] == "example/repository"
