"""A per-user MCP fleet behind a `langgraph dev` graph factory."""

from __future__ import annotations

from typing import Any

import httpx2
from _servers import token_for
from fastmcp.client import Client
from fastmcp.client.auth import BearerAuth
from fastmcp.client.group import ClientGroup
from fastmcp.client.transports import StreamableHttpTransport

# Runtime imports (not TYPE_CHECKING): `langgraph dev` classifies the factory
# via `get_type_hints(make_graph)`, which must resolve every annotation.
from langgraph.graph.state import CompiledStateGraph  # noqa: TC002
from langgraph_sdk.runtime import ServerRuntime  # noqa: TC002
from mcp.client.caching import CacheConfig, InMemoryResponseCacheStore

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter

SYSTEM_PROMPT = "You answer questions using the tools available to you."
SERVERS = {
    "calendar": "http://localhost:8001/mcp",
    "docs": "http://localhost:8002/mcp",
}

_POOL = httpx2.AsyncHTTPTransport()  # one pool, shared by everyone
_CACHE = InMemoryResponseCacheStore()  # one cache, partitioned by user


class _SharedPool(httpx2.AsyncBaseTransport):
    # Lends `_POOL` out without letting a borrower close it.
    handle_async_request = _POOL.handle_async_request

    async def aclose(self) -> None: ...


def _client_factory(**kwargs: Any) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=_SharedPool(), **kwargs)


async def make_graph(runtime: ServerRuntime) -> CompiledStateGraph:
    """Build an agent over the calling user's MCP fleet (called per run)."""
    user = runtime.user.identity if runtime.user is not None else "anonymous"
    auth = BearerAuth(token_for(user))
    group = ClientGroup(
        {
            name: Client(
                StreamableHttpTransport(url, auth=auth, httpx_client_factory=_client_factory),
                cache=CacheConfig(store=_CACHE, target_id=user, partition=user),
            )
            for name, url in SERVERS.items()
        }
    )
    # `cache_mode="use"` reads the per-user cache; the group's default would refresh it.
    tools = await MCPAdapter(group).list_tools(cache_mode="use")
    return create_agent("anthropic:claude-sonnet-5", tools, system_prompt=SYSTEM_PROMPT)
