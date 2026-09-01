"""A per-user MCP fleet behind a `langgraph dev` graph factory.

Each resource is shared at a different level:

- connections: one `httpx` pool for everyone (lives in a transport clients
  borrow, since FastMCP closes the client it's handed).
- clients: one per user (credentials live on the transport, so a client speaks
  as exactly one identity).
- discovery: cached per user via `CacheConfig`; the user is folded into the
  cache key, so one user never sees another's catalog.

`make_graph` is the factory `langgraph dev` calls per run. Register it in a
`langgraph.json`:

    {"dependencies": ["."], "graphs": {"fleet": "./graph_factory.py:make_graph"}}
"""

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


class _SharedPool(httpx2.AsyncBaseTransport):
    """Lends `_POOL` out without letting a borrower close it."""

    handle_async_request = _POOL.handle_async_request

    async def aclose(self) -> None:
        """The pool outlives any one client."""


def _client_factory(**kwargs: Any) -> httpx2.AsyncClient:
    # Forward `**kwargs` so the caller's `auth`/`headers` reach the request.
    return httpx2.AsyncClient(transport=_SharedPool(), **kwargs)


_CACHE = InMemoryResponseCacheStore()  # one cache, partitioned by user


async def make_graph(runtime: ServerRuntime) -> CompiledStateGraph:
    """Build the agent for one run, over that run's user's fleet."""
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
