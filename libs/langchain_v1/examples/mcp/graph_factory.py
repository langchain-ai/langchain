"""A per-user MCP fleet behind a `langgraph dev` graph factory.

The unit of sharing is different for each resource, and getting those levels
right is the whole design.

*Connections* are shared by everyone. One `httpx` pool serves every user and
every server. It lives in a transport the clients borrow rather than own,
because FastMCP closes the client it is handed at the end of each session — so
the pool must not be inside the client.

*Clients* are per user. Credentials live on the transport, so one client can
only ever speak as one identity. Client objects are cheap; the pool underneath
them is what costs, and that stays shared. Each user therefore gets their own
`ClientGroup`, built fresh for each run.

*Discovery* is cached per user. Rather than memoize the tool list in a dict of
our own, every client shares one response cache (SEP-2549) and carries a
`CacheConfig` keyed on the user. A repeat `tools/list` is served from that
cache instead of the wire, and because the key includes the user, one user
never sees another's catalog — the isolation a per-user fleet needs is a
property of the key, not code we maintain. `cache_mode="use"` is what lets
`get_tools` read the cache; a bare `ClientGroup.list_tools` would refresh it.

The cache only helps for servers that opt in (`cache_ttl`, `cache_scope`); see
`_servers.py`. A server that sends no TTL hint is never cached, so a fleet of
arbitrary third-party servers would fall back to fetching every time.

`make_graph` is the factory `langgraph dev` calls per run. It reads the
caller's identity off the injected `ServerRuntime` and mints that user's MCP
token, so the fleet a run talks to is the fleet its user is allowed to see.
Name it in a `langgraph.json`:

    {"dependencies": ["."], "graphs": {"fleet": "./examples/mcp/graph_factory.py:make_graph"}}
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx2
from _servers import token_for
from fastmcp.client import Client
from fastmcp.client.auth import BearerAuth
from fastmcp.client.group import ClientGroup
from fastmcp.client.transports import StreamableHttpTransport
from mcp.client.caching import CacheConfig, InMemoryResponseCacheStore

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
    from langgraph_sdk.runtime import ServerRuntime

SYSTEM_PROMPT = "You answer questions using the tools available to you."

SERVERS = {
    "calendar": "http://localhost:8001/mcp",
    "docs": "http://localhost:8002/mcp",
}
"""Server name to URL. A deployment reads this from its own configuration."""

_POOL = httpx2.AsyncHTTPTransport()
"""One connection pool, shared by every user and every server."""


class _SharedPool(httpx2.AsyncBaseTransport):
    """Lends `_POOL` out without letting a borrower close it.

    FastMCP wraps the client it builds in `async with`, so a client that owned
    the pool would take the pool down with its own session.
    """

    handle_async_request = _POOL.handle_async_request

    async def aclose(self) -> None:
        """Do nothing: the pool outlives any one client."""


def _client_factory(**kwargs: Any) -> httpx2.AsyncClient:
    """Build a throwaway client on the shared pool.

    Forwarding `**kwargs` rather than naming the arguments is deliberate: the
    transport passes the caller's `auth` and `headers` through here, and a
    factory that dropped them would unauthenticate every request silently.
    """
    return httpx2.AsyncClient(transport=_SharedPool(), **kwargs)


_CACHE = InMemoryResponseCacheStore()
"""One response cache, shared by every client. Entries are partitioned by user.

A deployment swaps this for a store backed by something durable and shared
across replicas, so a fleet of workers answers from one cache. `CacheConfig`
folds the user into every key, so a shared store never mixes one user's catalog
into another's.
"""


async def make_graph(runtime: ServerRuntime) -> CompiledStateGraph:
    """Build the agent for one run, over that run's user's fleet.

    The user comes off the `ServerRuntime` the server injects — the
    authenticated caller when custom auth is configured, and `anonymous`
    otherwise so the example still runs. Everything below is that user's alone:
    their token on every request, and a response cache read under their key, so
    an authorization-filtered server returns only the catalog they may see and
    a repeat discovery is served from cache rather than the wire.
    """
    user = runtime.user.identity if runtime.user is not None else "anonymous"
    auth = BearerAuth(token_for(user))
    group = ClientGroup(
        {
            name: Client(
                StreamableHttpTransport(url, auth=auth, httpx_client_factory=_client_factory),
                # Same store for everyone, but keyed on the user: `target_id`
                # is folded into the cache key, so one user's cached catalog is
                # never served to another.
                cache=CacheConfig(store=_CACHE, target_id=user, partition=user),
            )
            for name, url in SERVERS.items()
        }
    )
    # `cache_mode="use"` is the point: a bare `ClientGroup.list_tools` defaults
    # to `refresh` and would repopulate the cache instead of reading it.
    tools = await MCPAdapter(group).get_tools(cache_mode="use")
    return create_agent(
        "anthropic:claude-sonnet-5",
        tools,
        system_prompt=SYSTEM_PROMPT,
    )
