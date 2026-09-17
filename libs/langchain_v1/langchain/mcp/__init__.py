"""LangChain MCP adapters for connecting MCP servers with LangChain applications.

Interrupt-driven elicitation has its own types — the interrupt payload, the
answers a run resumes with, and the discriminator to recognize them by. Import
those from `langchain.mcp.elicitation`.

## Passing request metadata (`_meta`)

MCP tools produced by this namespace can forward a `_meta` field to the server
on every call. This is an MCP protocol-level field — the model never sees it —
useful for passing tenant identifiers, correlation IDs, or session context.

`_meta` forwarding is **opt-in**: pass an `MCPMetaConfig` to `MCPAdapter` or
`as_langchain_tool`. `None` (the default) disables the feature entirely —
nothing is read from `configurable` and no metadata is forwarded or surfaced.

`MCPMetaConfig.key` names the `configurable` key this adapter reads at runtime.
Different adapters in the same graph can use different keys to avoid collision.

### Standalone clients

For a standalone client, the value at `MCPMetaConfig.key` is forwarded directly:

```python
from langchain.mcp import MCPAdapter, MCPMetaConfig

meta_cfg = MCPMetaConfig(key="mcp_meta")
async with MCPAdapter("https://example.com/mcp", mcp_meta=meta_cfg) as adapter:
    tools = await adapter.list_tools()

await graph.ainvoke(
    input,
    config={
        "configurable": {
            "thread_id": "...",
            "mcp_meta": {
                "com.example/tenant-id": "acme",
                "com.example/correlation-id": "req-123",
            },
        }
    },
)
```

### `ClientGroup` metadata

For a `ClientGroup`, use its caller-assigned member keys. An empty dict `{}`
explicitly sends no metadata for that member. Server-reported names are never
used for routing.

!!! note "Multi-server `MCPConfig`"

    When `MCPAdapter` is given an `MCPConfig` or `dict` target, FastMCP mounts
    all servers behind a single composite router, so member keying cannot
    distinguish between them. Use a
    `ClientGroup` directly when per-server `_meta` is needed in a multi-server
    setup.

```python
meta_cfg = MCPMetaConfig(key="mcp_meta")
async with MCPAdapter(group, mcp_meta=meta_cfg) as adapter:
    tools = await adapter.list_tools()

await graph.ainvoke(
    input,
    config={
        "configurable": {
            "mcp_meta": {
                "crm-server": {
                    "com.example/tenant-id": "acme",
                    "com.example/correlation-id": "req-456",
                },
            },
        }
    },
)
```

### Response metadata

When `MCPMetaConfig.response_meta` is `True` (the default when `MCPMetaConfig`
is provided), response `_meta` from the server is surfaced in
`MCPToolArtifact._meta`. Set it to `False` to suppress it.

!!! note "Secrets in configurable"

    `config["configurable"]` is visible to every tool, subgraph, and callback
    in the run, and LangGraph may persist it via checkpointers. Prefer
    non-sensitive identifiers (tenant ID, correlation ID) in `mcp_meta`. Source
    credentials at call time from a secrets manager rather than storing them in
    `configurable`.

The metadata is forwarded as the MCP `_meta` protocol field to the server.
In LangGraph, `configurable` is re-passed on every `ainvoke` — including
resumes after an interrupt — so the values are always current.

!!! warning "This namespace is in beta"

    `langchain.mcp` is actively being worked on and its API may change. Importing
    from it raises a `LangChainBetaWarning` once per process. Silence it with
    `warnings.filterwarnings("ignore", category=LangChainBetaWarning)`, or scope
    the suppression with `langchain_core._api.suppress_langchain_beta_warning()`.
"""

import warnings

from langchain_core._api import LangChainBetaWarning

from langchain.mcp.adapter import MCPAdapter
from langchain.mcp.tools import MCPMetaConfig, MCPToolArtifact, as_langchain_tool

# Warned on import rather than through `@beta`, which annotates a function or
# class and so only fires once something is called. The status belongs to the
# whole namespace, and a caller should learn it when they reach for it —
# including when they import a submodule directly, which runs this module first.
warnings.warn(
    "`langchain.mcp` is in beta. It is actively being worked on, so the API may change.",
    LangChainBetaWarning,
    stacklevel=2,
)

__all__ = [
    "MCPAdapter",
    "MCPMetaConfig",
    "MCPToolArtifact",
    "as_langchain_tool",
]
