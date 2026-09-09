---
type: "Reference"
title: "Bearer token"
openwiki_generated: true
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:18:34.589Z
sources:
  - id: openwiki-source-6d1e3478d5b63988ee177552
    resource: repo://libs/langchain_v1/examples/mcp/auth_bearer.py
  - id: openwiki-source-4bad19dc422af3ebb00e7f2f
    resource: repo://libs/langchain_v1/examples/mcp/auth_oauth.py
  - id: openwiki-source-46fd56b09fa62a41e3c41f08
    resource: repo://libs/langchain_v1/examples/mcp/destructive_interrupt.py
  - id: openwiki-source-8b95e4b88972026f6c7678a3
    resource: repo://libs/langchain_v1/examples/mcp/graph_factory.py
  - id: openwiki-source-cfb2965ed32b54e99ffb6328
    resource: repo://libs/langchain_v1/examples/mcp/multi_server.py
  - id: openwiki-source-71f7ffcbb69cda81c2e3f940
    resource: repo://libs/langchain_v1/examples/mcp/protocol_eras.py
  - id: openwiki-source-37b31519003157eb1b1bdaef
    resource: repo://libs/langchain_v1/examples/mcp/README.md
  - id: openwiki-source-6df781509d081d60a037331b
    resource: repo://libs/langchain_v1/examples/mcp/tool_errors.py
  - id: openwiki-source-caa1f747bb1ba9b6514eeaac
    resource: repo://libs/langchain_v1/examples/mcp/transports.py
  - id: openwiki-source-0a3228970b0eadc4bcadbb5d
    resource: repo://libs/langchain_v1/langchain/mcp/adapter.py
  - id: openwiki-source-b4c5eca79ce58abf486c2776
    resource: repo://libs/langchain_v1/langchain/mcp/elicitation.py
  - id: openwiki-source-4715c337e9b93b9d00846133
    resource: repo://libs/langchain_v1/langchain/mcp/tools.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:18:34.589Z" }
---


## Overview

The Model Context Protocol (MCP) is a standard for LLM applications to discover and call tools from external servers. `langchain.mcp` provides the `MCPAdapter` class, which discovers MCP tools and converts them to LangChain `BaseTool` objects suitable for use with `create_agent`. The adapter handles protocol negotiation via FastMCP, manages multiple transports (stdio, HTTP, in-memory), supports mid-call user input via LangGraph interrupts, and surfaces tool errors to the model for recovery and retry.

## MCP Concept

MCP is a request–response protocol where:

- **Clients** (like LangChain agents) discover tools available on a server and invoke them with arguments
- **Servers** expose tools, describe their schemas, and handle calls
- **Tools** are named, documented functions with typed arguments; a server may expose many tools
- **Content** returned by a tool (text, images, files, structured data) is represented as content blocks

MCP evolved between protocol eras: the 2025-11-25 era negotiates capability via an `initialize` handshake; the 2026-07-28 era uses a `server/discover` handshake. Both eras can coexist in an agent when separate adapters are used per server.

## Architecture: MCPAdapter and FastMCP

`MCPAdapter` is the user-facing entry point. It wraps one or more FastMCP clients and exposes their tools as LangChain tools:

```python
async with MCPAdapter(target) as adapter:
    tools = await adapter.list_tools()
    agent = create_agent("anthropic:claude-sonnet-5", tools)
```

**Target types** (inferred by FastMCP):

- `str` (http/https URL only) — reached over streamable HTTP
- `Path` — launched as a subprocess over stdio, e.g. `Path("server.py")`
- `FastMCP` — in-process server with no network or subprocess
- `Client` or `ClientGroup` — pre-built FastMCP client(s)
- `MCPConfig` (dict) — multiple servers, each with independent transport and auth
- `ClientTransport` — explicit transport (HTTP, stdio, or custom)

String targets must be http(s) URLs to prevent silent local execution of existing `.py` or `.js` files. Local servers are accessed through `Path`, a transport, or `MCPConfig`.

## Tool Discovery and Conversion

`adapter.list_tools()` calls `fastmcp.Client.list_tools()` to fetch remote tools, then converts each via `as_langchain_tool()`:

- **Discovery** uses the client's response cache (configurable via `cache_mode`)
- **Conversion** reads the MCP tool's schema (from `tool.input_schema`) and creates a `StructuredTool` with:
  - `name`, `description`, and typed `args_schema` from the MCP definition
  - `coroutine` that calls the tool asynchronously
  - `response_format="content_and_artifact"` to return both model-visible content blocks and structured data
  - `metadata["mcp"]` carrying tool annotations, server identity, and destructive hints
  - `handle_tool_error` handler to surface MCP-reported errors (not transport failures) to the model

For multi-server setups (via `ClientGroup` or `MCPConfig`), tool names are prefixed per server (e.g., `weather_forecast`, `calc_add`) to disambiguate tools with the same name on different servers. The adapter's internal routing ensures each call reaches the correct server.

## Tool Invocation and Result Conversion

When a tool is called:

1. **Elicitation detection**: The adapter checks whether the underlying client is armed to drive LangGraph interrupts (see [Elicitation](#elicitation-input-mid-call) below)
2. **Tool call**: If interrupts are enabled, calls `_call_tool_with_interrupts()` to answer requests via `interrupt()`; otherwise calls `fastmcp.Client.call_tool()` directly
3. **Result conversion**: Converts MCP content blocks (text, images, files, resources) to LangChain content blocks
4. **Error handling**: If the server reports `isError=True`, raises `_MCPToolExecutionError` (a `ToolException`), which becomes a `ToolMessage` with `status="error"` so the model can see and retry
5. **Artifacts**: Extracts `structured_content` (JSON, tables, etc.) into a separate artifact field so the model receives both rendered content and structured data

Content blocks support:
- **Text** — plain string
- **Image** — base64 or URL-referenced
- **File** — base64 or URL-referenced  
- **Resource** — embedded binary or text, or URL link

Audio is not yet supported.

## Elicitation: Input Mid-Call

Some MCP tools cannot complete without asking the user a question mid-call. Instead of hanging or erroring, the server returns an `InputRequiredResult` describing what it needs. The adapter converts this to a LangGraph `interrupt()`, so a human can answer and the run resumes seamlessly.

**How it works:**

1. **Arming**: On construction, `MCPAdapter` calls `_arm_for_interrupts()` on each underlying client, setting an elicitation callback. This advertises the `elicitation` capability to the server. A pre-built client that already has its own handler is cloned first, so the caller's object is never mutated.

2. **Interrupt loop**: When `as_langchain_tool()` calls the tool, if the client was armed, it calls `_call_tool_with_interrupts()` instead of plain `call_tool()`. This loop:
   - Issues the tool call with `allow_input_required=True`
   - If the result is `InputRequiredResult`, extracts elicitation requests (form or URL)
   - Raises `interrupt()` with the request, pausing the run
   - On resume, receives answers keyed by request ID, builds response payloads, and retries the call
   - Repeats until the tool returns a terminal result

3. **Protocol era compatibility**: The interrupt loop only runs on modern servers (2026-07-28 and later) that return `InputRequiredResult`. Legacy servers never trigger it, so they work unchanged. A client armored with a pre-built handler uses that handler instead.

**Request types:**

- **Form** — server asks for structured data (JSON matching a schema); human must provide it
- **URL** — server asks the human to visit a URL (e.g., for approval or authentication); no data needed
- **Deny or cancel** — human can refuse a specific request (tool continues) or abort the call entirely

## Transport Types

Three main transports, selected automatically by FastMCP:

### In-Memory

A `FastMCP` server instance runs in the same process with no subprocess or network:

```python
from langchain.mcp import MCPAdapter

server = FastMCP("weather")
@server.tool
def get_forecast(city: str) -> str:
    return f"{city}: sunny"

async with MCPAdapter(server) as adapter:
    tools = await adapter.list_tools()
```

**Ideal for**: tests, development, and single-app deployments with full control.

### Stdio

A script (Python or Node.js) is launched as a subprocess and communicates over stdin/stdout:

```python
from pathlib import Path
from langchain.mcp import MCPAdapter

script_path = Path("server.py")  # must exist
async with MCPAdapter(script_path) as adapter:
    tools = await adapter.list_tools()
```

**Ideal for**: local development, private tools, and sandboxing. Each adapter instance spawns one subprocess.

### HTTP (Streamable)

A remote MCP server is reached over HTTP(S) using a streaming transport:

```python
from langchain.mcp import MCPAdapter

url = "https://api.example.com/mcp"
async with MCPAdapter(url) as adapter:  # no auth
    tools = await adapter.list_tools()
```

**Ideal for**: public MCP servers (e.g., DeepWiki), cloud services, and third-party integrations.

## Multi-Server Setup: Tool Prefixing

To connect multiple MCP servers and expose all their tools to a single agent:

```python
from langchain.mcp import MCPAdapter

config = {
    "mcpServers": {
        "weather": {"command": "python", "args": ["weather_server.py"]},
        "calc": {"command": "python", "args": ["calc_server.py"]},
    }
}

async with MCPAdapter(config) as adapter:
    tools = await adapter.list_tools()  # ["weather_forecast", "calc_add", ...]
    agent = create_agent("anthropic:claude-sonnet-5", tools)
```

FastMCP automatically prefixes tools by config key (`weather_` + `forecast` = `weather_forecast`). This prevents collisions and makes tool provenance visible. The adapter's internal router ensures each call reaches the correct server. Servers can mix transports within one config: some stdio, some HTTP, some in-process.

## Authentication

MCP servers can require credentials. The adapter and client support:

- **Bearer token** — static token, no discovery or refresh
- **OAuth 2.1** — full flow with dynamic client registration, browser redirect, and token exchange
- **Custom auth** — any `httpx2.Auth` implementation

```python
from fastmcp.client import Client
from langchain.mcp import MCPAdapter

# Bearer token
async with MCPAdapter(Client("https://api.example.com/mcp", auth="token-value")) as adapter:
    tools = await adapter.list_tools()

# OAuth (opens browser, auto-approves on demo server)
async with MCPAdapter(Client("https://api.example.com/mcp", auth="oauth")) as adapter:
    tools = await adapter.list_tools()
```

For multi-server setups, specify auth per server in the `MCPConfig`:

```python
config = {
    "mcpServers": {
        "api1": {
            "command": "python",
            "args": ["server.py"],
            "auth": {"type": "bearer", "token": "secret-1"},
        },
        "api2": {
            "command": "python",
            "args": ["server.py"],
            "auth": {"type": "oauth"},
        },
    }
}
```

## Metadata and Tool Annotations

MCP tools can carry annotations (e.g., `destructiveHint=True` for deletion operations). These are surfaced on the LangChain tool as `metadata["mcp"]["tool"]["annotations"]`:

```python
@server.tool(annotations=ToolAnnotations(destructiveHint=True))
def delete_file(path: str) -> str:
    return f"Deleted {path}"
```

Clients can read this to gate destructive tools behind approval without hardcoding tool names:

```python
def _is_destructive(tool):
    annotations = (tool.metadata or {}).get("mcp", {}).get("tool", {}).get("annotations", {})
    return annotations.get("destructive_hint", False)

destructive_tools = [tool.name for tool in tools if _is_destructive(tool)]
# Pass to HumanInTheLoopMiddleware or similar approval gate
```

## Error Handling and Recovery

**MCP tool errors** (when a server reports `isError=True`):

- Converted to `ToolMessage` with `status="error"` and the server's message
- Visible to the model, which can correct inputs and retry
- Example: division by zero, file not found, network timeout at the remote server

**Transport errors** (network, subprocess failure, malformed response):

- Raised as exceptions; the run fails
- Models cannot act on these, so they should be retried at the orchestration level
- Example: unreachable URL, subprocess crashed, invalid JSON from server

## Long-Lived Adapters: Graph Factory Pattern

For per-request server setup (e.g., per-user credentials), create tools inside a graph factory:

```python
async def make_graph(runtime):
    user = runtime.user.identity
    auth = BearerAuth(token_for(user))
    group = ClientGroup({
        "api1": Client("https://api.example.com/mcp", auth=auth),
        "api2": Client("https://api.example.com/mcp", auth=auth),
    })
    tools = await MCPAdapter(group).list_tools()
    return create_agent("anthropic:claude-sonnet-5", tools)
```

For cross-run state (shared HTTP connection pool, response cache), instantiate outside the factory:

```python
_pool = httpx2.AsyncHTTPTransport()
_cache = InMemoryResponseCacheStore()

async def make_graph(runtime):
    user = runtime.user.identity
    group = ClientGroup({
        name: Client(
            StreamableHttpTransport(url, httpx_client_factory=_client_factory),
            cache=CacheConfig(store=_cache, partition=user),
        )
        for name, url in SERVERS.items()
    })
    tools = await MCPAdapter(group).list_tools(cache_mode="use")
    return create_agent("anthropic:claude-sonnet-5", tools)
```

## Protocol Eras

Two MCP protocol eras can coexist in one agent by using separate adapters per era:

```python
# Legacy era server (2025-11-25, handshake-based)
legacy_client = Client(legacy_server(), mode="legacy")

# Modern era server (2026-07-28, discovery-based)
modern_client = Client(modern_server(), mode="auto")

async with MCPAdapter(legacy_client) as legacy_adapter, \
           MCPAdapter(modern_client) as modern_adapter:
    tools = await legacy_adapter.list_tools() + await modern_adapter.list_tools()
    agent = create_agent("anthropic:claude-sonnet-5", tools)
```

A single `MCPConfig` fleet negotiates one era across all its members: if one member only speaks the legacy era, the whole fleet drops to it. Separate adapters ensure each server keeps the best era its connection supports.

## Examples

LangChain ships runnable examples in `examples/mcp/`:

| Example | Shows | Notes |
|---------|-------|-------|
| `transports.py` | in-memory, stdio, and HTTP transports | one adapter, three targets |
| `remote_server.py` | public MCP server (DeepWiki) | agent researches a GitHub repo |
| `multi_server.py` | `MCPConfig` fleet with tool prefixing | two stdio servers |
| `graph_factory.py` | per-user credentials in a `langgraph dev` graph | long-lived adapter, shared pool |
| `protocol_eras.py` | legacy and modern era servers together | separate adapters per era |
| `tool_errors.py` | tool failure and model recovery | agent retries on error |
| `elicitation.py` | server requesting user input mid-call | form elicitation and resume |
| `destructive_interrupt.py` | gating destructive tools | using `destructiveHint` metadata |
| `auth_bearer.py` | static bearer token | simple auth example |
| `auth_oauth.py` | OAuth 2.1 with dynamic client registration | full flow, browser redirect |

Run examples with:

```bash
uv sync --extra mcp --extra anthropic
export ANTHROPIC_API_KEY=...
uv run examples/mcp/transports.py
```

## Integration Points

### `create_agent`

Tools from `MCPAdapter.list_tools()` pass directly to `create_agent()`, which routes tool calls through the agent's model and executor. Tools remain callable after the adapter context exits because they hold a reference to the underlying client.

### LangGraph Checkpointer

Elicitation-driven interrupts require a checkpointer so the run can pause and resume:

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    "anthropic:claude-sonnet-5",
    tools,
    checkpointer=InMemorySaver(),
)
config = {"configurable": {"thread_id": "user-1"}}
paused = await agent.ainvoke({"messages": [...]}, config)
# Human answers interrupt; resume with command
resumed = await agent.ainvoke(Command(resume={...}), config)
```

### Tool Middleware

Agents can apply middleware to gate or log tool calls. MCP tool metadata (e.g., `destructiveHint`) integrates with `HumanInTheLoopMiddleware`:

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

interrupt_on = {
    tool.name: InterruptOnConfig(...)
    for tool in tools
    if _is_destructive(tool)
}
agent = create_agent(..., middleware=[HumanInTheLoopMiddleware(interrupt_on=interrupt_on)])
```

## Configuration and Operations

### Response Cache

FastMCP caches tool lists and supports per-principal isolation. The adapter's `cache_mode` parameter controls cache use:

- `"use"` (default) — serve from cache if fresh
- `"refresh"` — refresh from server, repopulate cache
- `"bypass"` — skip cache entirely

```python
tools = await adapter.list_tools(cache_mode="refresh")
```

For long-lived adapters, configure the cache on the client to persist across runs:

```python
cache = CacheConfig(
    store=InMemoryResponseCacheStore(),
    target_id="user-id",
    partition="user-partition"
)
client = Client(url, cache=cache)
tools = await MCPAdapter(client).list_tools(cache_mode="use")
```

### Logging and Observability

`MCPAdapter` and `as_langchain_tool()` are transparent to LangChain's logging and observability hooks. Tool calls are logged as `ToolMessage` events in the agent's message history. Elicitation interrupts and responses are visible in the run's state transitions.

## Invariants and Failure Semantics

- **Tool availability**: Once `list_tools()` completes, tools remain callable even after the adapter context exits (they hold the client)
- **Elicitation re-run**: When a tool is resumed with an answer, it is called again from the start. A server that works first and asks after repeats that work once per round
- **Error propagation**: Transport errors propagate as exceptions; MCP tool errors (isError=True) become model-visible `ToolMessage` errors
- **Client reuse**: Clients are reentrant; a tool can open its client even if a connection is already held elsewhere
- **Pre-built client cloning**: If a caller passes a client with an existing elicitation handler, it is cloned so the caller's object is never mutated
- **Group naming**: Tools from a `ClientGroup` are prefixed by config key; the router resolves each call to the correct member
- **No concurrent elicitation**: Elicitation answers are driven sequentially, one `interrupt()` per round, so LangGraph can match resume values by order

## Extension Points

- **Custom transport**: Pass any `fastmcp.ClientTransport` to support non-standard protocols
- **Custom auth**: Implement `httpx2.Auth` for authentication schemes beyond bearer token and OAuth
- **Custom metadata handler**: Subclass `StructuredTool` to customize how MCP metadata is exposed on the LangChain tool
- **Custom error handler**: Override `_handle_mcp_tool_error()` or provide your own `handle_tool_error` to the tool
- **Custom interruption**: Provide a pre-built client with your own `elicitation_handler` to override the interrupt-driven default

## Related Pages

- **[tools.md](/openwiki/tools.md)** — LangChain tool abstractions, `BaseTool`, `StructuredTool`
- **[agent-execution.md](/openwiki/agent-execution.md)** — agent orchestration, `create_agent`, tool routing
