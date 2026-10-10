---
type: "Reference"
title: "MCP (Model Context Protocol) Integration"
description: "MCPAdapter bridges MCP servers to LangChain agents, discovering tools, handling protocol negotiation via FastMCP, managing multiple transports, and supporting mid-call user input via LangGraph interrupts."
tags: [mcp, protocol, tools, adapter, integration, langgraph]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-10T08:25:28.570Z
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
generated: { by: "openwiki/0.5.0", at: "2026-09-28T08:35:20.640Z" }
---

## Overview

The Model Context Protocol (MCP) is a standard for LLM applications to discover and invoke tools exposed by external servers. `langchain.mcp` provides the `MCPAdapter` class, which discovers MCP tools and converts them to LangChain `BaseTool` objects suitable for use with agents. The adapter handles:

- **Protocol negotiation** via FastMCP (client session management, version handshake)
- **Multiple transports** (in-process, stdio subprocess, HTTP streaming)
- **Multi-server routing** via `ClientGroup` with automatic tool name prefixing
- **Mid-call user input** via LangGraph interrupts (elicitation)
- **Error recovery** by surfacing tool-reported errors to the model for retry

## MCP Concept

MCP is a request–response protocol where:

- **Clients** (like LangChain agents) discover and invoke tools on a server
- **Servers** expose tool catalogs, define tool schemas, and execute calls
- **Tools** are named, documented functions with typed arguments
- **Content** returned by a tool is represented as content blocks (text, images, files, structured data)
- **Protocol versions** evolve over time (e.g., 2025-11-25 uses `initialize` handshake; 2026-07-28 uses `server/discover`)

Servers from different protocol eras can coexist in one agent when separate adapters are used per server.

## Architecture: MCPAdapter and FastMCP

`MCPAdapter` is the main user-facing entry point. It wraps one or more FastMCP clients and exposes their discovered tools as LangChain tools:

```python
async with MCPAdapter(target) as adapter:
    tools = await adapter.list_tools()
    agent = create_agent("anthropic:claude-sonnet-5", tools)
```

### Target Types and Transport Inference

FastMCP infers the transport from the target. `MCPAdapter` accepts:

- **`str` (HTTP/HTTPS URL only)** — reached over streamable HTTP. String validation rejects non-URL strings (e.g., local file paths) to prevent silent local execution.
  ```python
  async with MCPAdapter("https://api.example.com/mcp") as adapter:
      tools = await adapter.list_tools()
  ```

- **`Path`** — launched as a subprocess over stdio. The script must exist.
  ```python
  async with MCPAdapter(Path("server.py")) as adapter:
      tools = await adapter.list_tools()
  ```

- **`FastMCP` instance** — in-process server with no network or subprocess. Ideal for tests and embedded deployments.
  ```python
  server = FastMCP("weather")
  @server.tool
  def forecast(city: str) -> str:
      return f"{city}: sunny"
  
  async with MCPAdapter(server) as adapter:
      tools = await adapter.list_tools()
  ```

- **`Client` or `ClientGroup`** — pre-built FastMCP client(s). Allows custom configuration (auth, cache, transport).
  ```python
  client = Client("https://api.example.com/mcp", auth="bearer-token")
  async with MCPAdapter(client) as adapter:
      tools = await adapter.list_tools()
  ```

- **`MCPConfig` (dict)** — multiple servers, each with independent transport and auth. FastMCP prefixes tools by config key to avoid collisions.
  ```python
  config = {
      "mcpServers": {
          "weather": {"command": "python", "args": ["weather_server.py"]},
          "calc": {"url": "https://api.example.com/mcp"},
      }
  }
  async with MCPAdapter(config) as adapter:
      tools = await adapter.list_tools()  # ["weather_forecast", "calc_add", ...]
  ```

- **`ClientTransport`** — explicit transport (HTTP, stdio, or custom) for fine-grained control.

## Tool Discovery and Conversion

`adapter.list_tools()` calls `fastmcp.Client.list_tools()` to fetch remote tools, then converts each via `as_langchain_tool()`:

### Discovery Pipeline

1. **Connection**: Enters the adapter context (connects underlying client(s))
2. **Fetch**: Calls `client.list_tools(cache_mode=...)` to fetch tool definitions
3. **Conversion**: For each MCP tool, calls `as_langchain_tool(tool, client)` to produce a LangChain tool

### Conversion Details

Each MCP tool becomes a `StructuredTool` with:

- **name, description, args_schema** — from the MCP tool definition
- **coroutine** — async function that calls the MCP tool through the client
- **response_format="content_and_artifact"** — returns both model-visible content blocks and structured data
- **metadata["mcp"]** — carries tool annotations, server identity, and destructive hints
- **handle_tool_error=_handle_mcp_tool_error** — handler to surface MCP tool errors to the model

### Multi-Server Tool Prefixing

When using `ClientGroup` or `MCPConfig`, tool names are prefixed by server key (e.g., `weather_forecast` = `weather` + `forecast`). This prevents collisions and makes tool provenance visible. The adapter's internal router ensures each call reaches the correct server:

```python
adapter = MCPAdapter(group)
tools = await adapter.list_tools()  # Tool names are prefixed
# e.g., ["weather_forecast", "weather_current_conditions", "calc_add"]

# When a tool is called, the router resolves it to the correct server member
for tool in tools:
    if tool.name == "weather_forecast":
        # Routed to the "weather" client by resolve_tool()
        result = await tool.ainvoke({"city": "Oslo"})
```

### Schema Normalization

`_normalize_mcp_schema()` keeps open object arguments open during provider schema conversion, adding `additionalProperties: True` to object properties without explicit property definitions. This allows models to pass arbitrary keys when the schema does not constrain them.

## Tool Invocation and Result Conversion

When a tool is called:

1. **Elicitation detection**: The adapter checks whether the underlying client is armed to drive LangGraph interrupts
2. **Tool call routing**: 
   - For `ClientGroup`, calls `resolve_tool(name)` to find the member client and upstream name
   - Calls `_call_tool_with_interrupts()` if client is interrupt-armed and server is modern (2026-07-28+)
   - Otherwise calls `fastmcp.Client.call_tool()` directly
3. **Result conversion**: Converts MCP content blocks to LangChain content blocks
4. **Error handling**: If the server reports `isError=True`, raises `_MCPToolExecutionError` (a `ToolException`), which becomes a `ToolMessage` with `status="error"` so the model can see and retry
5. **Artifacts**: Extracts `structured_content` (JSON, tables, etc.) into a separate artifact field

### Content Block Conversion

Supported content types:

- **Text** — plain string → `TextContentBlock`
- **Image** — base64 (`ImageContent`) or URL-referenced (`ResourceLink`) → `ImageContentBlock`
- **File** — base64 or URL-referenced → `FileContentBlock`
- **Resource** — embedded binary/text (`EmbeddedResource`) or URL link (`ResourceLink`) → `ImageContentBlock` (if MIME type is image/*) or `FileContentBlock`
- **Audio** — not yet supported (raises `NotImplementedError`)

## Elicitation: Mid-Call User Input

Some MCP tools cannot complete without asking the user a question mid-call (e.g., "Approve this action?"). Instead of hanging or erroring, the server returns an `InputRequiredResult` describing what it needs. The adapter converts this to a LangGraph `interrupt()`, so a human can answer and the run resumes seamlessly.

### Elicitation Flow

**Arming for Interrupts:**

1. On construction, `MCPAdapter` calls `_arm_for_interrupts()` on each underlying client
2. This sets `_declare_elicitation_capability` as the elicitation callback and marks the client with `_ARMED_MARKER`
3. FastMCP advertises the `elicitation` capability to the server (modern servers only)
4. Pre-built clients that already have their own handler are cloned first, so the caller's object is never mutated

**Interrupt Loop:**

1. `as_langchain_tool()` checks if the client is armed (`_drives_interrupts()`) and server is modern (protocol version ≥ 2026-07-28)
2. If yes, calls `_call_tool_with_interrupts()` instead of plain `call_tool()`
3. The loop:
   - Issues tool call with `allow_input_required=True`
   - If result is `InputRequiredResult`, extracts elicitation requests
   - Narrows requests to `ElicitRequest` only (rejects sampling, roots, continuation requests)
   - Raises `interrupt(payload)` with the request payload, pausing the run
   - On resume, receives answers keyed by request ID in the `responses` dict
   - Builds response payloads via `_build_responses()`, validating each answer
   - Retries the call with `input_responses=...` and `request_state=...` (opaque, echoed back)
   - Repeats until the tool returns a terminal result (not `InputRequiredResult`)

**Request Types:**

- **Form** — server asks for structured data (JSON matching a schema)
- **URL** — server asks the human to visit a URL (e.g., for approval or authentication)

**Response Actions:**

- **Accept** — answer the question (form: provide `content` matching schema; URL: none needed)
- **Decline** — refuse this request only (tool continues with other requests)
- **Cancel** — refuse entirely (abandon the tool call)

### Protocol Compatibility

The interrupt loop only runs on modern servers (2026-07-28 and later) that return `InputRequiredResult`. Legacy servers (2025-11-25) never trigger it, so they work unchanged. A pre-built client that already has a handler uses that handler instead, allowing custom elicitation strategies.

## Transport Types

Three main transports, selected automatically by FastMCP:

### In-Memory

A `FastMCP` server instance runs in the same process with no subprocess or network:

```python
from langchain.mcp import MCPAdapter
from fastmcp import FastMCP

server = FastMCP("weather")
@server.tool
def get_forecast(city: str) -> str:
    return f"{city}: sunny"

async with MCPAdapter(server) as adapter:
    tools = await adapter.list_tools()
```

**Ideal for**: tests, development, single-app deployments with full control.

### Stdio

A script (Python or Node.js) is launched as a subprocess and communicates over stdin/stdout:

```python
from pathlib import Path
from langchain.mcp import MCPAdapter

script_path = Path("server.py")  # must exist
async with MCPAdapter(script_path) as adapter:
    tools = await adapter.list_tools()
```

**Ideal for**: local development, private tools, sandboxing. Each adapter instance spawns one subprocess.

### HTTP (Streamable)

A remote MCP server is reached over HTTP(S) using a streaming transport:

```python
from langchain.mcp import MCPAdapter

url = "https://api.example.com/mcp"
async with MCPAdapter(url) as adapter:  # no auth
    tools = await adapter.list_tools()
```

**Ideal for**: public MCP servers, cloud services, third-party integrations.

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

async with MCPAdapter(config) as adapter:
    tools = await adapter.list_tools()
```

## Metadata and Tool Annotations

MCP tools can carry annotations (e.g., `destructiveHint=True` for deletion operations). These are surfaced on the LangChain tool as `metadata["mcp"]["tool"]["annotations"]`:

```python
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

server = MCPServer("example")

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

Tool metadata also includes `mcp.server` with server identity (name, version), allowing clients to distinguish tools by their origin.

## Error Handling and Recovery

### MCP Tool Errors

When a server reports `isError=True`:

- Converted to `ToolMessage` with `status="error"` and the server's message
- Visible to the model, which can correct inputs and retry
- Example: division by zero, file not found, network timeout at the remote server

### Transport Errors

Network, subprocess failure, or malformed response:

- Raised as exceptions; the run fails
- Models cannot act on these, so they should be retried at the orchestration level
- Example: unreachable URL, subprocess crashed, invalid JSON from server

## Multi-Server Setup and Configuration

### MCPConfig Fleet

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

### ClientGroup

For programmatic multi-server setup (e.g., per-request auth):

```python
from fastmcp import Client
from fastmcp.client.group import ClientGroup
from langchain.mcp import MCPAdapter

group = ClientGroup({
    "weather": Client("https://api1.example.com/mcp", auth="token-1"),
    "calc": Client("https://api2.example.com/mcp", auth="token-2"),
})

async with MCPAdapter(group) as adapter:
    tools = await adapter.list_tools()
```

## Protocol Eras and Version Negotiation

Two MCP protocol eras can coexist in one agent by using separate adapters per era:

```python
from fastmcp import Client
from langchain.mcp import MCPAdapter

# Legacy era server (2025-11-25, `initialize` handshake)
legacy_client = Client(legacy_server(), mode="legacy")

# Modern era server (2026-07-28, `server/discover` handshake)
modern_client = Client(modern_server(), mode="auto")

async with MCPAdapter(legacy_client) as legacy_adapter, \
           MCPAdapter(modern_client) as modern_adapter:
    tools = await legacy_adapter.list_tools() + await modern_adapter.list_tools()
    agent = create_agent("anthropic:claude-sonnet-5", tools)
```

A single `MCPConfig` fleet negotiates one era across all its members: if one member only speaks the legacy era, the whole fleet drops to it. Separate adapters ensure each server keeps the best era its connection supports.

## Graph Factory Pattern for Per-Request Setup

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

# Use with langgraph dev or persistent deployments
agent = CompiledStateGraph(make_graph)
```

For cross-run state (shared HTTP connection pool, response cache partitioned per user), instantiate outside the factory:

```python
import httpx2
from langchain.mcp import MCPAdapter
from fastmcp import Client
from fastmcp.client.group import ClientGroup
from mcp.client.caching import InMemoryResponseCacheStore, CacheConfig

_pool = httpx2.AsyncHTTPTransport()
_cache = InMemoryResponseCacheStore()

async def make_graph(runtime):
    user = runtime.user.identity
    group = ClientGroup({
        name: Client(
            url,
            httpx_client_factory=lambda: httpx2.AsyncClient(transport=_pool),
            cache=CacheConfig(store=_cache, partition=user),
        )
        for name, url in SERVERS.items()
    })
    tools = await MCPAdapter(group).list_tools(cache_mode="use")
    return create_agent("anthropic:claude-sonnet-5", tools)
```

## Response Cache and Cache Modes

FastMCP caches tool lists server-side and supports per-principal isolation. The adapter's `cache_mode` parameter controls cache use:

- **`"use"`** (default) — serve from cache if fresh (within server's TTL hint)
- **`"refresh"`** — refresh from server, repopulate cache
- **`"bypass"`** — skip cache entirely

```python
tools = await adapter.list_tools(cache_mode="refresh")
```

The cache and its per-principal isolation are configured on the client itself (`Client(cache=...)`); this parameter only selects how discovery reads it. Configured caches are honored — note that a bare `ClientGroup.list_tools()` defaults to `refresh`, while `MCPAdapter.list_tools()` defaults to `use`.

## Examples

LangChain ships runnable examples in `examples/mcp/`:

| Example | Shows | Model | Network |
|---------|-------|:-----:|:-------:|
| `transports.py` | in-memory, stdio, HTTP transports | | |
| `remote_server.py` | public MCP server (DeepWiki) | ✅ | ✅ |
| `multi_server.py` | `MCPConfig` fleet with tool prefixing | ✅ | |
| `graph_factory.py` | per-user credentials in a graph factory | | |
| `protocol_eras.py` | legacy and modern era servers together | ✅ | |
| `tool_errors.py` | tool failure and model recovery | ✅ | |
| `elicitation.py` | server requesting user input mid-call | ✅ | |
| `destructive_interrupt.py` | gating destructive tools via metadata | ✅ | |
| `auth_bearer.py` | static bearer token | | |
| `auth_oauth.py` | OAuth 2.1 with dynamic client registration | | |

Run examples with:

```bash
uv sync --extra mcp --extra anthropic
export ANTHROPIC_API_KEY=...
uv run examples/mcp/transports.py
```

## Integration Points

### `create_agent`

Tools from `MCPAdapter.list_tools()` pass directly to `create_agent()`, which routes tool calls through the agent's model and executor. Tools remain callable after the adapter context exits because they hold a reference to the underlying client.

```python
async with MCPAdapter(target) as adapter:
    tools = await adapter.list_tools()
    agent = create_agent("anthropic:claude-sonnet-5", tools)
    
# Tools are still callable after adapter context exits
result = await agent.ainvoke({"messages": [...]})
```

### LangGraph Interrupts and Checkpointer

Elicitation-driven interrupts require a checkpointer so the run can pause and resume:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langchain.mcp.elicitation import ELICITATION_INTERRUPT_TYPE

agent = create_agent(
    "anthropic:claude-sonnet-5",
    tools,
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "user-1"}}

# First invocation pauses on elicitation interrupt
paused = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Approve deletion of file X"}]},
    config,
)

# Check for interrupt
if paused.values.get("interrupt"):
    interrupt_data = paused.values["interrupt"]
    # Handle based on interrupt_data["type"] == ELICITATION_INTERRUPT_TYPE
    # ...
    
    # Resume with answers
    resumed = await agent.ainvoke(
        {"interrupt_answers": {...}},  # Depends on your graph's schema
        config,
    )
```

### Tool Metadata and Middleware

Agents can apply middleware to gate or log tool calls. MCP tool metadata (e.g., `destructiveHint`) integrates with `HumanInTheLoopMiddleware`:

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

def _is_destructive(tool):
    annotations = (tool.metadata or {}).get("mcp", {}).get("tool", {}).get("annotations", {})
    return annotations.get("destructive_hint", False)

interrupt_on = {
    tool.name: InterruptOnConfig(...)
    for tool in tools
    if _is_destructive(tool)
}

agent = create_agent(..., middleware=[HumanInTheLoopMiddleware(interrupt_on=interrupt_on)])
```

## Logging and Observability

`MCPAdapter` and `as_langchain_tool()` are transparent to LangChain's logging and observability hooks. Tool calls are logged as `ToolMessage` events in the agent's message history. Elicitation interrupts and responses are visible in the run's state transitions via LangGraph's built-in tracing.

## Invariants and Failure Semantics

- **Tool availability**: Once `list_tools()` completes, tools remain callable even after the adapter context exits (they hold the client)
- **Elicitation re-run**: When a tool is resumed with an answer, the entire call is re-executed from the start. A server that performs work before asking repeats that work once per elicitation round
- **Error propagation**: Transport errors propagate as exceptions; MCP tool errors (`isError=True`) become model-visible `ToolMessage` errors
- **Client reuse**: Clients are reentrant; a tool can open its client even if a connection is already held elsewhere
- **Pre-built client cloning**: If a caller passes a client with an existing elicitation handler, it is cloned so the caller's object is never mutated
- **Group routing**: Tools from a `ClientGroup` are prefixed by config key; the router resolves each call to the correct member via `resolve_tool()`
- **No concurrent elicitation**: Elicitation answers are driven sequentially, one `interrupt()` per round, so LangGraph can match resume values by order
- **String target validation**: A bare string target must be an http/https URL to prevent silent local file execution

## Extension Points

- **Custom transport**: Pass any `fastmcp.ClientTransport` to support non-standard protocols
- **Custom auth**: Implement `httpx2.Auth` for authentication schemes beyond bearer token and OAuth
- **Custom metadata handler**: Subclass `StructuredTool` to customize how MCP metadata is exposed on the LangChain tool
- **Custom error handler**: Override `_handle_mcp_tool_error()` or provide your own `handle_tool_error` to the tool
- **Custom elicitation**: Provide a pre-built client with your own `elicitation_handler` to override the interrupt-driven default

## Related Pages

- **[tools.md](/openwiki/tools.md)** — LangChain tool abstractions, `BaseTool`, `StructuredTool`, metadata
- **[agent-factory.md](/openwiki/agent-factory.md)** — agent orchestration, `create_agent`, tool routing, middleware
