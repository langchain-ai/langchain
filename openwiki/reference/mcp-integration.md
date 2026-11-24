---
type: "Reference"
title: "MCPAdapter API Reference"
description: "API reference documentation for MCPAdapter, the main entry point for integrating MCP (Model Context Protocol) servers with LangChain agents, covering target types, method signatures, and lifecycle management."
tags: [mcp, protocol, tools, adapter, api-reference, fastmcp, langchain]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-13T08:16:35.902Z
sources:
  - id: openwiki-source-0a3228970b0eadc4bcadbb5d
    resource: repo://libs/langchain_v1/langchain/mcp/adapter.py
  - id: openwiki-source-3cde861e631d90545bbebcaa
    resource: repo://libs/langchain_v1/tests/unit_tests/mcp/test_adapter.py
generated: { by: "openwiki/0.5.0", at: "2026-11-13T08:16:35.902Z" }
---

## Overview

`MCPAdapter` is the main user-facing class for integrating MCP servers with LangChain agents. It bridges protocol negotiation via FastMCP, discovers remote tools, converts them to LangChain tools, and manages bidirectional communication including mid-call input via LangGraph interrupts.

## Class Signature

```python
class MCPAdapter:
    """Adapt an MCP target into LangChain tools.
    
    `MCPAdapter` uses FastMCP for protocol negotiation and connection management,
    then converts discovered MCP tools into asynchronous LangChain tools. The
    resulting tools can be passed directly to `create_agent`.
    """
    
    def __init__(self, target: MCPAdapterTarget) -> None:
        """Initialize the adapter around a FastMCP target, client, or group."""
        ...
```

## Target Types

### `MCPAdapterTarget` Union

The adapter accepts the following target types, defined by the `MCPAdapterTarget` type alias:

```python
MCPAdapterTarget: TypeAlias = (
    FastMCPClient[Any]
    | ClientGroup
    | ClientTransport
    | FastMCP
    | MCPServer
    | AnyUrl
    | Path
    | MCPConfig
    | dict[str, Any]
    | str
)
```

Each target type selects a different transport and connection model:

### HTTP/HTTPS URL String

A `str` target must be a valid `http` or `https` URL. String validation rejects non-URL strings (e.g., local file paths) to prevent silent local execution when configuration is untrusted.

- **Transport**: Streamable HTTP
- **Use case**: Remote MCP servers accessible over the network
- **Security**: Strings from untrusted sources (config, requests, model outputs) cannot accidentally spawn local subprocesses

```python
async with MCPAdapter("https://api.example.com/mcp") as adapter:
    tools = await adapter.list_tools()
```

**Validation rules:**
- Must parse as `AnyUrl` (Pydantic type)
- Scheme must be exactly `http` or `https` (no `file:`, `ws:`, or Windows drive letters like `C:`)
- Raises `ValueError` with a hint if rejected

### `Path`

A filesystem path to a local script. The script is launched as a subprocess and connected over stdio.

- **Transport**: Python stdio subprocess
- **Use case**: Local MCP servers, development and testing
- **Requirements**: Script must be executable; must exist on the filesystem

```python
async with MCPAdapter(Path("./weather_server.py")) as adapter:
    tools = await adapter.list_tools()
```

### `FastMCP` Instance

An in-process MCP server instance. No network or subprocess overhead. Ideal for embedded deployments, tests, and multi-server scenarios.

- **Transport**: In-process attachment
- **Use case**: Testing, embedded workflows, controlled server lifecycle
- **Benefits**: No I/O overhead, server state shared with caller

```python
from fastmcp import FastMCP

server = FastMCP("weather")

@server.tool
def forecast(city: str) -> str:
    """Get a 3-day forecast."""
    return f"{city}: sunny tomorrow"

async with MCPAdapter(server) as adapter:
    tools = await adapter.list_tools()
```

### `FastMCPClient[Any]`

A pre-built FastMCP client. Allows custom configuration including authentication, caching behavior, and custom transports before passing to the adapter.

- **Use case**: Advanced scenarios requiring custom client configuration
- **Mutation**: The adapter may clone and mutate the client (see elicitation behavior below)

```python
from fastmcp.client import Client

client = Client("https://api.example.com/mcp", auth="bearer-token")
async with MCPAdapter(client) as adapter:
    tools = await adapter.list_tools()
```

### `ClientGroup`

A group of multiple pre-built FastMCP clients, each connected to a different server. Tool names are automatically prefixed by server key to avoid collisions.

- **Use case**: Multi-server agents with centralized tool discovery
- **Prefixing**: Tools are named `{server_key}_{tool_name}` (e.g., `weather_forecast`, `calc_add`)
- **Routing**: Calls automatically route to the correct server member

```python
from fastmcp.client import Client
from fastmcp.client.group import ClientGroup

group = ClientGroup({
    "weather": Client(weather_server),
    "calc": Client(calc_server),
})

async with MCPAdapter(group) as adapter:
    tools = await adapter.list_tools()  # ["weather_forecast", "calc_add", ...]
```

### `MCPConfig` (dict) or `MCPConfig` Object

A configuration dictionary or object defining multiple MCP servers, each with independent transport, authentication, and connection settings. FastMCP's standard configuration format.

- **Format**: `{"mcpServers": {name: config_dict, ...}}`
- **Use case**: Complex multi-server deployments with mixed transports
- **Prefixing**: Like `ClientGroup`, tools are prefixed by config key

```python
config = {
    "mcpServers": {
        "weather": {
            "command": "python",
            "args": ["weather_server.py"],
        },
        "calc": {
            "url": "https://api.example.com/mcp",
        },
    }
}

async with MCPAdapter(config) as adapter:
    tools = await adapter.list_tools()
```

### `ClientTransport`

An explicit FastMCP transport instance (e.g., `StreamableHttpTransport`, `PythonStdioTransport`, custom). For fine-grained control over transport behavior.

- **Use case**: Advanced scenarios, custom protocol handling
- **When needed**: Rarely; most use cases are covered by other target types

### `MCPServer`

The MCP SDK's `MCPServer` type (from `mcp.server.mcpserver`). An in-process server object.

- **Transport**: In-process attachment
- **Compatibility**: Similar to `FastMCP` but uses the base SDK type

## Initialization Behavior

### Elicitation Arming

When initialized, `MCPAdapter` automatically prepares each underlying client to answer MCP elicitation (mid-call input requests) via LangGraph `interrupt()`:

1. **Pre-built clients without a handler**: Cloned (caller's object never mutated) and armed to drive the interrupt loop on each call.
2. **Pre-built clients with an existing handler**: Honored; the adapter leaves them as-is and does not apply arming.
3. **Clients built from targets**: Created fresh and armed.
4. **`ClientGroup` members**: Each member is cloned and armed independently.

```python
# These clients are cloned and armed
adapter = MCPAdapter(client)

# This client is left as-is (has a handler already)
custom_client = Client(target, elicitation_callback=my_handler)
adapter = MCPAdapter(custom_client)  # Adapter honors my_handler
```

### String Target Validation

String targets are validated to reject non-URLs:

- Parsed as `AnyUrl` (Pydantic)
- Scheme must be `http` or `https`
- Raises `ValueError` with a helpful hint if rejected

This prevents a configuration string from silently selecting local file execution, which FastMCP allows but MCP adapter forbids.

## Instance Properties

### `client` Property

```python
@property
def client(self) -> FastMCPClient[Any] | ClientGroup:
    """Return the underlying FastMCP client or group for advanced MCP operations."""
```

**Returns**: The underlying FastMCP `Client` or `ClientGroup`.

**Use case**: Direct access for advanced operations (e.g., calling `call_tool()` directly, inspecting transport, custom error handling).

**Lifecycle**: Valid for the lifetime of the adapter; remains valid after the adapter context exits.

```python
adapter = MCPAdapter(target)
# Direct tool invocation
result = await adapter.client.call_tool("tool_name", arguments={})
```

## Lifecycle Methods

### `__aenter__` and `__aexit__`

The adapter is an async context manager. Entering the context establishes the underlying client connection(s); exiting releases them.

```python
async with MCPAdapter(target) as adapter:
    # Client(s) are connected here
    tools = await adapter.list_tools()
    # Use tools
# Client(s) are disconnected here
```

**Behavior**:
- **Enter (`__aenter__`)**: Calls the underlying client's `__aenter__()`, establishing transport connections (socket for HTTP, subprocess for stdio, etc.).
- **Exit (`__aexit__`)**: Calls the underlying client's `__aexit__()`, releasing connections and cleaning up resources.
- **Nesting**: `ClientGroup` tracks nesting, so multiple `list_tools()` calls inside one context are safe.

**Important**: Tools returned by `list_tools()` hold a reference to the client and remain callable after the context exits (clients auto-reconnect on demand).

```python
async with MCPAdapter(target) as adapter:
    tools = await adapter.list_tools()

# Tools still work here; client reconnects on first call
result = await tools[0].ainvoke({...})
```

## Methods

### `list_tools()`

Discover and adapt MCP tools for use with LangChain.

```python
async def list_tools(
    self, 
    *, 
    cache_mode: CacheMode = "use"
) -> list[BaseTool]:
    """Discover and adapt MCP tools for use with LangChain.
    
    Args:
        cache_mode: How discovery interacts with the client-side response
            cache (SEP-2549). `use` serves a cached tool list when one is
            present and still within the server's TTL hint, `refresh` calls
            the server and repopulates the cache, and `bypass` skips the
            cache entirely. The cache and its per-principal isolation are
            configured on the client itself (`Client(cache=...)`); this only
            selects how discovery reads it. Defaults to `use` so a
            configured cache is honored — note this differs from a bare
            `ClientGroup.list_tools()`, whose own default is `refresh`.
    
    Returns:
        list[BaseTool]: LangChain tools that invoke the corresponding MCP tools
            asynchronously. Each holds the adapter's client, so the tools
            stay callable after this adapter's context exits.
    """
```

**Implementation**:

1. Enters the adapter context (connects client if needed)
2. Calls `self._client.list_tools(cache_mode=cache_mode)` to fetch tool definitions
3. Converts each MCP tool via `as_langchain_tool(tool, self._client)` to a LangChain `StructuredTool`
4. Returns the list of LangChain tools

**Cache modes**:

- `"use"` (default): Serve cached results if the server's TTL permits, otherwise call the server
- `"refresh"`: Call the server and repopulate the cache
- `"bypass"`: Skip the cache entirely

**Behavior with `ClientGroup`**: When the adapter wraps a `ClientGroup`, each tool name is automatically prefixed by its server's config key (e.g., `weather_forecast` for tool `forecast` in the `weather` server).

**Tool metadata**: Each returned tool carries metadata linking it back to its server, including:
- Tool annotations (description, input schema)
- Server identity
- Destructive operation hints (for interrupt gating)

**Error handling**: Surfaces MCP tool errors to the model as failed tool output (ToolMessage with status="error"), allowing the model to read the error and retry.

**Example**:

```python
async with MCPAdapter("https://api.example.com/mcp") as adapter:
    tools = await adapter.list_tools()
    
    # Use with an agent
    agent = create_agent("anthropic:claude-sonnet-5", tools)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": "..."}]})

# Or with custom cache mode
async with MCPAdapter(target) as adapter:
    tools = await adapter.list_tools(cache_mode="refresh")  # Force server call
```

## Related Types and Utilities

### `MCPToolArtifact`

```python
class MCPToolArtifact(TypedDict):
    """Artifact attached to the `ToolMessage` produced by an MCP tool call.
    
    Attributes:
        structured_content: The `structuredContent` of the MCP tool result.
    """
    structured_content: Any
```

Wraps structured data returned by an MCP tool, accessible via `ToolMessage.artifact`.

### `as_langchain_tool()`

```python
async def as_langchain_tool(
    tool: Tool,
    client: FastMCPClient[Any] | ClientGroup,
    server_key: str | None = None,
) -> BaseTool:
    """Convert one MCP tool into a LangChain StructuredTool."""
```

**Internal use**: Called by `list_tools()` for each discovered tool. Normally not called directly.

**Behavior**:
- Reads the tool's input schema and creates a LangChain `StructuredTool`
- Wraps the tool in an async function that calls the MCP tool through the client
- Attaches metadata including annotations, server identity, and destructive hints
- Configures error handling to surface MCP errors to the model
- Supports mid-call elicitation via interrupt-driven input requests

### Elicitation Support

The adapter arms every client to answer MCP elicitation (mid-call input requests) via LangGraph `interrupt()`:

- **Automatic**: No configuration needed; the adapter does this by default
- **Interrupt type**: `"mcp_elicitation"` (available via `langchain.mcp.elicitation.ELICITATION_INTERRUPT_TYPE`)
- **Payload**: Contains the tool name and list of input requests (form or URL)
- **Resumption**: The run resumes with user-provided answers; the tool call completes and results flow back to the model

See `langchain.mcp.elicitation` for interrupt payload shapes and handler patterns.

## Validation and Error Handling

### String Target Rejection

A `str` target is validated to prevent silent local execution:

```python
# ✅ Accepted
MCPAdapter("https://example.com/mcp")
MCPAdapter("http://localhost:2024/mcp")

# ❌ Rejected (ValueError)
MCPAdapter("server.py")  # Non-URL; would launch subprocess
MCPAdapter("./server.js")  # Local path
MCPAdapter("example.com/mcp")  # Missing scheme
MCPAdapter("file:///etc/passwd")  # File scheme
MCPAdapter(r"C:\Users\me\server.py")  # Windows path (scheme `c`)
```

**Error message hints** what to do instead:
- Use `Path("server.py")` for local scripts
- Use an explicit `fastmcp` transport for fine-grained control
- Use `MCPConfig` for complex scenarios

### Tool Invocation Errors

When a tool call fails, the error is wrapped as a `ToolMessage` with `status="error"`, exposing the MCP server's error detail to the model so it can read, understand, and retry:

```python
# Tool returns is_error=True
result = await tool.ainvoke({...})

# The ToolMessage carries the error content with status="error"
# Model sees the error and can retry or adjust the call
```

## Usage Patterns

### Single Server

```python
from langchain.mcp import MCPAdapter
from langchain.agents import create_agent

async with MCPAdapter("https://api.example.com/mcp") as adapter:
    tools = await adapter.list_tools()
    agent = create_agent("anthropic:claude-sonnet-5", tools)
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What's the weather?"}]
    })
```

### Multiple Servers

```python
from fastmcp.client import Client
from fastmcp.client.group import ClientGroup

group = ClientGroup({
    "weather": Client("https://weather.example.com/mcp"),
    "calc": Client("https://calc.example.com/mcp"),
})

async with MCPAdapter(group) as adapter:
    tools = await adapter.list_tools()
    # Tools are prefixed: ["weather_forecast", "weather_current", "calc_add", ...]
    agent = create_agent("anthropic:claude-sonnet-5", tools)
```

### In-Process Server (Testing)

```python
from fastmcp import FastMCP

server = FastMCP("calc")

@server.tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

async with MCPAdapter(server) as adapter:
    tools = await adapter.list_tools()
    # Use tools directly in tests
```

### Local Subprocess Server

```python
async with MCPAdapter(Path("./weather_server.py")) as adapter:
    tools = await adapter.list_tools()
```

### Custom Client with Authentication

```python
from fastmcp.client import Client

client = Client(
    "https://private.example.com/mcp",
    auth="bearer-token-xyz"
)

async with MCPAdapter(client) as adapter:
    tools = await adapter.list_tools()
```

## Important Invariants

1. **Tools survive adapter context exit**: Tools returned by `list_tools()` hold a reference to the client and remain callable after the adapter's async context exits.

2. **Caller objects are not mutated**: When a pre-built `Client` or `ClientGroup` is passed to `MCPAdapter`, the adapter clones them before arming for elicitation. The caller's original object is never modified.

3. **String targets must be URLs**: A string target is assumed to be a URL; it is validated and rejected if it is not `http(s)`. This prevents configuration strings from accidentally selecting local execution.

4. **Multi-server tool prefixing is automatic**: When using `ClientGroup` or `MCPConfig`, tool names are automatically prefixed by server key (e.g., `weather_forecast`). Calls automatically route to the correct server.

5. **Elicitation is answered via interrupts**: By default, servers that request mid-call input are answered using LangGraph `interrupt()`, allowing a human to provide answers and the run to resume without manual loop management.

6. **Caching defaults to "use"**: The `cache_mode` parameter defaults to `"use"`, which respects client-side response caches. This differs from a bare `ClientGroup.list_tools()` which defaults to `"refresh"`.

## Import Path

```python
from langchain.mcp import MCPAdapter
```

**Note**: The `langchain.mcp` namespace is in beta and raises a `LangChainBetaWarning` on import. Suppress it if desired:

```python
import warnings
from langchain_core._api import LangChainBetaWarning

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
from langchain.mcp import MCPAdapter
```

## See Also

- **Integration guide**: `/openwiki/integrations/mcp-integration.md` — Full MCP integration documentation including protocol concepts, architecture, tool discovery pipeline, and practical examples
- **Elicitation details**: `langchain.mcp.elicitation` — Interrupt types, payload shapes, and handler patterns for mid-call input
- **Tool conversion**: `langchain.mcp.tools` — `as_langchain_tool()` implementation and `MCPToolArtifact`
