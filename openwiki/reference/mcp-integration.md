---
type: "Reference"
title: "MCP Adapter Configuration and Protocol"
description: "Technical reference for MCPAdapter implementation, protocol handling, target types, elicitation mechanism, tool conversion, and error handling in LangChain MCP integration."
tags: [mcp, protocol, adapter, configuration, reference, fastmcp, tool-discovery, elicitation]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-07T08:15:07.286Z
sources:
  - id: openwiki-source-0a3228970b0eadc4bcadbb5d
    resource: repo://libs/langchain_v1/langchain/mcp/adapter.py
  - id: openwiki-source-b4c5eca79ce58abf486c2776
    resource: repo://libs/langchain_v1/langchain/mcp/elicitation.py
  - id: openwiki-source-4715c337e9b93b9d00846133
    resource: repo://libs/langchain_v1/langchain/mcp/tools.py
generated: { by: "openwiki/0.5.0", at: "2026-11-07T08:15:07.286Z" }
---

## Overview

`MCPAdapter` is the main user-facing class for integrating Model Context Protocol (MCP) servers with LangChain agents. It bridges protocol negotiation, tool discovery, execution, and bidirectional communication via FastMCP client lifecycle management. This reference documents the adapter's architecture, configuration, and internal mechanisms.

## MCPAdapter: Core Responsibilities

The `MCPAdapter` class handles these responsibilities:

1. **Target translation** — Accept diverse input types and delegate transport inference to FastMCP, with validation for string targets
2. **Client lifecycle** — Manage FastMCP client or group lifecycle through async context manager
3. **Tool discovery** — Call `list_tools()` on the underlying client and convert MCP tools to LangChain tools
4. **Elicitation setup** — Arm each underlying client to answer mid-call input requests via LangGraph interrupts
5. **Client mutation safety** — Clone pre-built clients before arming them so the caller's own object is never modified
6. **Tool invocation** — Convert discovered tools into `BaseTool` objects that invoke MCP tools asynchronously

## Target Types and Transport Inference

`MCPAdapterTarget` is a union type encompassing all valid inputs to `MCPAdapter.__init__()`:

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

### String Target Validation

A **bare `str` target must be an `http` or `https` URL**. This constraint prevents a subtle security issue: `fastmcp.Client` infers transport by testing the string as a filesystem path *before* testing it as a URL, so a string naming an existing `.py` or `.js` file silently launches that file as a subprocess. Since strings often arrive from configuration, request bodies, or model output rather than hardcoded code, `MCPAdapter` validates upfront with `_validate_url_target()`:

- Parses the string as a Pydantic `AnyUrl`
- Checks the scheme is exactly `"http"` or `"https"`
- Rejects Windows paths (e.g., `C:\server.py`), `file:` URLs, and other schemes
- Raises `ValueError` with a hint to use `Path()`, a transport, or `MCPConfig` for local servers

### FastMCP Target Types

For non-string targets, transport inference is delegated entirely to `FastMCPClient`:

- **`Path`** — Launched as a subprocess over stdio. FastMCP resolves the script and starts it.
- **`FastMCP` instance** — In-process server with no network or subprocess overhead.
- **`MCPServer`** — MCP SDK server instance, wrapped for in-process use.
- **`ClientTransport`** — Custom transport (e.g., WebSocket, gRPC) for advanced deployments.
- **`FastMCPClient` or `ClientGroup`** — Pre-built clients, optionally configured with auth, caching, or custom handlers.
- **`MCPConfig` or `dict[str, Any]`** — Multi-server configuration (e.g., from `claude_desktop_config.json`), each with independent transport and auth.
- **`AnyUrl`** — Pydantic URL object, forwarded to FastMCP.

## Initialization: Client Arming for Elicitation

When `MCPAdapter.__init__()` is called, it arms each underlying FastMCP client to advertise and handle the elicitation capability via LangGraph interrupts:

```python
def __init__(self, target: MCPAdapterTarget) -> None:
    def armed(client: FastMCPClient[Any]) -> FastMCPClient[Any]:
        if getattr(client, "_elicitation_callback", None) is not None:
            return client  # Caller's handler already set; honor it
        clone = client.new()  # Clone to avoid mutating caller's client
        _arm_for_interrupts(clone)
        return clone
    
    if isinstance(target, ClientGroup):
        members = {name: armed(client) for name, client in target.clients.items()}
        self._client = ClientGroup(members)
    elif isinstance(target, FastMCPClient):
        self._client = armed(target)
    else:
        if isinstance(target, str):
            _validate_url_target(target)
        client = FastMCPClient(target)
        _arm_for_interrupts(client)
        self._client = client
```

### Client Cloning and Mutation Safety

If a pre-built `FastMCPClient` is passed in, the adapter **clones it** via `client.new()` before arming. This ensures:

- The caller's original client is never modified
- The caller retains control over their own elicitation handler if they set one
- Multiple adapters can wrap the same pre-built client independently

A `ClientGroup` is treated as a peer object, not a transport, so each member client is armed independently.

### Respecting Caller's Elicitation Handler

If a pre-built client already carries a custom `_elicitation_callback` (set by the caller), the adapter returns it unchanged and does not arm it:

```python
if getattr(client, "_elicitation_callback", None) is not None:
    return client  # Use caller's handler instead
```

This allows callers with their own elicitation strategy (e.g., a callback-based handler) to opt out of interrupt-driven behavior.

## Lifecycle: Async Context Manager

`MCPAdapter` implements the async context manager protocol:

```python
async def __aenter__(self) -> Self:
    await self._client.__aenter__()
    return self

async def __aexit__(self, exc_type, exc_value, traceback) -> None:
    await self._client.__aexit__(exc_type, exc_value, traceback)
```

This delegates lifecycle to the underlying FastMCP client or group, which:

- On **enter** — Connects the transport (HTTP session, subprocess, in-process setup)
- On **exit** — Cleans up resources (closes connections, terminates subprocesses, finalizes in-process cleanup)

Callers must enter the context to connect before calling `list_tools()` or using any returned tools.

## Tool Discovery: `list_tools()`

```python
async def list_tools(self, *, cache_mode: CacheMode = "use") -> list[BaseTool]:
    async with self:
        remote_tools = await self._client.list_tools(cache_mode=cache_mode)
        return [await as_langchain_tool(tool, self._client) for tool in remote_tools]
```

This method:

1. **Enters the context** — Ensures the underlying client is connected
2. **Calls `Client.list_tools()`** — Queries the server's tool catalog, respecting the MCP client-side response cache per `cache_mode` (default `"use"`)
3. **Converts each tool** — Passes each MCP tool to `as_langchain_tool()`, which wraps the tool in a `StructuredTool` that invokes the MCP tool asynchronously
4. **Returns LangChain tools** — All returned tools hold a reference to the underlying client, so they remain callable after the adapter's context exits (the client may be re-entered as needed)

### Cache Mode Behavior

The `cache_mode` parameter (MCP SEP-2549) controls interaction with the client-side response cache:

- **`"use"`** (default) — Serve a cached tool list if present and within the server's TTL hint, otherwise fetch from the server
- **`"refresh"`** — Fetch from the server and repopulate the cache (ignoring any prior entry)
- **`"bypass"`** — Skip the cache entirely and fetch directly from the server

The default `"use"` differs from a bare `ClientGroup.list_tools()` call, which defaults to `"refresh"` to ensure fresh discovery.

## Tool Conversion: `as_langchain_tool()`

The `as_langchain_tool()` function converts an MCP tool to a `StructuredTool`:

```python
async def as_langchain_tool(
    tool: Tool,
    client: Client[Any] | ClientGroup,
) -> BaseTool:
```

### Tool Call Execution

Each converted tool has a `call_tool` coroutine that:

1. **Resolves the member client** — For a `ClientGroup`, resolves the tool's serving client via `client.resolve_tool()`
2. **Decides on interrupt driving** — Checks if the requesting client carries the adapter's interrupt sentinel via `_drives_interrupts()`
3. **Enters the client context** — Ensures the transport is connected
4. **Calls the MCP tool** — Either via `_call_tool_with_interrupts()` (interrupt-driven elicitation) or plain `client.call_tool()` (no mid-call input)
5. **Converts the result** — Transforms MCP content blocks and structured content into LangChain message content blocks and artifacts

### Interrupt-Driven Elicitation

If the client is armed with the adapter's interrupt sentinel and the server is modern-era (version ≥ 2026-07-28, which returns `InputRequiredResult`), the tool call drives the elicitation loop:

```python
if drives_interrupts:
    result = await _call_tool_with_interrupts(client, tool.name, arguments)
else:
    result = await client.call_tool(tool.name, arguments, raise_on_error=False)
```

### Result Conversion

`_convert_call_tool_result()` splits the MCP tool result into two pieces:

1. **Content blocks** — Converted from MCP types (`TextContent`, `ImageContent`, `ResourceLink`, etc.) to LangChain content blocks
2. **Structured content artifact** — If present, wrapped in `MCPToolArtifact` and attached to the `ToolMessage`

If the result is an error (`is_error=True`), an `_MCPToolExecutionError` is raised, allowing the error handler to surface the server's own error detail to the model.

### Schema Normalization

`_normalize_mcp_schema()` ensures open object schemas remain open during provider schema conversion. It adds `"additionalProperties": True` to any object property that lacks explicit property definitions or property constraint keywords. This prevents providers from rejecting inputs that the MCP server would accept.

### Tool Metadata

Each converted tool carries MCP-specific metadata under an `mcp` namespace:

```python
{
    "mcp": {
        "tool": {
            "annotations": {...},  # Tool-level annotations from MCP
            "_meta": {...}         # Tool's custom metadata
        },
        "server": {...}            # Server identity (name, version, etc.)
    }
}
```

This namespace keeps MCP provenance distinct from other tool metadata and preserves the server's identity for downstream reference.

## Elicitation: Mid-Call User Input via Interrupts

When an MCP server needs input mid-call (e.g., requesting user confirmation or authentication), the adapter answers via LangGraph `interrupt()` so a human can provide the answer and the call resumes.

### Arming: `_arm_for_interrupts()`

```python
def _arm_for_interrupts(client: Client[Any]) -> None:
    client.set_elicitation_callback(_declare_elicitation_capability)
    setattr(client, _ARMED_MARKER, True)
```

The adapter installs a **sentinel handler** (`_declare_elicitation_capability`) that only advertises the capability. It never handles requests; instead, the interrupt loop in `_call_tool_with_interrupts()` answers them.

This sentinel ensures FastMCP advertises `elicitation` (which it only does when a handler is set) while keeping the actual loop in the call frame where interrupt matching stays deterministic.

### Interrupt-Driving: `_call_tool_with_interrupts()`

```python
async def _call_tool_with_interrupts(
    client: Client[Any] | ClientGroup,
    tool_name: str,
    arguments: dict[str, Any],
) -> CallToolResult:
```

This function:

1. **Resolves the member session** — For a `ClientGroup`, finds the tool's serving client and its session
2. **Calls the tool with `allow_input_required=True`** — Instructs the session to return `InputRequiredResult` rather than driving the loop itself
3. **Loops on `InputRequiredResult`** — If the result contains input requests:
   - Filters to elicitation requests only (rejects sampling and roots as unsupported)
   - Builds an `MCPElicitationInterrupt` payload with the tool name and request list
   - Calls `interrupt(payload)` to unwind and prompt a human
   - On resume, assembles `InputResponses` from the resumed answers
   - Re-calls the tool with `input_responses` and `request_state` (opaque round state)
4. **Returns the terminal result** — Once the server returns a non-`InputRequiredResult`

### Interrupt Payloads and Responses

The interrupt payload structure:

```python
{
    "type": "mcp_elicitation",
    "tool_name": "example_tool",
    "requests": [
        {
            "key": "auth",
            "message": "Please provide your API key",
            "mode": "form",
            "requested_schema": {"type": "object", "properties": {...}}
        },
        {
            "key": "confirm",
            "message": "Visit this link to authorize",
            "mode": "url",
            "url": "https://auth.example.com/authorize?code=xyz"
        }
    ]
}
```

The resume payload (passed to `resume()`):

```python
{
    "responses": {
        "auth": {"action": "accept", "content": {"api_key": "secret123"}},
        "confirm": {"action": "accept"}
    }
}
```

Each response has an `action` (`"accept"`, `"decline"`, or `"cancel"`) and optional `content` (required for `"accept"`).

### Interrupt-Driving Sentinel

`_drives_interrupts()` determines whether a call should drive the interrupt loop:

```python
def _drives_interrupts(client: Client[Any]) -> bool:
    if not getattr(client, _ARMED_MARKER, False):
        return False
    version = getattr(client, "protocol_version", None)
    return version is not None and is_version_at_least(version, LATEST_MODERN_VERSION)
```

It returns `True` only if:

- The client carries the adapter's `_ARMED_MARKER` (set by `_arm_for_interrupts()`)
- The connected server reports a protocol version ≥ 2026-07-28 (modern era)

Legacy servers (< 2026-07-28) never return `InputRequiredResult`, so the loop never runs and calls fall back to plain `call_tool()`.

## Error Handling

MCP tool failures are surfaced to the model for retry rather than ending the run.

### Tool Execution Errors

When an MCP tool runs and sets `is_error=True`, `_convert_call_tool_result()` raises `_MCPToolExecutionError`, which carries the converted content blocks. The tool's `handle_tool_error` handler (`_handle_mcp_tool_error`) then:

1. Checks if the error is an `_MCPToolExecutionError`
2. If yes, returns the content blocks as the `ToolMessage` with `status="error"`, so the model sees the server's own error detail
3. If no (transport or conversion failure), re-raises the error to propagate it

### Error Summarization

`_summarize_tool_error()` builds a readable message from content blocks:

- Extracts text from any `TextContent` blocks
- Summarizes image and file blocks by count (never interpolates base64 payloads into error messages)
- Falls back to a generic message if no text is present

## Protocol Version Eras

MCP protocol versions evolve, and servers from different eras may coexist:

- **Modern era** (≥ 2026-07-28) — Returns `InputRequiredResult` for mid-call input requests, enabling interrupt-driven elicitation
- **Legacy era** (< 2026-07-28) — Initiates elicitation via callback, which interrupt-driven handlers cannot answer; requires a pre-built client with a custom elicitation handler

When `_drives_interrupts()` detects a legacy server, the call falls back to plain `call_tool()` without the interrupt loop.

## Client Access

The underlying FastMCP client or group can be accessed via the `client` property for advanced operations:

```python
@property
def client(self) -> FastMCPClient[Any] | ClientGroup:
    return self._client
```

This allows callers to:

- Inspect server info (`client.server_info`)
- Manage caching configuration
- Set up custom monitoring or logging
- Call other FastMCP methods directly

## Beta Status and Warnings

`langchain.mcp` is in beta and actively being developed. The namespace raises `LangChainBetaWarning` on import to notify users of potential API changes. Callers can silence this with:

```python
import warnings
from langchain_core._api import LangChainBetaWarning
warnings.filterwarnings("ignore", category=LangChainBetaWarning)
```

Or use the context manager:

```python
from langchain_core._api import suppress_langchain_beta_warning
with suppress_langchain_beta_warning():
    from langchain.mcp import MCPAdapter
```
