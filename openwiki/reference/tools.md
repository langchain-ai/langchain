---
type: Reference
title: Tools Interface and Binding Reference
description: Reference documentation for LangChain tool interfaces, decorators, binding strategies, and runtime injection mechanisms that enable structured agent interactions with external systems.
tags: [tool, interface, decorator, schema, binding, injection, runnable]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-07T08:15:07.286Z
sources:
  - id: openwiki-source-4ff475d7b00540f962384251
    resource: repo://libs/core/langchain_core/tools/base.py
generated: { by: "openwiki/0.5.0", at: "2026-11-07T08:15:07.286Z" }
---

## Overview

This reference documents the core tool interfaces and mechanisms in LangChain's tools system, including the `BaseTool` protocol, concrete implementations (`Tool`, `StructuredTool`), the `@tool` decorator, schema generation and validation, tool binding to chat models, and runtime argument injection patterns.

## BaseTool: Core Interface

`BaseTool` is the abstract base class that all LangChain tools implement, extending `RunnableSerializable[str | dict[str, Any] | ToolCall, Any]`. Every tool represents an action an agent or model can call with structured input, error handling, and callback integration.

### Required Properties

- **`name: str`** — Unique identifier that clearly communicates the tool's purpose. Used by models and agents for tool selection.
- **`description: str`** — Human-readable explanation of when, why, and how to use the tool. Guides model decision-making.
- **`args_schema: ArgsSchema | None`** — Pydantic `BaseModel` class, Pydantic v1 `BaseModel` subclass, or JSON schema dict specifying valid input arguments. Optional if the tool has no parameters.

### Execution and Error Handling

- **`return_direct: bool`** (default: `False`) — When `True`, the agent stops looping immediately after the tool executes (terminal action). Used for final response tools.
- **`handle_tool_error: bool | str | Callable[[ToolException], str | list[dict]]`** (default: `False`) — Strategy for handling `ToolException`:
  - `False`: Re-raise the exception.
  - `True`: Return the exception message as tool output.
  - `str`: Return a fixed message as tool output.
  - `Callable`: Invoke the callable with the exception; return value becomes tool output (string or message content blocks).
- **`handle_validation_error: bool | str | Callable[[ValidationError], str]`** (default: `False`) — Strategy for handling Pydantic validation errors during input parsing (similar options as `handle_tool_error`).
- **`response_format: Literal["content", "content_and_artifact"]`** (default: `"content"`) — Output format:
  - `"content"`: Tool output is the sole content of a `ToolMessage`.
  - `"content_and_artifact"`: Tool output must be a `(content, artifact)` tuple for structured output with optional artifacts.

### Callbacks and Metadata

- **`callbacks: Callbacks`** — Lifecycle callbacks (`on_tool_start`, `on_tool_end`, `on_tool_error`) for tracing, monitoring, and observability.
- **`tags: list[str] | None`** — Optional semantic labels attached to all tool invocations for filtering and metrics.
- **`metadata: dict[str, Any] | None`** — Custom application-specific metadata passed to callbacks.
- **`verbose: bool`** (default: `False`) — Whether to log tool execution progress.
- **`extras: dict[str, Any] | None`** — Provider-specific configuration not covered by standard fields (e.g., Anthropic `cache_control`, `defer_loading`, OpenAI-specific settings).

## Tool Implementations

### Tool: String-Based Interface

`Tool` is a concrete implementation for functions with simple string input/output contracts. It accepts a callable or coroutine directly without schema inference.

**Key Properties:**
- **`func: Callable[..., str] | None`** — Synchronous function that accepts tool input and returns a string.
- **`coroutine: Callable[..., Awaitable[str]] | None`** — Async function equivalent of `func`.
- **`description: str`** — Defaults to empty string; must be provided or inferred from docstring.
- **`args: dict[str, Any]`** — Returns `{"tool_input": {"type": "string"}}` if no `args_schema` is set, or derives schema from `args_schema` if provided. Used for backward compatibility when schema is not available.

**When to Use:** For simple functions that operate on a single string argument without complex input validation, or when integrating legacy code without type hints.

### StructuredTool: Multi-Argument Schema-Based Interface

`StructuredTool` is the primary tool implementation for functions with multiple typed parameters, automatic schema inference, and structured validation.

**Key Properties:**
- **`func: Callable[..., Any] | None`** — Synchronous function to execute.
- **`coroutine: Callable[..., Awaitable[Any]] | None`** — Async function equivalent.
- **`args_schema: ArgsSchema`** (required) — Pydantic `BaseModel` or JSON schema dict defining input arguments.
- **`description: str`** — Defaults to empty string; inherited from docstring or schema if not provided.

**Creation via `from_function()`:**
```python
StructuredTool.from_function(
    func: Callable[..., Any] | None,
    coroutine: Callable[..., Awaitable[Any]] | None = None,
    name: str | None = None,
    description: str | None = None,
    return_direct: bool = False,
    args_schema: ArgsSchema | None = None,
    infer_schema: bool = True,
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = False,
) -> StructuredTool
```

**When to Use:** For functions with multiple parameters, complex inputs, or when automatic schema inference from type hints is desired.

## Tool Creation: The `@tool` Decorator

The `@tool` decorator converts Python functions and `Runnable` objects into `BaseTool` instances. It supports multiple signatures and can be used as a bare decorator or with keyword arguments.

### Usage Patterns

**Bare Decorator (Type Hints Required):**
```python
@tool
def search(query: str) -> str:
    """Search for information about a query."""
    return api.search(query)
```

**With Custom Description:**
```python
@tool("custom_name", description="Search the API for information.")
def search(query: str) -> str:
    return api.search(query)
```

**Async Function:**
```python
@tool
async def fetch_data(url: str) -> str:
    """Fetch data from a URL."""
    return await http_client.get(url)
```

**Runnable Conversion (Requires String Name):**
```python
@tool("model_runner")
def my_runnable: my_chain
```

### Decorator Parameters

- **`name_or_callable: str | Callable | None`** — Override tool name (string) or the function to decorate (callable). If omitted, function `__name__` is used.
- **`runnable: Runnable[Any, Any] | None`** — Optional `Runnable` to convert (must provide a string name as first positional argument).
- **`description: str | None`** — Tool description. If omitted, uses function docstring (highest precedence), then schema description.
- **`return_direct: bool`** (default: `False`) — Stop agent loop after execution.
- **`args_schema: ArgsSchema | None`** — Explicit input schema (Pydantic model or JSON schema dict). If omitted, inferred from signature.
- **`infer_schema: bool`** (default: `True`) — Whether to infer schema from function signature. If `False`, function must have a docstring and `args_schema` must be `None`.
- **`response_format: Literal["content", "content_and_artifact"]`** (default: `"content"`) — Output format configuration.
- **`parse_docstring: bool`** (default: `False`) — Parse Google-style docstrings for parameter descriptions (only if `infer_schema=True`).
- **`error_on_invalid_docstring: bool`** (default: `True`) — Raise on invalid Google-style docstrings if `parse_docstring=True`.
- **`extras: dict[str, Any] | None`** — Provider-specific configuration passed to chat models.

### Result Type

The decorator returns a `StructuredTool` (if `infer_schema=True` or `args_schema` provided) or a `Tool` (if `infer_schema=False`).

## Input Schema Generation and Validation

Tools validate input using Pydantic models built from function signatures or explicitly provided schemas.

### Schema Inference Pipeline

**Priority Order:**
1. Explicit `args_schema` passed to tool decorator or factory.
2. JSON schema dict if `args_schema` is already a dict.
3. Inferred from function signature via `create_schema_from_function()` if `infer_schema=True`.

**Inference Process** (when `infer_schema=True`):
1. Extract type hints from function signature using `get_type_hints()` with `Annotated` support.
2. Parse function docstring (if `parse_docstring=True`) following Google style for parameter descriptions.
3. Merge descriptions from: `Annotated` metadata → docstring `Args` section → none.
4. Filter reserved parameter names (`run_manager`, `callbacks`, `config`) and automatically injected arguments (`InjectedToolArg`, `InjectedToolCallId`, `ToolRuntime`).
5. Generate Pydantic model class with `extra="forbid"` and `arbitrary_types_allowed=True`.

### Schema Caching and Memoization

The `tool_call_schema` property builds and caches a schema class excluding injected arguments:
- Memoized in `_tool_call_schema_memo` per tool instance.
- Schema class's `model_json_schema()` method is patched to cache the generated dict, preventing expensive regeneration on every agent loop.
- Memo is cleared when `name`, `description`, or `args_schema` is reassigned.

### Input Parsing and Validation

During execution, `_parse_input()` handles input type conversions:
- **String input** with single-argument schema: Mapped to the single field.
- **String input** with JSON schema: Raises `ValueError` (not allowed).
- **Dict input**: Validated via Pydantic model; annotated field descriptions provide documentation.
- **Injected arguments** (identified by signature): Re-injected from tool metadata or invocation context (e.g., `tool_call_id`).
- **Validation errors**: Handled according to `handle_validation_error` configuration.

## Runtime Argument Injection

Tools can receive arguments that are not provided by the model but are injected at runtime by the framework.

### Annotation-Driven Injection

Arguments annotated with `InjectedToolArg` or its subclasses are automatically excluded from the schema sent to models and re-injected during execution:

**`InjectedToolArg`** — Base marker for explicitly injected arguments:
```python
from typing import Annotated
from langchain_core.tools import tool, InjectedToolArg

@tool
def my_tool(x: str, context: Annotated[dict, InjectedToolArg]) -> str:
    """Tool that uses injected context."""
    return f"{x}: {context['user_id']}"
```

**`InjectedToolCallId`** — Injects the tool call ID from the model invocation:
```python
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage

@tool
def my_tool(x: int, tool_call_id: Annotated[str, InjectedToolCallId]) -> ToolMessage:
    """Return x with tool call ID for tracing."""
    return ToolMessage(str(x), name="my_tool", tool_call_id=tool_call_id)
```

### Direct Injection via Type Annotation

`ToolRuntime` is a directly injected argument (does not use `Annotated`), providing access to runtime state:

```python
from langchain_core.tools import tool, ToolRuntime

@tool
def my_tool(x: str, runtime: ToolRuntime) -> str:
    """Access runtime state, context, and store."""
    return f"{x}: {runtime.state}, {runtime.context}, {runtime.store}"
```

**`ToolRuntime` Attributes:**
- **`state`** — Mutable state shared across tool invocations.
- **`context`** — Immutable context passed from the parent runnable.
- **`store`** — Persistent key-value store for agent memory and data.
- **`config`** — Current `RunnableConfig` for this tool execution.

### Filtering Injected Arguments

The `_filter_injected_args()` method removes injected arguments from callback input to prevent exposing internal state to tracing/monitoring systems.

## Tool Binding to Chat Models

Chat models expose a `bind_tools()` method to configure them for tool calling. This method returns a new `Runnable` that accepts the same inputs but instructs the model to call tools from the provided list.

### bind_tools() Interface

```python
def bind_tools(
    self,
    tools: Sequence[dict | type | Callable | BaseTool],
    *,
    tool_choice: str | None = None,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, AIMessage]
```

**Parameters:**
- **`tools`** — List of tools the model can call. Each tool can be:
  - A `BaseTool` instance.
  - A Pydantic class or callable (auto-converted to a tool schema).
  - A dict (raw tool/function schema).
- **`tool_choice: str | None`** — Control tool selection:
  - `None`: Model may choose to use tools or not.
  - `"any"`: Model must use a tool.
  - `"required"`: Model must use a tool (provider-dependent).
  - Tool name: Model must use the specified tool.
- **`**kwargs`** — Provider-specific options (e.g., `tool_search_enabled` for Anthropic).

**Return Value:** A new `Runnable` that generates `AIMessage` objects with tool calls in `additional_kwargs['tool_calls']`.

### Tool Rendering and Schema Conversion

Before sending tools to the model, they are converted to provider-specific formats. Utility functions for rendering:

**`render_text_description(tools: list[BaseTool]) -> str`**
Renders tool names and descriptions in plain text for prompt injection:
```txt
search: This tool searches the web
calculate: Performs mathematical calculations
```

**`render_text_description_and_args(tools: list[BaseTool]) -> str`**
Renders tool names, descriptions, and argument schemas in plain text:
```txt
search: This tool searches the web, args: {"query": {"type": "string"}}
calculate: Performs math, args: {"expression": {"type": "string"}}
```

## Execution Lifecycle

### Synchronous Execution Path

1. **`invoke(input, config)`** → Calls `run(tool_input, **kwargs)` via `_prep_run_args()`.
2. **`run(tool_input, ...)`** — Main execution method:
   - Configures callbacks (`CallbackManager.configure()`).
   - Fires `on_tool_start` callback.
   - Parses and validates input via `_parse_input()`.
   - Converts input to args/kwargs via `_to_args_and_kwargs()`.
   - Injects `run_manager` and config parameters if function signature accepts them.
   - Executes `_run()` within callback context.
   - Handles response format (`"content"` or `"content_and_artifact"`).
   - Catches and handles `ToolException` and validation errors.
   - Fires `on_tool_end` or `on_tool_error` callback.
   - Returns output (string, dict, `ToolMessage`, or tuple for artifacts).
3. **`_run(*args, **kwargs)`** — Abstract method implemented by subclasses to execute the wrapped function/coroutine.

### Asynchronous Execution Path

- **`ainvoke(input, config)`** → Calls `arun(tool_input, **kwargs)` via `_prep_run_args()`.
- **`arun()`** — Async equivalent of `run()` with same structure but uses `AsyncCallbackManager` and calls `_arun()`.
- **`_arun(*args, **kwargs)`** — Async implementation; falls back to sync with executor if not defined.

## Configuration and Composition

### Tool Binding with Explicit Configuration

Tools can be bound to chat models with configuration:
```python
model = ChatOpenAI(model="gpt-4")
bound_model = model.bind_tools(
    [search_tool, calculator_tool],
    tool_choice="any"
)
```

### Reserved Parameter Names

Functions should avoid these parameter names, which are reserved for framework injection:
- **`config`** — Shadowed by injected `RunnableConfig`; causes `TypeError` at call time.
- **`run_manager`** — Silently filtered from schema and injected from callbacks.
- **`callbacks`** — Silently filtered from schema and injected from callbacks.

Use `ToolRuntime` instead to access runtime information.

### Schema-Only Mode (No Execution)

The `convert_runnable_to_tool()` function wraps a `Runnable` as a tool without changing execution semantics. The `@tool` decorator uses this internally when decorating runnables.

## Error Handling and Validation

### Tool Exception Handling

Tools catch exceptions during execution:
- **`ToolException`** — Custom exception for tool-specific errors, handled per `handle_tool_error`.
- **Validation errors** — Caught during input parsing, handled per `handle_validation_error`.
- **Other exceptions** — May be re-raised or wrapped in `ToolMessage` with `status="error"` depending on error handling config.

### Callback Integration

When an error occurs:
- `on_tool_error(event)` is fired with the exception.
- Handler output (if applicable) becomes the `ToolMessage` content with `status="error"`.
- Exception traceback and context are available to tracing systems.

## Key Invariants and Guarantees

1. **Schema Stability** — `args_schema` is immutable once set; reassignment clears memoized schema.
2. **Injection Isolation** — Injected arguments are excluded from model-facing schemas and not exposed in callbacks unless explicitly handled.
3. **Callback Lifetime** — `on_tool_start` fires before validation; `on_tool_end` fires after successful execution; `on_tool_error` fires on exception.
4. **Input Parsing** — Input is parsed and validated exactly once per invocation; validation errors are handled per configuration.
5. **Async Support** — All tools support both sync and async invocation; async-only tools fall back to executor if sync is called.
