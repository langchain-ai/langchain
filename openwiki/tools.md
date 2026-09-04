---
type: "Reference"
title: "Form 1: No arguments (name from function)"
openwiki_generated: true
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:18:34.589Z
sources:
  - id: openwiki-source-9861ba5cf0c42c142cf732f9
    resource: repo://libs/core/langchain_core/messages/tool.py
  - id: openwiki-source-4ff475d7b00540f962384251
    resource: repo://libs/core/langchain_core/tools/base.py
  - id: openwiki-source-9c422fcb5ac12738f17d1cd1
    resource: repo://libs/core/langchain_core/tools/convert.py
  - id: openwiki-source-1ab4436ccb637ddf41e35732
    resource: repo://libs/core/langchain_core/tools/render.py
  - id: openwiki-source-80e84f93417c922f44011393
    resource: repo://libs/core/langchain_core/tools/simple.py
  - id: openwiki-source-b816e651a5890bde13cf8013
    resource: repo://libs/core/langchain_core/tools/structured.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:18:34.589Z" }
---


## Overview

LangChain's tool system enables agents and language models to execute structured actions by converting Python functions and Runnables into schema-aware components. Tools form the core execution mechanism for agentic workflows, providing automatic argument validation, error handling, and integration with callback systems.

The tool ecosystem consists of three layers:

1. **BaseTool**: Core abstract interface defining tool protocol and execution semantics
2. **Tool Types**: Concrete implementations (StructuredTool, Tool) for different input patterns
3. **Tool Creation**: Decorators and factories (@tool, convert_runnable_to_tool) that generate tools from functions and runnables

## BaseTool Protocol and Core Responsibilities

BaseTool is the abstract base class extending RunnableSerializable that defines the contract for all tools. Every tool carries three essential descriptors and configuration for execution control.

**Required Properties:**
- `name: str` — Unique identifier that clearly communicates purpose; used by agents and models to select tools
- `description: str` — Human-readable text explaining when and why to use the tool; guides model decisions
- `args_schema: TypeBaseModel | dict | None` — Pydantic model or JSON schema dict specifying valid input arguments

**Execution Control:**
- `return_direct: bool` — When True, agent stops looping immediately after tool execution (terminal action)
- `response_format: "content" | "content_and_artifact"` — If "content_and_artifact", tool must return a two-tuple (content, artifact) for structured output with optional artifacts
- `handle_tool_error: bool | str | Callable` — Strategy for ToolException: False (re-raise), True (use exception message), str (fixed message), or callable (custom handler)
- `handle_validation_error: bool | str | Callable` — Strategy for pydantic ValidationError during input parsing

**Callbacks & Metadata:**
- `callbacks: Callbacks` — Lifecycle callbacks (on_tool_start, on_tool_end, on_tool_error) for tracing and monitoring
- `tags: list[str]` — Optional semantic labels attached to all invocations for filtering and metrics
- `metadata: dict` — Custom application-specific metadata passed to callbacks
- `verbose: bool` — Whether to log tool progress

**Provider Integration:**
- `extras: dict[str, Any]` — Provider-specific configuration (e.g., Anthropic cache_control, defer_loading) passed to chat models during tool rendering

## Input Schema Generation and Validation

Tool input validation is built on Pydantic models generated from function signatures. The schema generation pipeline handles both automatic inference and explicit specification.

**Schema Sources (by precedence):**
1. Explicit `args_schema` parameter provided to tool decorator or factory
2. JSON schema dict if `args_schema` is already a dict
3. Inferred from function signature via `create_schema_from_function()`

**Inference Process:**

When `infer_schema=True` (default), the tool examines the function signature to generate a Pydantic model:

- Type hints are extracted via `get_type_hints()` with support for `Annotated` types
- Function docstring is parsed (if `parse_docstring=True`) following Google style to extract parameter descriptions
- Descriptions are merged from: Annotated metadata → docstring Args section → none
- Injected arguments (those annotated with `InjectedToolArg`, `InjectedToolCallId`, or `ToolRuntime`) are automatically excluded from the schema sent to models but re-injected at runtime
- Reserved parameter names (`run_manager`, `callbacks`, `config`) are filtered from the user-facing schema

**Memoization:** The `tool_call_schema` property builds and caches a subset model class per tool instance, excluding injected arguments. The schema class's `model_json_schema()` method is patched to cache the generated dict, preventing expensive regeneration on every agent loop.

**Input Parsing and Validation:**

During execution, tool input is parsed by `_parse_input()`:
- String input is mapped to the single argument if the schema defines exactly one field
- Dict input is validated via Pydantic, with Annotated descriptions providing field documentation
- Injected arguments are identified by signature inspection and re-injected from tool metadata or invocation context (e.g., `tool_call_id`, `ToolRuntime`)
- Validation errors are caught and handled according to `handle_validation_error` configuration

**Annotation-Driven Descriptions:**

Parameter descriptions can come from Annotated field metadata:

```python
from typing import Annotated
from pydantic import Field
from langchain_core.tools import tool

@tool
def my_function(
    query: Annotated[str, Field(description="The search query")],
    limit: Annotated[int, "Maximum number of results"] = 10
) -> str:
    return f"search: {query}"
```

Both `Field(description=...)` and direct string annotations are supported and merged into the generated schema.

## ToolCall and ToolMessage: Request-Response Protocol

Tools are invoked via ToolCall objects and respond with ToolMessage objects, enabling structured communication in agentic loops.

**ToolCall (from messages/tool.py):**

A ToolCall is a TypedDict representing a model's request to execute a tool:

```python
{
    "name": "search_tool",        # Tool name to invoke
    "args": {"query": "python"},  # Validated arguments as dict
    "id": "call_abc123",          # Unique ID for pairing with response
    "type": "tool_call"           # Discriminator
}
```

Multiple ToolCalls can be streamed and merged via `AIMessageChunk`, with streaming yielding `ToolCallChunk` objects that progressively build the arguments JSON string.

**ToolMessage (from messages/tool.py):**

Returned by tools to communicate results back to the model:

```python
ToolMessage(
    content="Result of the tool execution",
    tool_call_id="call_abc123",  # Must match ToolCall.id
    name="search_tool",           # Tool name (optional)
    artifact={"raw": "data"},     # Unshown to model (optional)
    status="success"              # "success" or "error"
)
```

- `artifact`: Stores full tool output when only a summary is sent to the model
- `status`: Allows tools to report errors without raising exceptions (e.g., when `handle_tool_error=True`)
- Content supports rich formatting: plain text or list of message content blocks (images, JSON, search results, documents, etc.)

**ToolOutputMixin:** An empty mixin class used to identify custom objects that tools can return directly without coercion to string. Tools can return ToolOutputMixin instances or lists of them, bypassing automatic ToolMessage wrapping.

## Execution: run() and arun() Methods

Both synchronous and asynchronous execution follow the same lifecycle:

1. **Configuration:** Merge callbacks from tool config, invocation args, and runnable config
2. **Parsing:** Convert tool input (str/dict/ToolCall) to function args/kwargs via `_parse_input()` and `_to_args_and_kwargs()`
3. **Injection:** Inject runtime values (run_manager, callbacks, RunnableConfig) if function signature declares them
4. **Execution:** Call `_run()` or `_arun()` within callback context, propagating config through context variables
5. **Formatting:** Convert output to ToolMessage if invoked with `tool_call_id`, preserving status and artifacts
6. **Error Handling:** Catch ToolException and ValidationError, apply handler strategy (re-raise, return message, or invoke custom handler)

**Callback Lifecycle:**

- `on_tool_start()`: Fired before execution with filtered inputs (injected args removed), tool metadata, and trace ID
- `on_tool_end()`: Fired after success with formatted output
- `on_tool_error()`: Fired on exception with the exception and trace ID

**Config Propagation:**

RunnableConfig passed to invoke/ainvoke is patched with child callbacks and injected into tool execution context, enabling nested tools and state/store access via `ToolRuntime` parameters.

## Tool Types: Tool and StructuredTool

LangChain provides two concrete tool implementations with different input handling semantics.

**Tool (simple.py):**
- Single-input tool expecting string or dict coercion to string
- No explicit args schema required; defaults to `{"tool_input": {"type": "string"}}`
- Used for simple function wrappers and legacy compatibility
- Validates that exactly one argument is passed after schema parsing

**StructuredTool (structured.py):**
- Multi-argument tool with explicit schema-driven parsing
- Each function parameter becomes a separate schema field (unless injected)
- Supports both `func` (sync) and `coroutine` (async)
- Falls back to executor for sync invocation if no coroutine defined
- Preferred pattern for agent tools with multiple named parameters

Both inherit from BaseTool and override `_run()` and `_arun()` to delegate to the wrapped function while preserving config and callbacks.

## Tool Creation: @tool Decorator and Factories

The `@tool` decorator provides the primary user-facing API for converting functions into tools, with overloads supporting multiple usage patterns.

**Decorator Forms:**

```python
# Form 1: No arguments (name from function)
@tool
def search(query: str) -> str:
    """Search the API."""
    return f"Results for {query}"

# Form 2: With parameters
@tool(description="Custom description", return_direct=True)
def calculate(expression: str) -> str:
    return str(eval(expression))

# Form 3: With explicit name
@tool("my_search")
def search(query: str) -> str:
    return query

# Form 4: With Runnable
tool_obj = tool("math_tool", my_runnable, description="...")
```

**Key Behaviors:**

- Default name is `function.__name__` unless overridden
- Description precedence: explicit param → function docstring → args_schema description
- `parse_docstring=True` extracts Google-style Args sections for parameter descriptions (with validation that documented args match signature)
- `infer_schema=True` (default) automatically generates schema from type hints
- `infer_schema=False` requires explicit description and creates Tool (string-input) instead of StructuredTool
- `response_format="content_and_artifact"` expects function to return `(content, artifact)` tuple

**Runnable Conversion:**

When decorating a Runnable, the tool automatically:
- Wraps sync/async invoke methods to inject callbacks
- Uses Runnable's input_schema as the tool's args_schema
- Generates description from input schema if not provided
- Delegates to StructuredTool.from_function() for multi-argument runnables or Tool for string schemas

**Async Support:**

The decorator detects coroutines and creates StructuredTool with both `func` and `coroutine` set, enabling true async execution. Mixed sync/async patterns work via the executor fallback.

## Schema Rendering for Models

Tools are rendered for language models via utility functions in `render.py`:

- `render_text_description(tools: list[BaseTool]) -> str` — Returns `"name - description\n..."` format for prompts
- `render_text_description_and_args(tools: list[BaseTool]) -> str` — Includes args: `"name - description, args: {...}"`

Models receive tool schemas in provider-specific formats (OpenAI function_calling, Anthropic tool_use, etc.), generated by `function_calling.py` utilities that convert tool_call_schema to FunctionDescription dicts with JSON schema parameters.

The `tool_call_schema` property ensures models never see injected arguments or reserved parameter names, protecting tool implementation details.

## Advanced Patterns: Injected Arguments and ToolRuntime

Tools can receive runtime values not controlled by the model via injected arguments.

**InjectedToolArg:** A marker class for parameters that should be injected at runtime:

```python
from typing import Annotated
from langchain_core.tools import tool, InjectedToolArg

@tool
def my_tool(
    user_query: str,
    context_var: Annotated[str, InjectedToolArg]
) -> str:
    # context_var is injected; user only provides user_query
    return f"{user_query} in {context_var}"
```

**InjectedToolCallId:** Specialized marker to inject the tool_call_id:

```python
@tool
def track_call(query: str, call_id: InjectedToolCallId) -> str:
    # call_id automatically populated with tool_call_id from invocation
    return f"Call {call_id}: {query}"
```

**ToolRuntime:** A directly-injected argument type providing access to state, context, and store:

```python
from langchain_core.tools import tool, ToolRuntime

@tool
def stateful_tool(query: str, runtime: ToolRuntime) -> str:
    # Access application state, context, and LangGraph store
    state = runtime.state
    context = runtime.context
    store = runtime.store
    return f"State: {state}, Context: {context}"
```

Injected arguments are:
- Excluded from tool_call_schema sent to models
- Identified via signature inspection in `_get_injected_args_keys_from_signature()`
- Re-injected during `_parse_input()` from tool metadata or invocation context
- Filtered from callback inputs via `_filter_injected_args()`

## Error Handling Strategies

Tools support flexible error handling to allow graceful recovery in agentic loops.

**ToolException:** Custom exception for controlled tool errors:

```python
from langchain_core.tools import tool, ToolException

@tool
def validate_input(value: str) -> str:
    if not value:
        raise ToolException("Value cannot be empty")
    return f"Valid: {value}"
```

**Validation Errors:** Pydantic validation failures are caught and handled per `handle_validation_error`:

```python
@tool(handle_validation_error="Invalid input format")
def my_tool(count: int) -> str:
    return f"Count: {count}"

# If user passes non-integer, returns "Invalid input format" instead of raising
```

**Tool Errors:** ToolException handling per `handle_tool_error`:

```python
@tool(handle_tool_error=True)  # Use exception message
def risky_operation() -> str:
    raise ToolException("Operation failed")
    
# Returns ToolMessage with status="error", content="Operation failed"
```

Custom handlers receive the exception and return str or list of message content blocks:

```python
def my_error_handler(e: ToolException) -> str:
    logger.error(f"Tool failed: {e}")
    return "Operation failed. Please try again later."

@tool(handle_tool_error=my_error_handler)
def operation() -> str:
    raise ToolException("Internal error")
```

Handled errors return ToolMessage with `status="error"` when invoked with `tool_call_id`, allowing agents to observe and respond to failures without breaking the loop.

## BaseToolkit: Organizing Related Tools

For complex systems, tools are organized into toolkits via `BaseToolkit`:

```python
from langchain_core.tools import BaseToolkit, tool

class MathToolkit(BaseToolkit):
    """Toolkit for mathematical operations."""
    
    @property
    def description(self) -> str:
        return "Tools for arithmetic and algebra"
    
    def get_tools(self) -> list[BaseTool]:
        @tool
        def add(a: int, b: int) -> int:
            return a + b
        
        @tool
        def multiply(a: int, b: int) -> int:
            return a * b
        
        return [add, multiply]

toolkit = MathToolkit()
tools = toolkit.get_tools()  # Retrieve all related tools
```

Toolkits enable:
- Logical grouping of related functionality
- Conditional tool availability (return subset based on runtime state)
- Dynamic tool generation
- Integration with agent initialization pipelines

## Converting Runnables to Tools

Runnables can be converted to tools via `tool()` decorator or `convert_runnable_to_tool()` function:

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import convert_runnable_to_tool

my_runnable = RunnablePassthrough()

# Via convert function
tool_obj = convert_runnable_to_tool(
    my_runnable,
    name="passthrough",
    description="Passes input through unchanged"
)

# Via decorator
tool_obj = tool("passthrough", my_runnable)
```

The conversion:
- Extracts input_schema from Runnable.get_input_jsonschema()
- Validates schema is object type (required for multi-arg tools)
- Wraps invoke/ainvoke to inject callbacks into config
- Delegates to StructuredTool.from_function() with wrapped functions
- Falls back to Tool for string-input runnables

## Lifecycle and Invariants

**Tool Instance Lifecycle:**

1. **Construction:** Schema memoization cleared on `__setattr__` or `model_copy()` if name/description/args_schema changed
2. **First Schema Access:** tool_call_schema builds subset model, patches class to cache JSON schema
3. **Execution:** Callbacks configured, input parsed, injected args identified, function called, output formatted
4. **Pickling:** Schema memo cleared (dynamic classes cannot pickle by reference); rebuilt on next access

**Schema Caching Invariants:**

- Memoized subset model class never regenerates if name/description/args_schema unchanged
- Pydantic model_json_schema() called on subset class returns cached dict on subsequent calls
- Cache invalidation is explicit via private _TOOL_CALL_SCHEMA_FIELDS check
- Preserves performance under high-frequency agent loops

**Execution Invariants:**

- Callbacks always fire in order: on_tool_start → (on_tool_error | on_tool_end)
- Config context is set during execution, allowing nested tools to access state/store
- ToolMessage wrapping only occurs if tool_call_id provided
- Injected arguments are never visible to the model or in callback inputs
- Status="error" set only when handle_tool_error converts exception to message

## Extension Points

**Subclassing BaseTool:**

Custom tool implementations override:
- `_run(self, *args, **kwargs) -> Any` — Sync execution logic
- `_arun(self, *args, **kwargs) -> Any` — Async execution logic (default delegates to _run via executor)
- `get_input_schema()` — Override schema source (default uses args_schema or creates from _run signature)

**Custom Error Handlers:**

Passed as callables to `handle_tool_error` and `handle_validation_error`:

```python
def custom_validation_handler(e: ValidationError) -> str:
    # Extract user-friendly message from Pydantic error
    return ", ".join(f"{err['loc'][0]}: {err['msg']}" for err in e.errors())

@tool(handle_validation_error=custom_validation_handler)
def my_tool(count: int) -> str:
    return str(count)
```

**Callback Managers:**

Tools inject CallbackManager/AsyncCallbackManager to enable:
- Custom event handlers (logging, metrics, tracing)
- Nested tool execution with callback propagation
- on_tool_start/on_tool_end hooks for observability

Tools expose run_manager in `_run()` signature to allow direct callback invocation.

## Configuration and Operational Concerns

**Reserved Parameter Names:**

Parameters named `config`, `run_manager`, or `callbacks` are filtered from the tool schema because they conflict with LangChain's runtime injection. Use `ToolRuntime` annotation to access runtime state instead.

**Provider Extras:**

The `extras` dict allows passing provider-specific configuration:

```python
@tool(extras={"cache_control": {"type": "ephemeral"}})
def cached_operation(query: str) -> str:
    return query
```

Chat models inspect extras and apply provider-specific behavior when rendering tools.

**Docstring Parsing:**

When `parse_docstring=True`, Google-style docstrings are parsed for parameter descriptions:

```python
@tool(parse_docstring=True)
def process(name: str, count: int) -> str:
    """Process items by name.
    
    Args:
        name: The item name
        count: Number of items to process
    """
    return f"{name}: {count}"
```

Invalid docstrings (missing Args section, args not in signature, malformed) raise ValueError if `error_on_invalid_docstring=True`.

**Verbose Output:**

Set `verbose=True` to log tool execution. Combined with callback managers for comprehensive observability.

## Summary: When to Use Each Pattern

- **@tool decorator:** Primary pattern for converting functions to tools; use with type hints for automatic schema inference
- **StructuredTool.from_function():** Direct factory when decorator syntax isn't convenient or for programmatic tool creation
- **Tool (simple):** Legacy compatibility or single-string-input tools
- **BaseToolkit:** Organizing related tools or dynamic tool generation
- **convert_runnable_to_tool():** Wrapping existing Runnables as tools with consistent invocation
- **Injected arguments:** Share runtime context (state, store, call IDs) without model visibility
- **Custom error handlers:** Transform Pydantic or tool errors into user-friendly messages for agents
