---
type: "API Reference"
title: "BaseTool API and Schema Generation"
description: "Reference documentation for BaseTool class, schema generation, input validation, and execution lifecycle in LangChain's tool system."
tags: ["tools", "api", "schema", "validation", "BaseTool"]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-13T08:16:35.902Z
sources:
  - id: openwiki-source-4ff475d7b00540f962384251
    resource: repo://libs/core/langchain_core/tools/base.py
generated: { by: "openwiki/0.5.0", at: "2026-11-13T08:16:35.902Z" }
---

## BaseTool Class

`BaseTool` is the abstract base class for all LangChain tools, extending `RunnableSerializable[str | dict[str, Any] | ToolCall, Any]`. It defines the core protocol and lifecycle for tool execution.

**Location:** `repo://libs/core/langchain_core/tools/base.py`

### Essential Properties

Every tool must define these three descriptive properties:

#### `name: str` (required)
The unique identifier for the tool that clearly communicates its purpose. Used by agents and models to select and invoke tools.

#### `description: str` (required)
Human-readable text explaining when, why, and how to use the tool. Guides model decisions in agentic workflows. The description is included in tool schemas sent to language models and supports few-shot examples.

#### `args_schema: ArgsSchema | None = None` (optional)
Specifies valid input arguments via:
- A Pydantic `BaseModel` subclass (v1 or v2)
- A JSON schema dict
- `None` to auto-infer from function signature

When `args_schema` is `None`, the tool's `get_input_schema()` method automatically generates a Pydantic model from the `_run()` method signature.

### Execution Control

#### `return_direct: bool = False`
When `True`, the agent executor stops looping after the tool returns. Used for terminal actions that directly produce final answers.

#### `response_format: Literal["content", "content_and_artifact"] = "content"`
Controls output format:
- `"content"`: Tool output is the message content (standard)
- `"content_and_artifact"`: Tool must return a two-tuple `(content, artifact)` where content goes to the model and artifact is stored separately

#### `handle_tool_error: bool | str | Callable[[ToolException], ToolExceptionHandlerOutput] | None = False`
Strategy for handling `ToolException` raised during execution:
- `False`: Re-raise the exception (default)
- `True`: Return the exception message as tool output
- `str`: Return the string as tool output
- `Callable`: Invoke custom handler with exception; returns string or list of message content blocks

When `tool_call_id` is provided, handled errors are wrapped in `ToolMessage` with `status="error"`.

#### `handle_validation_error: bool | str | Callable[[ValidationError], str] | None = False`
Strategy for handling Pydantic `ValidationError` during input parsing:
- `False`: Re-raise (default)
- `True`: Return "Tool input validation error"
- `str`: Return the string
- `Callable`: Invoke handler with error object

### Callbacks and Metadata

#### `callbacks: Callbacks = None`
Lifecycle callbacks attached to the tool. Fired during execution:
- `on_tool_start(tool_info, input_str, ...)` — Before execution
- `on_tool_end(output, ...)` — After success
- `on_tool_error(exception, ...)` — On exception

#### `tags: list[str] | None = None`
Optional semantic labels attached to every invocation. Passed to callback handlers for filtering and metrics. Useful for identifying tool instances by use case.

#### `metadata: dict[str, Any] | None = None`
Custom application-specific metadata passed to callbacks. Used to associate contextual information (e.g., user ID, request ID) with tool execution.

#### `verbose: bool = False`
Enable tool progress logging. May be overridden per invocation.

### Provider Integration

#### `extras: dict[str, Any] | None = None`
Provider-specific configuration passed through to language models during tool binding:
- **Anthropic**: `cache_control`, `defer_loading`, `input_examples`
- **OpenAI**: Additional parameters as needed
- Custom implementations may define their own extras

## Input Schema Management

### Schema Generation and Inference

When `args_schema` is `None`, the `get_input_schema(config)` method returns a Pydantic model automatically generated from the `_run()` method signature via `create_schema_from_function()`.

The schema generation process:
1. Extracts type hints from the function signature, including support for `Annotated` types
2. Parses function docstring (if `parse_docstring=True`) following Google style for parameter descriptions
3. Merges descriptions from `Annotated` metadata, docstrings, and field defaults
4. Filters out reserved parameter names (`run_manager`, `callbacks`) and injected arguments
5. Returns a Pydantic model class with field validation

**Injected Arguments** are automatically excluded from the schema sent to models but re-injected at runtime. These include:
- `InjectedToolArg`-annotated parameters
- `InjectedToolCallId` for receiving the tool call ID
- `ToolRuntime` parameters for accessing state/context/store
- Parameters named `run_manager`, `callbacks`, or matching `RunnableConfig` type

### Schema Memoization

The `tool_call_schema` property builds and caches a subset model class per tool instance. This subset model:
- Excludes injected arguments
- Is memoized in `_tool_call_schema_memo` (cleared when `name`, `description`, or `args_schema` is reassigned)
- Has its `model_json_schema()` method patched to cache the generated dict

This optimization prevents expensive schema regeneration on every agent loop, especially important when multiple tools are bound to a model.

### Input Parsing and Validation

The `_parse_input(tool_input: str | dict, tool_call_id: str | None)` method:
1. **String input**: Maps to the single argument if `args_schema` defines exactly one field
2. **Dict input**: Validates via Pydantic model validation
3. **Injected arguments**: Identified by signature inspection and re-injected from metadata or context
4. **Validation errors**: Caught and handled per `handle_validation_error` configuration

For Pydantic model validation:
- v2 uses `model_validate()`; v1 uses `parse_obj()`
- Provided fields, alias-based fields, and fields with explicit defaults are included in the validated input
- Synthetic fields (`args`, `kwargs`) are excluded unless they capture `*args`/`**kwargs`

### Annotation-Driven Descriptions

Parameter descriptions are extracted from `Annotated` metadata in precedence order:

```python
from typing import Annotated
from pydantic import Field
from langchain_core.tools import tool

@tool
def search(
    query: Annotated[str, Field(description="The search query")],
    limit: Annotated[int, "Maximum results"] = 10
) -> str:
    """Search the knowledge base."""
    return f"Results for {query}"
```

Both `Field(description=...)` and direct string annotations are supported.

## Execution Lifecycle

### invoke() and ainvoke()

Both methods implement the `Runnable` interface:

```python
def invoke(input: str | dict[str, Any] | ToolCall, config: RunnableConfig | None = None) -> Any
async def ainvoke(input: str | dict[str, Any] | ToolCall, config: RunnableConfig | None = None) -> Any
```

These methods prepare arguments via `_prep_run_args()` and delegate to `run()` or `arun()`.

### run() and arun() Lifecycle

Both follow the same synchronous and asynchronous flow:

1. **Configuration**: Merge callbacks from tool config, invocation args, and runnable config via `CallbackManager.configure()` / `AsyncCallbackManager.configure()`

2. **Callback Start**: Call `callback_manager.on_tool_start()` with:
   - Tool metadata (name, description)
   - Filtered input (injected args removed)
   - Trace ID and other metadata

3. **Input Preparation**: Convert tool input (str/dict/ToolCall) to function args/kwargs:
   - Extract `tool_call_id` from ToolCall if present
   - Call `_parse_input()` to validate dict arguments
   - Call `_to_args_and_kwargs()` to split into positional and keyword args
   - Filter injected args from input before passing to callback handlers

4. **Runtime Injection**: Inject runtime values if function signature expects them:
   - `run_manager: CallbackManagerForToolRun` (if parameter declared)
   - `RunnableConfig` parameter (if parameter declared and matches type)
   - Injected arguments from `_injected_args_keys`

5. **Execution**: Call `_run(*args, **kwargs)` or `_arun(*args, **kwargs)` within config context:
   - Config is propagated via context variables
   - Nested tools can access state/store via `ToolRuntime`

6. **Output Processing**:
   - If `response_format="content_and_artifact"`, expect tuple `(content, artifact)`
   - Validate response type; raise ValueError if mismatch
   - Format output via `_format_output()` which creates `ToolMessage` if `tool_call_id` is set

7. **Error Handling**:
   - `ValidationError | ValidationErrorV1`: Handled per `handle_validation_error`
   - `ToolException`: Handled per `handle_tool_error`
   - Other exceptions: Re-raised after `on_tool_error()` callback

8. **Callback End**: Call `run_manager.on_tool_end(output)` or `on_tool_error(exception)`

### Output Formatting

The `_format_output(content, artifact, tool_call_id, name, status)` method:
- Returns `content` unchanged if it's a `ToolOutputMixin` instance or list of `ToolOutputMixin` instances
- Returns `content` unchanged if `tool_call_id` is `None` (not part of agentic flow)
- Otherwise, wraps in `ToolMessage(content, tool_call_id=tool_call_id, artifact=artifact, name=name, status=status)`

This enables tools to return either simple strings, structured objects, or rich multi-block content.

## Tool Input Types

### String Input
Simple single-argument tools accept string input directly, mapping to the schema's sole field.

### Dict Input
Multi-argument tools accept dict input where keys match schema field names. Pydantic validates and coerces types.

### ToolCall Input
Tools invoked by agents receive `ToolCall` objects (dict-based TypedDict):
```python
{
    "name": "tool_name",
    "args": {"param": "value"},
    "id": "call_abc123",
    "type": "tool_call"
}
```

The tool extracts `tool_call_id` and uses `args` as the dict input.

## Validation and Error Handling

### Pydantic Version Compatibility

BaseTool supports both Pydantic v1 and v2:
- v1 models use `validate_arguments_v1()` and `parse_obj()`
- v2 models use `validate_arguments()` and `model_validate()`
- Mixed v1/v2 annotations in a single function raise `NotImplementedError`

### Schema Annotation Validation

During class definition, `__init_subclass__` validates that `args_schema` is properly annotated:
- Bare `BaseModel` type annotation (not a subclass) raises `SchemaAnnotationError`
- Runtime `__init__` validates `args_schema` is a Pydantic BaseModel subclass or dict

### Message Content Format

Tools can return rich message content via list of content blocks:
```python
[
    {"type": "text", "text": "Summary"},
    {"type": "json", "data": {...}},
    {"type": "image_url", "image_url": {...}},
    {"type": "document", "document": {...}}
]
```

Valid block types: `text`, `image_url`, `image`, `json`, `search_result`, `custom_tool_call_output`, `document`, `file`.

## Key Methods and Properties

### `args: dict[str, Any]`
Returns the tool's input argument properties extracted from the schema JSON representation.

### `is_single_input: bool`
Returns `True` if the tool has exactly one input argument (excluding `kwargs`).

### `get_input_schema(config: RunnableConfig | None) -> TypeBaseModel`
Returns the tool's input schema. If `args_schema` is set, returns it; otherwise, generates from `_run()` signature.

### `tool_call_schema: ArgsSchema`
Returns the schema for tool calls excluding injected arguments. Memoized and cached per instance.

### Abstract Methods

#### `_run(*args: Any, **kwargs: Any) -> Any`
The actual tool implementation. Subclasses must implement this synchronous method.

Optional signature supports:
- `run_manager: CallbackManagerForToolRun | None = None` — To enable callback tracing
- `config: RunnableConfig` — For nested tool access to state/store

#### `async _arun(*args: Any, **kwargs: Any) -> Any`
Asynchronous implementation. Default implementation uses executor to call `_run()` if not overridden.

### `_filter_injected_args(tool_input: dict[str, Any]) -> dict[str, Any]`
Removes injected and reserved arguments from input before passing to callbacks. Called internally before callback start.

### `_to_args_and_kwargs(tool_input: str | dict, tool_call_id: str | None) -> tuple[tuple[str, ...], dict[str, Any]]`
Converts parsed tool input to positional and keyword arguments for calling the implementation function.

### Serialization Methods

#### `__setattr__(name: str, value: Any)`
Overridden to clear the `tool_call_schema_memo` when `name`, `description`, or `args_schema` is reassigned.

#### `model_copy(*, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self`
Creates a copy with optional updates. Clears schema memo if update affects schema-building fields.

#### `__getstate__() -> dict`
Drops the tool_call_schema_memo when pickling (dynamically created classes cannot be pickled by reference).

## Exceptions

### `ToolException`
Raised by tools to signal errors without stopping the agent. Handled according to `handle_tool_error` configuration.

### `SchemaAnnotationError`
Raised during class definition if `args_schema` has incorrect type annotation (e.g., bare `BaseModel` instead of `Type[BaseModel]`).

## Concrete Implementations

LangChain provides two concrete tool classes:

### `Tool` (simple.py)
Single-input tool enforcing exactly one argument after schema parsing. Defaults to `{"tool_input": {"type": "string"}}` schema. Used for simple function wrappers and backward compatibility.

### `StructuredTool` (structured.py)
Multi-argument tool with explicit schema-driven parsing. Each function parameter becomes a schema field (except injected args). Supports both sync (`func`) and async (`coroutine`) invocation. Preferred for agent tools with multiple named parameters.

Both inherit from `BaseTool` and override `_run()` and `_arun()` to delegate to wrapped functions while preserving config and callbacks.
