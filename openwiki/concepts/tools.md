---
type: "Reference"
title: "Bare decorator - infers schema from function signature"
openwiki_generated: true
verified:
  - by: openwiki/0.5.0
    at: 2026-10-25T08:23:03.589Z
sources:
  - id: openwiki-source-c52037e7b642f7ac5a7642a8
    resource: repo://libs/core/langchain_core/language_models/chat_models.py
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
generated: { by: "openwiki/0.5.0", at: "2026-10-25T08:23:03.589Z" }
---


## Overview

Tools extend the capabilities of language models by enabling them to interact with external systems and perform specialized tasks. In LangChain, tools are components that agents can select and invoke to accomplish user objectives. Tools provide structured interfaces with input schemas, descriptions for model guidance, and execution logic that may be sync or async.

A tool workflow involves three key phases:

1. **Schema Definition & Model Binding**: Tools register their input requirements via JSON schemas so models can generate well-formed invocations.
2. **Tool Calling**: The language model generates structured tool calls based on the user query and tool descriptions.
3. **Execution & Result Formatting**: The tool execution framework validates inputs, runs the tool logic, handles errors, and returns results wrapped in `ToolMessage` to continue the agentic loop.

## Tool Types and Definition

### BaseTool: The Foundation

All tools in LangChain extend `BaseTool`, a `RunnableSerializable` that accepts string, dict, or `ToolCall` inputs and produces output. Tools implement the `_run` abstract method (sync) and optionally `_arun` (async).

Core properties:
- **name**: Unique identifier for the tool, used by models and in tool call references.
- **description**: Describes the tool's purpose, parameters, and usage; guides model selection and parameter generation.
- **args_schema**: A Pydantic `BaseModel` or JSON schema dict defining input argument validation and structure. When omitted, inferred from function signatures.
- **return_direct**: Whether agent loops terminate after this tool executes (typically for final answer tools).

Validation and error handling:
- **handle_tool_error**: Controls how `ToolException` is handled: `False` (re-raise), `True` (return exception message), string (return custom message), or callable (custom handler).
- **handle_validation_error**: Controls handling of input validation errors: `False` (re-raise), `True` (return "Tool input validation error"), string (custom message), or callable (custom handler).

### The Tool Decorator

The `@tool` decorator is the most concise way to create tools from Python functions. It supports multiple usage patterns:

```python
# Bare decorator - infers schema from function signature
@tool
def search(query: str) -> str:
    """Search for the query."""
    return f"Results for {query}"

# With custom name
@tool("search_documents")
def my_search(q: str) -> str:
    """Search documents."""
    return q

# With configuration
@tool(description="Advanced search", return_direct=True)
def advanced_search(q: str, limit: int = 10) -> str:
    """Search with advanced options."""
    return f"Searched {q} with limit {limit}"

# With explicit schema
@tool(args_schema=MySchema)
def configured_tool(x: int) -> str:
    """Tool with explicit schema."""
    return str(x)
```

The decorator creates either a `StructuredTool` (when input schema inference is enabled) or a simple `Tool` (for single string inputs when `infer_schema=False`).

### StructuredTool

`StructuredTool` wraps a function or coroutine and enforces a Pydantic schema. It accepts multiple named arguments, validates them against the schema, and passes them to the wrapped function.

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

class SearchSchema(BaseModel):
    query: str
    limit: int = 10

def search_fn(query: str, limit: int) -> str:
    return f"Searched {query} with limit {limit}"

tool = StructuredTool.from_function(
    search_fn,
    name="search",
    description="Search for documents",
    args_schema=SearchSchema
)
```

### Simple Tool

`Tool` is a lighter-weight variant that accepts single string inputs and returns string outputs. Used when `infer_schema=False` or for backwards compatibility.

### Class-Based Tools

Custom tools inherit directly from `BaseTool`:

```python
from langchain_core.tools import BaseTool

class CustomTool(BaseTool):
    name: str = "custom_tool"
    description: str = "Does something custom"
    args_schema: type[BaseModel] = MySchema
    
    def _run(self, **kwargs) -> str:
        # Sync implementation
        return "result"
    
    async def _arun(self, **kwargs) -> str:
        # Async implementation
        return "result"
```

## Tool Schemas

### Schema Inference

When a tool is created with `infer_schema=True` (the default), the schema is automatically derived from the function signature and type annotations:

```python
@tool
def process(value: int, flag: bool = False) -> str:
    """Process a value with optional flag."""
    return f"Processed {value}"
```

The decorator generates a Pydantic model with `value` (required) and `flag` (optional, defaults to False). The tool's JSON schema becomes the model's `model_json_schema()`.

### Manual Schema Definition

Pass an explicit Pydantic model as `args_schema`:

```python
from pydantic import BaseModel, Field

class InputSchema(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Result limit")

@tool(args_schema=InputSchema)
def search(query: str, limit: int) -> str:
    return f"Searched {query} with limit {limit}"
```

### JSON Schema Format

For models that don't use Pydantic, pass a JSON schema dict:

```python
tool = BaseTool(
    name="my_tool",
    description="Tool description",
    args_schema={
        "type": "object",
        "properties": {
            "param": {"type": "string"}
        },
        "required": ["param"]
    }
)
```

### Injected Arguments

Arguments can be marked for runtime injection rather than user input. These arguments are excluded from the tool schema sent to models.

**InjectedToolArg** marks arguments to be supplied by the framework:

```python
from typing import Annotated
from langchain_core.tools import tool, InjectedToolArg

@tool
def my_tool(
    query: str,
    injected_value: Annotated[str, InjectedToolArg]
) -> str:
    """Tool with injected argument."""
    return f"{query} + {injected_value}"
```

**InjectedToolCallId** provides the tool call ID:

```python
from langchain_core.tools import tool, InjectedToolCallId

@tool
def my_tool(
    x: int,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> ToolMessage:
    """Tool that receives its call ID."""
    return ToolMessage(str(x), tool_call_id=tool_call_id, name="my_tool")
```

## Tool Calling Workflow

### Schema Generation and Model Binding

When tools are bound to a chat model via `bind_tools()`, the model receives each tool's JSON schema:

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4")
tools = [search_tool, calculator_tool]
bound_model = model.bind_tools(tools)
```

The model then uses the schemas to guide tool selection and parameter generation. Models typically format tool calls as function calls or structured outputs in their response.

### Tool Call Structure

A tool call is a structured object containing:
- **name**: The tool name (matches tool instance name).
- **args**: A dict of arguments (or a single string for simple tools).
- **tool_call_id**: A unique ID generated by the model (typically a UUID or provider-specific string).

```python
tool_call = {
    "name": "search",
    "args": {"query": "LangChain documentation"},
    "tool_call_id": "call_abc123"
}
```

### Input Validation and Parsing

When a tool is invoked, the `_parse_input` method validates input against the tool's args schema:

```python
tool_input, kwargs = tool._prep_run_args(
    {"query": "test", "limit": 5},
    config=config
)
```

The framework:
1. Validates the input dict against the Pydantic schema if present.
2. Applies Pydantic defaults for fields with explicit defaults.
3. Injects runtime-supplied arguments (e.g., `run_manager`, `tool_call_id`).
4. Filters out framework arguments (`run_manager`, `callbacks`) from the user-supplied input before passing to the wrapped function.

### Execution and Error Handling

The `run()` method (or async `arun()`) executes the tool with callback and config context:

1. **Validation**: Input is validated against the schema. Validation errors are handled per `handle_validation_error`.
2. **Execution**: The `_run` method is invoked with validated arguments, optionally receiving `run_manager` and `config` if the signature accepts them.
3. **Error Handling**: `ToolException` is handled per `handle_tool_error`. Other exceptions are re-raised unless handled.
4. **Response Formatting**: Tool output is wrapped in a `ToolMessage` (if a tool_call_id was provided) or returned directly.

```python
try:
    result = tool.run(
        {"query": "test"},
        tool_call_id="call_123"
    )
    # result is a ToolMessage
except ToolException as e:
    # Handled per handle_tool_error
    pass
```

### Tool Response Formats

**content** (default): Tool output becomes the `ToolMessage` content.

**content_and_artifact**: Tool output is a `(content, artifact)` tuple; both are stored in the `ToolMessage`.

```python
@tool(response_format="content_and_artifact")
def structured_output() -> tuple[str, dict]:
    """Returns content and artifact."""
    return "summary", {"detailed": "data"}
```

## Rendering and Tool Description

Tools can be rendered for display and model context:

```python
from langchain_core.tools import render_text_description, render_text_description_and_args

# Render just names and descriptions
text = render_text_description([search_tool, calc_tool])
# Output:
# search - Search for information
# calculator - Perform math calculations

# Render with argument schemas
detailed = render_text_description_and_args([search_tool])
# Output:
# search - Search for information, args: {"query": {"type": "string"}}
```

## Advanced Patterns

### Error Handling with Custom Handlers

Define custom logic for handling tool errors:

```python
def handle_search_error(e: ToolException) -> str:
    if "timeout" in str(e).lower():
        return "Search timed out. Please try a simpler query."
    return f"Search failed: {e}"

@tool(handle_tool_error=handle_search_error)
def search(query: str) -> str:
    if len(query) > 100:
        raise ToolException("Query too long")
    return f"Found: {query}"
```

### Async Tools

Coroutines are converted to async tools:

```python
@tool
async def async_search(query: str) -> str:
    """Async search implementation."""
    import asyncio
    await asyncio.sleep(0.1)
    return f"Async results for {query}"
```

### Tool Selection and Composition

Agents select tools based on descriptions and execute them in sequence:

```python
tools = [search_tool, calculator_tool, weather_tool]
agent = initialize_agent(tools, model, agent="openai-functions")
result = agent.run("What is the square root of 144 in Celsius?")
```

Each tool call produces a `ToolMessage` that becomes input context for the next model invocation.

### Tool Metadata and Tags

Tools support metadata for tracking and instrumentation:

```python
@tool(tags=["search", "documentation"])
def search(query: str) -> str:
    """Search documentation."""
    return query

tool.metadata = {
    "version": "1.0",
    "category": "search",
    "provider": "langchain"
}
```

### Provider-Specific Extras

Pass provider-specific configuration via `extras`:

```python
@tool(extras={"cache_control": {"type": "ephemeral"}})
def cached_search(query: str) -> str:
    """Search with caching."""
    return query
```

## Tool Binding to Models

Chat models implement `bind_tools()` to attach tools and configure tool calling behavior:

```python
model = ChatOpenAI(model="gpt-4")

# Bind tools
bound_model = model.bind_tools(
    [search_tool, calculator_tool],
    tool_choice="any"  # Let model choose any tool
)

# Invoke - model response contains ToolCall objects
response = bound_model.invoke("What is 2+2?")

# response.tool_calls contains structured tool calls
for tool_call in response.tool_calls:
    result = invoke_tool(tool_call)  # Execute tool
    # Continue agent loop with result...
```

## Lifecycle and State

### Tool Call Schema Memoization

For performance, the `tool_call_schema` (the schema sent to models) is memoized per tool instance and cached after first access. This avoids regenerating Pydantic subset models and JSON schemas on every agent invocation.

The memo is cleared when `name`, `description`, or `args_schema` is reassigned:

```python
tool = search_tool
schema1 = tool.tool_call_schema  # Generated and cached

tool.description = "Updated description"
# Memo cleared; next access regenerates

schema2 = tool.tool_call_schema  # New schema generated and cached
```

### Pickling

Tools can be pickled for serialization, but memoized schemas (which are dynamically created classes) are dropped during pickling and rebuilt on next access.

## Key Responsibilities and Design

- **Schema Management**: Tools maintain input schema as Pydantic models or dicts, infer from function signatures when needed, and expose the schema for model binding.
- **Input Validation**: Validate user-supplied arguments against the schema, handle validation errors per configuration, and filter framework arguments.
- **Execution**: Run the tool logic (sync or async), catch exceptions, apply error handlers, and format output.
- **Callback Integration**: Emit lifecycle events (`on_tool_start`, `on_tool_end`, `on_tool_error`) for observability and debugging.
- **Runnable Protocol**: Implement `invoke` and `ainvoke` for pipeline composition and streaming support.
- **Model Integration**: Expose schemas in provider formats (OpenAI, Anthropic, etc.) for model binding and structured tool calling.
