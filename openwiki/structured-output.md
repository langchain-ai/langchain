---
type: "Reference"
title: "AutoStrategy (recommended)"
openwiki_generated: true
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:18:34.589Z
sources:
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
  - id: openwiki-source-ec30ab6256dd50cc670919f6
    resource: repo://libs/langchain_v1/langchain/agents/structured_output.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:18:34.589Z" }
---


## Overview

Structured output is the mechanism that ensures a language model returns responses matching a specific JSON schema. Rather than receiving unparsed text or tool calls, agents can enforce that model outputs conform to Pydantic models, dataclasses, TypedDicts, or raw JSON schemas. The factory configures one of three strategies—tool-based, provider-native, or automatic—each with different tradeoffs around compatibility, validation, and retry behavior.

### Core Concept

When an agent is created with a `response_format` parameter, it tells the model "all your responses must match this schema." The agent:

1. **Registers** the schema as an artificial tool (for tool-calling strategy) or sends it to the model provider's native API (for provider-native strategy)
2. **Detects** whether the model supports native structured output (AutoStrategy)
3. **Parses** the model's response against the schema using Pydantic's `TypeAdapter` for validation
4. **Retries** automatically on validation errors (if enabled via `handle_errors`)
5. **Stores** the parsed result in `structured_response` state field

This is distinct from tool calling—structured output constrains the response itself, not the model's tool invocations.

## Response Format Strategies

Three strategies control how structured output is enforced:

### ToolStrategy: Tool-Based Structured Output

The model is presented with an artificial tool whose name and arguments match the schema. When the model calls this tool, its arguments are parsed and validated against the schema.

**Lifecycle:**

1. Schema is wrapped as a `StructuredTool` with the schema's JSON schema as `args_schema`
2. Tool is added to the model's tool list with `tool_choice="any"` to force use
3. Model generates a tool call with the schema name
4. Tool call arguments are parsed via `_parse_with_schema` using Pydantic's `TypeAdapter`
5. Parsed result stored in `structured_response`
6. Empty `ToolMessage` returned (tool has no real execution)

**Advantages:**

- Works with all models that support tool calling
- Full validation available for non-dict schemas
- Automatic retry on validation errors (configurable)
- Supports Union types (multiple schema variants)

**Limitations:**

- Requires tool calling capability
- Adds tool to the tool list (may consume tool slot on some models)
- Raw JSON schema dicts skip validation, making `handle_errors` inert

**Error Handling:**

The `handle_errors` parameter controls retry behavior on validation failure:

- `True` (default): Catch all errors, retry with default error template
- `False`: Let exceptions propagate without retry
- `str`: Catch all errors, retry with custom message
- `type[Exception]` or `tuple[type[Exception], ...]`: Only retry specific exception types
- `Callable[[Exception], str]`: Custom function returns retry message per exception

Failed parses generate a `ToolMessage` with the error message, allowing the model to correct its output.

### ProviderStrategy: Native Structured Output

The schema is sent to the model provider's native structured output API (e.g., OpenAI's `response_format` with `type: "json_schema"`). The provider enforces schema compliance on their side; the agent only needs to parse the JSON response.

**Lifecycle:**

1. Schema converted to JSON Schema via Pydantic's `model_json_schema()`
2. Wrapped in provider-specific format: `{"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": ...}}`
3. Passed to model via `model.bind(..., response_format={...})`
4. Model returns JSON text (guaranteed valid by provider)
5. Text parsed via `json.loads()` then validated against schema
6. Parsed result stored in `structured_response`

**Advantages:**

- Provider enforces schema—no invalid JSON possible
- Doesn't consume tool slots
- Works alongside tool calling
- Strict mode available (supported providers only)

**Limitations:**

- Limited to models with native structured output (OpenAI, Claude, etc.)
- No automatic retry on validation errors (provider side is strict)
- Must explicitly test model capability support

**Capability Detection:**

A model supports provider-native structured output if:

1. Its profile includes `"structured_output": True` (checked via `model.profile`), AND
2. Not a pre-3-series Gemini model (which cannot mix tools with structured output), OR
3. Model name matches fallback patterns like `gpt-4o`, `claude-opus`, etc.

### AutoStrategy: Automatic Strategy Selection

Defers strategy selection until model invocation time. The factory inspects the bound model and chooses:

1. **ProviderStrategy** if the model supports native structured output
2. **ToolStrategy** as fallback for all other models

**Lifecycle:**

1. User passes raw schema or `AutoStrategy(schema=...)`
2. Factory converts to `ToolStrategy` upfront to pre-build tools
3. At model call time, `_supports_provider_strategy()` checks model capabilities
4. If supported, `ProviderStrategy` is created and model kwargs bound
5. Otherwise, `ToolStrategy` is used (tools already prepared)

**Advantage:** Best of both worlds—uses provider when available, falls back to tools.

```python
from langchain.agents import create_agent
from pydantic import BaseModel

class WeatherResponse(BaseModel):
    """Weather forecast response."""
    location: str
    temperature: int
    condition: str

# AutoStrategy (recommended)
agent = create_agent(
    model="openai:gpt-4o",
    response_format=WeatherResponse,  # Wrapped in AutoStrategy automatically
)

# Explicit strategies
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy

agent_native = create_agent(
    model="openai:gpt-4o",
    response_format=ProviderStrategy(schema=WeatherResponse),
)

agent_tools = create_agent(
    model="openai:gpt-4o",
    response_format=ToolStrategy(
        schema=WeatherResponse,
        handle_errors=True,  # Retry on validation failure
    ),
)
```

## Schema Types

Supported schema types for structured output:

| Type | Example | Validation | Tool Use |
|------|---------|-----------|----------|
| **Pydantic model** | `class Response(BaseModel): ...` | Full validation | Yes (via TypeAdapter) |
| **Dataclass** | `@dataclass class Response: ...` | Full validation | Yes (via TypeAdapter) |
| **TypedDict** | `class Response(TypedDict): ...` | Full validation | Yes (via TypeAdapter) |
| **JSON Schema dict** | `{"type": "object", "properties": {...}}` | None (dict schemas skip validation) | Returns dict as-is |

The factory normalizes all types via `_SchemaSpec`, which:

- Extracts schema name (class name, `title` field, or generated UUID fragment)
- Extracts description (docstring, `description` field, or empty)
- Computes JSON Schema representation for tool binding
- Tracks `strict` mode flag for provider-side enforcement

## Integration with Agent Factory

The `create_agent()` function integrates structured output through:

### 1. Upfront Schema Registration

```python
# At agent creation time
if tool_strategy_for_setup:
    for response_schema in tool_strategy_for_setup.schema_specs:
        structured_tool_info = OutputToolBinding.from_schema_spec(response_schema)
        structured_output_tools[structured_tool_info.tool.name] = structured_tool_info
```

Pre-builds `OutputToolBinding` instances wrapping schemas as `StructuredTool` instances. These bindings store the original schema, its classification (`pydantic`, `dataclass`, etc.), and the tool for later parsing.

### 2. Model Binding During Invocation

The `_get_bound_model()` function (called on each model invocation) performs auto-detection:

```python
# Determine effective response format (auto-detect if needed)
effective_response_format: ResponseFormat[Any] | None
if isinstance(response_format, AutoStrategy):
    if _supports_provider_strategy(request.model, tools=request.tools):
        effective_response_format = ProviderStrategy(schema=response_format.schema)
    else:
        effective_response_format = ToolStrategy(schema=response_format.schema)
else:
    effective_response_format = response_format
```

Then binds the model:

- **ProviderStrategy**: `model.bind(..., response_format={...})`
- **ToolStrategy**: `model.bind_tools(final_tools, tool_choice="any", ...)`

### 3. Response Parsing

After model invocation, `_handle_model_output()` dispatches to the appropriate parser:

**For ProviderStrategy:**

```python
if isinstance(effective_response_format, ProviderStrategy):
    if not output.tool_calls:
        provider_strategy_binding = ProviderStrategyBinding.from_schema_spec(...)
        structured_response = provider_strategy_binding.parse(output)
        return {"messages": [output], "structured_response": structured_response}
```

**For ToolStrategy:**

```python
if isinstance(effective_response_format, ToolStrategy):
    structured_tool_calls = [tc for tc in output.tool_calls if tc["name"] in structured_output_tools]
    if structured_tool_calls:
        # Single call: parse args, handle errors, return response
        structured_response = structured_output_tools[tool_call["name"]].parse(tool_call["args"])
        return {"messages": [...], "structured_response": structured_response}
```

## OutputToolBinding: Schema to Tool Conversion

`OutputToolBinding` is the bridge between a schema and a tool. It stores:

- **schema**: Original schema (Pydantic, dataclass, TypedDict, or dict)
- **schema_kind**: Classification (`'pydantic'`, `'dataclass'`, `'typeddict'`, `'json_schema'`)
- **tool**: `StructuredTool` instance with `args_schema` bound to the JSON schema

The `parse()` method reconstructs the original type from tool call arguments:

```python
def parse(self, tool_args: dict[str, Any]) -> SchemaT | dict[str, Any]:
    return _parse_with_schema(self.schema, self.schema_kind, tool_args)
```

**Parsing Flow:**

1. For dict schemas: Return arguments as-is (no validation)
2. For typed schemas: Use Pydantic's `TypeAdapter` to validate Python type
3. On validation error: Raise `ValueError` with schema name and error details

This allows the factory to maintain a single mapping of structured output tool names to their binding metadata throughout the agent's lifetime, enabling quick lookup during response handling.

## Error Handling and Validation

### Error Types

**StructuredOutputError** (base class):

- Holds the `AIMessage` that caused the error
- Parent of specific error types

**MultipleStructuredOutputsError**:

Raised when a single structured output schema is expected but the model calls multiple structured output tools.

```python
tool_names = [tc["name"] for tc in structured_tool_calls]
exception = MultipleStructuredOutputsError(tool_names, output)
```

**StructuredOutputValidationError**:

Raised when tool call arguments fail to parse according to the schema.

```python
exception = StructuredOutputValidationError(tool_name, source_exception, output)
```

### Retry Logic (ToolStrategy Only)

When `handle_errors` is enabled and a validation error occurs during tool parsing:

1. `_handle_structured_output_error()` determines if retry should happen
2. Returns `(should_retry: bool, error_message: str)`
3. If retry: Error message wrapped in `ToolMessage` appended to conversation
4. Model receives error context and can correct its response

**Error Callback:**

```python
should_retry, error_message = _handle_structured_output_error(
    exception, effective_response_format
)
if not should_retry:
    raise exception from exc

# Return error message to model
return {
    "messages": [
        output,
        ToolMessage(
            content=error_message,
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
        ),
    ],
}
```

The model's next turn receives the error and can attempt to correct the output.

### Validation Limitations

**Dict schemas skip validation:** Raw JSON schema dicts (not Pydantic, dataclass, or TypedDict) return arguments as-is:

```python
if schema_kind == "json_schema":
    return data  # No validation, no retry possible
```

To enable validation and retries, express schemas as Pydantic models or TypedDicts.

**Provider strategy has no retry:** Native structured output is provider-enforced; the agent receives valid JSON or an API error. No in-conversation retry is possible for schema mismatches.

## State and Lifecycle

### State Fields

The agent state includes structured output handling via:

- **messages**: Includes tool calls and tool messages from structured output invocation
- **structured_response**: Holds the parsed schema instance (set when output is valid, cleared on error retry)

### Lifecycle Events

```
User input
  ↓
[pre-model middleware]
  ↓
_get_bound_model() → detect strategy, bind model with tools or provider format
  ↓
model.invoke() → model returns AIMessage with tool_calls (ToolStrategy)
                 or text (ProviderStrategy)
  ↓
_handle_model_output() → parse response, validate against schema
  ↓
[structured_response set] or [error + retry message]
  ↓
[post-model middleware]
  ↓
Return to user or continue loop
```

## Configuration and Middleware

Middleware can override `response_format` at invocation time via `ModelRequest.override()`:

```python
class MyMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        # Narrow union response format to a specific variant
        narrow_format = ToolStrategy(schema=request.response_format.schema_specs[0])
        return handler(request.override(response_format=narrow_format))
```

The agent re-detects strategy and rebuilds tool bindings on each invocation, allowing dynamic schema changes. However, **all structured output schemas must be declared upfront**—middleware cannot add new schemas not present in the initial `response_format`.

## Example: Weather Agent with Structured Output

```python
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

class WeatherResponse(BaseModel):
    """Current weather forecast."""
    location: str
    temperature_f: int
    condition: str
    humidity_percent: int

# Create agent with structured output
agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    response_format=WeatherResponse,  # AutoStrategy
    system_prompt="You are a weather forecaster. Return accurate weather data.",
)

# Invoke
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]
})

# Access structured response
weather: WeatherResponse = result["structured_response"]
print(f"Temperature: {weather.temperature_f}°F, Condition: {weather.condition}")
```

With explicit error handling:

```python
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model="openai:gpt-4o",
    response_format=ToolStrategy(
        schema=WeatherResponse,
        handle_errors=True,  # Retry on validation errors
        tool_message_content="Invalid weather data format. Please provide: location, temperature_f, condition, humidity_percent.",
    ),
)
```

## See Also

- [Agent Factory](/openwiki/agent-factory.md): Entry point for creating agents; handles schema registration and strategy binding
- [Agent Execution Flow](/openwiki/agent-execution.md): Runtime loop where structured output is parsed and validated
- [LangChain Structured Output Documentation](https://python.langchain.com/docs/guides/structured_output/)
