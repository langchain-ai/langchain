---
type: "Reference"
title: "Structured Output: Request Formats and Response Parsing"
description: "Schema specification for structured generation (JSON schema, Pydantic, TypedDict), model enforcement strategies by provider, response marshaling and validation, and error recovery."
tags: [structured-output, response-format, schema, validation, pydantic, json-schema, chat-models, output-parser, agent]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-21T08:14:55.070Z
sources:
  - id: openwiki-source-c52037e7b642f7ac5a7642a8
    resource: repo://libs/core/langchain_core/language_models/chat_models.py
  - id: openwiki-source-3afd4d755887348d6b456045
    resource: repo://libs/core/langchain_core/output_parsers/base.py
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
  - id: openwiki-source-ec30ab6256dd50cc670919f6
    resource: repo://libs/langchain_v1/langchain/agents/structured_output.py
generated: { by: "openwiki/0.5.0", at: "2026-11-21T08:14:55.070Z" }
---

## Overview

**Structured output** is the practice of constraining a language model's response to match a specific schema, ensuring type-safe, validated data extraction. The LangChain ecosystem provides multiple complementary layers:

1. **Core abstractions** (`BaseChatModel`, `BaseOutputParser`): The fundamental interfaces for schema-aware models and response parsing.
2. **Chat model binding** (`with_structured_output()`): Direct schema binding on chat models for single-request structured output.
3. **Output parsers** (`BaseOutputParser` subclasses): Composable, reusable parsers for custom response transformation.
4. **Agent framework strategies** (`ToolStrategy`, `ProviderStrategy`, `AutoStrategy`): Multi-turn structured output with error recovery and retry logic.

This page documents the schema specification, enforcement mechanisms, and integration patterns across all layers.

## Core Abstractions

### BaseChatModel and Schema Binding

`BaseChatModel` (in `repo://libs/core/langchain_core/language_models/chat_models.py`) is the base class for all chat model implementations in LangChain. It provides the foundation for structured output through:

- **Standard message I/O**: Input and output normalized to `BaseMessage` objects.
- **Tool binding** (`bind_tools()`): Declares available tools for function calling.
- **Schema binding** (`with_structured_output()`): Wraps a model to constrain output to a schema.
- **Streaming support**: Progressive token-by-token or chunk-by-chunk output.
- **Callback hooks**: Integration with observability and tracing systems.

### BaseOutputParser and Response Parsing

`BaseOutputParser` (in `repo://libs/core/langchain_core/output_parsers/base.py#L140-L170`) is the abstract base for all output parsing in LangChain:

```python
class BaseOutputParser(Generic[T], RunnableSerializable[LanguageModelOutput, T]):
    """Base class for parsing language model outputs into structured data."""

    @abstractmethod
    def parse_result(
        self, result: list[Generation], *, partial: bool = False
    ) -> T:
        """Parse a list of Generation objects into a structured output."""

    def parse(self, text: str) -> T:
        """Parse a single string model output into some structure."""

    async def aparse_result(self, result: list[Generation], *, partial: bool = False) -> T:
        """Async version of parse_result()."""
```

**Key responsibilities:**

- Accept raw model output (strings, messages, or Generation objects)
- Validate against a schema (Pydantic, dict, custom)
- Raise `OutputParserException` on validation failure
- Support sync, async, batch, and streaming execution (via `RunnableSerializable`)

## Schema Types and Representation

Structured output supports four schema types, normalized via `_SchemaSpec` in the agent framework:

| Type | Example | Validation | Serialization |
|------|---------|-----------|---------------|
| **Pydantic BaseModel** | `class Response(BaseModel): ...` | Full (via `TypeAdapter`) | `model_json_schema()` |
| **Dataclass** | `@dataclass class Response: ...` | Full (via `TypeAdapter`) | `TypeAdapter(type).json_schema()` |
| **TypedDict** | `class Response(TypedDict): ...` | Full (via `TypeAdapter`) | `TypeAdapter(type).json_schema()` |
| **JSON Schema dict** | `{"type": "object", "properties": {...}}` | None (returned as-is) | Used directly |

**Key principle:** All types are normalized to JSON Schema via Pydantic's `TypeAdapter`, enabling consistent provider integration. Dict schemas skip validation to avoid false negatives on incomplete JSON schema definitions.

## Schema Binding: with_structured_output()

`with_structured_output()` is a method on `BaseChatModel` that wraps a model to enforce structured responses:

```python
def with_structured_output(
    self,
    schema: dict[str, Any] | type,
    *,
    include_raw: bool = False,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
    """Wrap a model to return outputs conforming to a schema.
    
    Args:
        schema: Pydantic model, TypedDict, dataclass, or JSON Schema dict
        include_raw: If True, return dict with 'raw', 'parsed', 'parsing_error'
        **kwargs: Provider-specific options (e.g., method='json_schema')
    
    Returns:
        A Runnable that takes model input and outputs parsed schema instances
    """
```

**Flow:**

1. Schema is registered with the model via provider-specific binding (tool calling, JSON mode, etc.)
2. Model is invoked; response is captured
3. Response is parsed: JSON text → Python object via schema validation
4. Parsed object (or error dict if `include_raw=True`) is returned

**Limitations:**

- Single-turn only (no in-conversation error recovery)
- Requires model support for structured output (OpenAI, Claude, etc.)
- No automatic retry; validation errors propagate

## Agent Framework Structured Output

For multi-turn agents with error recovery and complex schema handling, the agent factory provides three strategies.

### Response Format Strategies

Three strategies control how structured output is enforced in agents:

#### ToolStrategy: Tool-Based Structured Output

The model is presented with an artificial tool whose arguments must match the schema.

**Lifecycle:**

1. Schema is wrapped as a `StructuredTool` with JSON schema as `args_schema`
2. Tool is added to model's tool list with `tool_choice="any"` to force use
3. Model generates a tool call with the schema name
4. Tool call arguments are parsed via `_parse_with_schema()` using `TypeAdapter`
5. Parsed result stored in agent state's `structured_response` field
6. Empty `ToolMessage` returned (tool has no real execution)

**Advantages:**

- Works with all models that support tool calling
- Full validation for non-dict schemas
- Automatic retry on validation errors (configurable via `handle_errors`)
- Supports Union types (multiple schema variants)

**Limitations:**

- Requires tool calling capability
- Adds tool to tool list (may consume slots on some models)
- Raw JSON schema dicts skip validation, making `handle_errors` inert

**Error Handling:**

The `handle_errors` parameter controls retry behavior when tool argument validation fails:

- `True` (default): Catch all errors, retry with default error template
- `False`: Let exceptions propagate without retry
- `str`: Catch all errors, retry with custom message
- `type[Exception]` or `tuple[type[Exception], ...]`: Only retry specific exception types
- `Callable[[Exception], str]`: Custom function returns retry message per exception

Failed parses generate a `ToolMessage` with the error message, allowing the model to correct its output on the next turn.

#### ProviderStrategy: Native Structured Output

The schema is sent to the model provider's native structured output API (e.g., OpenAI's `response_format` with `type: "json_schema"`). The provider enforces schema compliance; the agent parses the JSON response.

**Lifecycle:**

1. Schema converted to JSON Schema via `TypeAdapter.json_schema()` or `model_json_schema()`
2. Wrapped in provider-specific format: `{"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": ...}}`
3. Passed to model via `model.bind(..., response_format={...})`
4. Model returns JSON text (guaranteed valid by provider)
5. Text parsed via `json.loads()` then validated against schema
6. Parsed result stored in agent state's `structured_response` field

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

1. Its profile includes `"structured_output": True` (via `model.profile`), AND
2. Not a pre-3-series Gemini model (which cannot mix tools with structured output), OR
3. Model name matches fallback patterns like `gpt-4o`, `claude-opus`, etc.

#### AutoStrategy: Automatic Strategy Selection

Defers strategy selection until model invocation time. The factory inspects the bound model and chooses:

1. **ProviderStrategy** if the model supports native structured output
2. **ToolStrategy** as fallback for all other models

**Lifecycle:**

1. User passes raw schema (wrapped in `AutoStrategy` automatically)
2. Factory pre-builds `ToolStrategy` tools upfront
3. At model call time, `_supports_provider_strategy()` checks model capabilities
4. If supported, `ProviderStrategy` is created and model kwargs bound
5. Otherwise, `ToolStrategy` is used (tools already prepared)

**Advantage:** Best of both worlds—uses provider when available, falls back to tools.

### Schema Specification and Normalization

The agent factory normalizes schemas via `_SchemaSpec`:

```python
@dataclass
class _SchemaSpec(Generic[SchemaT]):
    """Describes a structured output schema."""
    
    schema: type[SchemaT] | dict[str, Any]
    """Original schema (Pydantic, dataclass, TypedDict, or JSON schema dict)"""
    
    name: str
    """Name for tool calling; derived from class name, title field, or generated UUID"""
    
    description: str
    """Description for tool calling; from docstring, description field, or empty"""
    
    schema_kind: SchemaKind
    """Classification: 'pydantic', 'dataclass', 'typeddict', or 'json_schema'"""
    
    json_schema: dict[str, Any]
    """JSON Schema representation for tool or provider binding"""
    
    strict: bool | None
    """Whether to enforce strict validation (provider-specific)"""
```

**Parsing and Validation:**

The `_parse_with_schema()` function reconstructs typed objects from tool or provider responses:

```python
def _parse_with_schema(
    schema: type[SchemaT] | dict[str, Any], 
    schema_kind: SchemaKind, 
    data: dict[str, Any]
) -> SchemaT | dict[str, Any]:
    """Parse data according to schema type.
    
    - Dict schemas: Return data as-is (no validation)
    - Typed schemas: Use TypeAdapter to validate Python type
    - On error: Raise ValueError with schema name and details
    """
```

### OutputToolBinding and ProviderStrategyBinding

**`OutputToolBinding`** (for ToolStrategy):

Stores mapping of schema to tool:

```python
@dataclass
class OutputToolBinding(Generic[SchemaT]):
    schema: type[SchemaT] | dict[str, Any]
    schema_kind: SchemaKind
    tool: BaseTool  # StructuredTool instance
    
    @classmethod
    def from_schema_spec(cls, schema_spec: _SchemaSpec) -> Self:
        """Create binding from schema spec."""
    
    def parse(self, tool_args: dict[str, Any]) -> SchemaT | dict[str, Any]:
        """Parse tool arguments and reconstruct original type."""
```

**`ProviderStrategyBinding`** (for ProviderStrategy):

Stores schema and parsing metadata:

```python
@dataclass
class ProviderStrategyBinding(Generic[SchemaT]):
    schema: type[SchemaT] | dict[str, Any]
    schema_kind: SchemaKind
    
    @classmethod
    def from_schema_spec(cls, schema_spec: _SchemaSpec) -> Self:
        """Create binding from schema spec."""
    
    def parse(self, response: AIMessage) -> SchemaT | dict[str, Any]:
        """Extract text, parse JSON, validate against schema."""
```

### Error Handling

**Exception Types:**

- `StructuredOutputError`: Base class; holds the `AIMessage` that caused the error
- `MultipleStructuredOutputsError`: Model called multiple structured output tools when only one is expected
- `StructuredOutputValidationError`: Tool call arguments failed to parse according to schema

**Retry Logic (ToolStrategy Only):**

When `handle_errors` is enabled and validation fails:

1. `_handle_structured_output_error()` determines if retry should happen
2. Returns `(should_retry: bool, error_message: str)`
3. If retry: Error message wrapped in `ToolMessage` appended to conversation
4. Model receives error context and can correct its response on next turn

**Validation Limitations:**

- **Dict schemas skip validation:** Raw JSON schema dicts return arguments as-is, making `handle_errors` inert. Use Pydantic models or TypedDicts to enable validation and retries.
- **Provider strategy has no retry:** Native structured output is provider-enforced; the agent receives valid JSON or an API error. No in-conversation retry is possible for schema mismatches.

### Agent Integration

The `create_agent()` factory integrates structured output through:

#### 1. Upfront Schema Registration

At agent creation time, `ToolStrategy` schemas are pre-built as `OutputToolBinding` instances:

```python
if tool_strategy_for_setup:
    for response_schema in tool_strategy_for_setup.schema_specs:
        structured_tool_info = OutputToolBinding.from_schema_spec(response_schema)
        structured_output_tools[structured_tool_info.tool.name] = structured_tool_info
```

#### 2. Model Binding During Invocation

`_get_bound_model()` performs auto-detection and binds the model:

```python
# Determine effective response format (auto-detect if needed)
if isinstance(response_format, AutoStrategy):
    if _supports_provider_strategy(request.model, tools=request.tools):
        effective_response_format = ProviderStrategy(schema=response_format.schema)
    else:
        effective_response_format = ToolStrategy(schema=response_format.schema)
else:
    effective_response_format = response_format

# Bind model based on effective strategy
if isinstance(effective_response_format, ProviderStrategy):
    return request.model.bind_tools(final_tools, **effective_response_format.to_model_kwargs())
elif isinstance(effective_response_format, ToolStrategy):
    return request.model.bind_tools(final_tools, tool_choice="any", ...)
else:
    return request.model
```

#### 3. Response Parsing

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

### State and Lifecycle

The agent state includes structured output handling via:

- **messages**: Includes tool calls and tool messages from structured output invocation
- **structured_response**: Holds the parsed schema instance (set when output is valid, cleared on error retry or when no structured output is produced)

**Structured Response Clearing:**

The `_build_commands()` function ensures that `structured_response` field is explicitly cleared when:

1. Agent is configured with `response_format` (has_structured_output = True)
2. Model invocation did not produce a structured response

This prevents stale values from previous checkpointed turns in persistent agent state:

```python
state: dict[str, Any] = {"messages": messages}
if model_response.structured_response is not None:
    state["structured_response"] = model_response.structured_response
elif has_structured_output:
    state["structured_response"] = None  # Explicitly clear to avoid stale values
```

### Configuration and Middleware

Raw schemas and strategy decisions are passed to middleware via `ModelRequest.response_format`. Middleware can override `response_format` at invocation time via `ModelRequest.override()`:

```python
class MyMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        # Narrow union response format to a specific variant
        narrow_format = ToolStrategy(schema=request.response_format.schema_specs[0])
        return handler(request.override(response_format=narrow_format))
```

**Key constraint:** All structured output schemas must be declared upfront—middleware cannot add new schemas not present in the initial `response_format`.

## Supported Models

The following models support provider-native structured output (AutoStrategy will use ProviderStrategy when available):

**OpenAI:**
- `gpt-4.1`, `gpt-4o`, `gpt-5`, `gpt-5.1`, `gpt-5.2`, `gpt-5.3`, `gpt-5.4`, `gpt-5.5`
- `gpt-6` (astra, luna, sol)

**Claude:**
- `claude-fable-5`, `claude-mythos-5`
- `claude-haiku-4-5`
- `claude-opus-4-5`, `claude-opus-4-6`, `claude-opus-4-7`, `claude-opus-4-8`
- `claude-sonnet-4-5`, `claude-sonnet-4-6`

**Other:**
- `grok-4`, `grok-build`

For models not in this list, ToolStrategy is used. Pre-3-series Gemini models are excluded from provider strategy when tools are present due to incompatibility.

## Examples

### Chat Model with Structured Output

```python
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

class Answer(BaseModel):
    """An answer to a question."""
    answer: str
    confidence: float

model = ChatOpenAI(model="gpt-4o")
structured_model = model.with_structured_output(Answer)

result = structured_model.invoke("What is 2+2?")
print(result)  # Answer(answer="4", confidence=0.99)
```

### Agent with Structured Output

```python
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

class WeatherResponse(BaseModel):
    """Weather forecast response."""
    location: str
    temperature_f: int
    condition: str
    humidity_percent: int

# Create agent with structured output (AutoStrategy)
agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    response_format=WeatherResponse,  # Wrapped in AutoStrategy automatically
    system_prompt="You are a weather forecaster. Return accurate weather data.",
)

# Invoke with error recovery
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]
})

weather: WeatherResponse = result["structured_response"]
print(f"Temperature: {weather.temperature_f}°F, Condition: {weather.condition}")
```

### Explicit Strategy Selection

```python
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy

# Use provider strategy explicitly
agent_native = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    response_format=ProviderStrategy(schema=WeatherResponse),
)

# Use tool strategy with error handling
agent_tools = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    response_format=ToolStrategy(
        schema=WeatherResponse,
        handle_errors=True,  # Retry on validation failure
        tool_message_content="Invalid weather data format. Please provide: location, temperature_f, condition, humidity_percent.",
    ),
)
```

## See Also

- [Chat Models: Core Abstractions and Protocol](/openwiki/concepts/chat-models.md): Base class contract, lifecycle phases, message I/O semantics
- [Structured Output: Type-Safe Results from Language Models](/openwiki/workflows/structured-output.md): Workflow patterns combining all structured output approaches
- [OpenAI Provider Implementation](/openwiki/integrations/openai-provider.md): Provider-specific structured output support
