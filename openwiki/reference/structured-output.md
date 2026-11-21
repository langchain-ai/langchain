---
type: API Reference
title: Structured Output API Reference
description: Complete API reference for output parsers, structured output strategies, schema specifications, validation mechanisms, and error handling used to enforce JSON schema compliance on language model responses.
tags: [structured-output, api-reference, schema, validation, error-handling, output-parser]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-21T08:14:55.070Z
sources:
  - id: openwiki-source-3afd4d755887348d6b456045
    resource: repo://libs/core/langchain_core/output_parsers/base.py
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
  - id: openwiki-source-ec30ab6256dd50cc670919f6
    resource: repo://libs/langchain_v1/langchain/agents/structured_output.py
generated: { by: "openwiki/0.5.0", at: "2026-11-21T08:14:55.070Z" }
---

## Overview

Structured output is the system of parsing and validating language model responses against a specified JSON schema. The system provides two complementary layers:

1. **Output Parsers** (core/langchain_core): Generic base classes and interfaces for parsing LLM outputs into structured types. These enable custom parsing logic for any LLM output format.

2. **Response Format Strategies** (langchain/agents): Agent-level implementation that enforces schema compliance on model responses using three strategies—tool-based, provider-native, and automatic selection. These strategies directly use Pydantic's `TypeAdapter` for validation and support automatic retry on validation failure.

## Output Parsers (Base Abstractions)

Output parsers form the foundation for converting language model outputs into structured data. All parsers inherit from one of the base classes and implement a `parse()` or `parse_result()` method.

### BaseLLMOutputParser

Abstract base class for all parsers that work with `Generation` objects (the low-level output format from model invocations).

```python
class BaseLLMOutputParser(ABC, Generic[T]):
    """Abstract base class for parsing the outputs of a model."""
    
    @abstractmethod
    def parse_result(
        self, result: list[Generation], *, partial: bool = False
    ) -> T:
        """Parse a list of candidate model `Generation` objects.
        
        Args:
            result: List of Generation objects to be parsed.
            partial: Whether to parse the output as a partial result.
        
        Returns:
            Structured output of type T.
        """
    
    async def aparse_result(
        self, result: list[Generation], *, partial: bool = False
    ) -> T:
        """Async variant. Default implementation runs parse_result in executor."""
```

**Key Points:**

- Works with raw `Generation` objects from model outputs
- Supports both sync and async parsing
- `partial=True` indicates streaming or incomplete output (optional feature)

### BaseGenerationOutputParser

Extends `BaseLLMOutputParser` to work directly with model outputs as `Generation` or `ChatGeneration` objects. Implements `Runnable` interface.

```python
class BaseGenerationOutputParser(
    BaseLLMOutputParser[T], RunnableSerializable[LanguageModelOutput, T]
):
    """Base class to parse the output of an LLM call."""
    
    @property
    def InputType(self) -> Any:
        """Returns: str | AnyMessage"""
    
    @property
    def OutputType(self) -> type[T]:
        """Returns the inferred output type T."""
    
    def invoke(
        self,
        input: str | BaseMessage,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> T:
        """Parse a string or BaseMessage input.
        
        Wraps input in a Generation or ChatGeneration and calls parse_result.
        """
    
    async def ainvoke(...) -> T:
        """Async variant of invoke."""
```

**Usage Pattern:**

```python
from langchain_core.output_parsers import BaseGenerationOutputParser
from langchain_core.outputs import Generation
from langchain_core.messages import AIMessage

class CustomParser(BaseGenerationOutputParser[dict]):
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> dict:
        return {"parsed": result[0].text}

parser = CustomParser()
# Can invoke with string or message
result = parser.invoke("model output")
```

### BaseOutputParser

Higher-level parser that works directly with text strings (not raw `Generation` objects). Implements `Runnable[str | BaseMessage, T]`.

```python
class BaseOutputParser(
    BaseLLMOutputParser[T], RunnableSerializable[LanguageModelOutput, T]
):
    """Base class to parse the output of an LLM call.
    
    Output parsers help structure language model responses.
    """
    
    @property
    def InputType(self) -> Any:
        """Returns: str | AnyMessage"""
    
    @property
    def OutputType(self) -> type[T]:
        """Returns the inferred output type T from generic parameter."""
    
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> T:
        """Default implementation: calls parse(result[0].text)."""
    
    @abstractmethod
    def parse(self, text: str) -> T:
        """Parse a single string model output into structured format.
        
        Subclasses must implement this method.
        """
    
    async def aparse(self, text: str) -> T:
        """Async variant of parse. Default runs parse in executor."""
    
    def parse_with_prompt(
        self,
        completion: str,
        prompt: PromptValue,
    ) -> T:
        """Parse output with prompt context. Default calls parse()."""
    
    def get_format_instructions(self) -> str:
        """Return instructions on how the LLM should format its output.
        
        Useful for prompt engineering.
        """
```

**Example Implementation:**

```python
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException

class BooleanOutputParser(BaseOutputParser[bool]):
    true_val: str = "YES"
    false_val: str = "NO"

    def parse(self, text: str) -> bool:
        cleaned = text.strip().upper()
        if cleaned not in (self.true_val.upper(), self.false_val.upper()):
            raise OutputParserException(
                f"Expected {self.true_val} or {self.false_val}, got {cleaned}"
            )
        return cleaned == self.true_val.upper()

    @property
    def _type(self) -> str:
        return "boolean_output_parser"
    
    def get_format_instructions(self) -> str:
        return f"Respond with either {self.true_val} or {self.false_val}."
```

**Key Design Patterns:**

- `invoke(input)` automatically wraps string/message input and calls `parse_result()`
- `parse()` is the core method that subclasses override for string parsing
- `aparse()` defaults to running `parse()` in executor; override for true async
- `_type` property enables serialization/deserialization
- `OutputType` infers from generic parameter `BaseOutputParser[T]`

## Structured Output Response Format Strategies

The structured output system provides three strategies for enforcing schema compliance at the agent level. Each strategy represents a different tradeoff between compatibility, validation capability, and control flow.

### Strategy Classes

#### ToolStrategy

Wraps a schema as an artificial tool and forces the model to call it, parsing and validating the tool's arguments.

```python
@dataclass(init=False)
class ToolStrategy(Generic[SchemaT]):
    """Use a tool calling strategy for model responses."""
    
    schema: type[SchemaT] | UnionType | dict[str, Any]
    """The schema for tool calls. Can be Pydantic, dataclass, TypedDict, 
    JSON schema dict, or Union of these types."""
    
    schema_specs: list[_SchemaSpec[Any]]
    """Pre-computed schema specs (expanded if Union types used)."""
    
    tool_message_content: str | None
    """Custom content for the ToolMessage returned after parsing.
    Default: 'Returning structured response: {parsed_result}'"""
    
    handle_errors: (
        bool | str | type[Exception] | tuple[type[Exception], ...] 
        | Callable[[Exception], str]
    )
    """Error handling strategy on validation failure.
    
    - True (default): Catch all, retry with default template
    - False: No retry, raise exception
    - str: Catch all, retry with custom message
    - type[Exception]: Catch only this exception type
    - tuple[type[Exception], ...]: Catch these exception types
    - Callable[[Exception], str]: Function returns retry message per exception
    """
    
    def __init__(
        self,
        schema: type[SchemaT] | UnionType | dict[str, Any],
        *,
        tool_message_content: str | None = None,
        handle_errors: bool | str | type[Exception] | tuple[type[Exception], ...] 
                       | Callable[[Exception], str] = True,
    ) -> None:
        """Initialize ToolStrategy.
        
        Args:
            schema: Schema for the tool call arguments.
            tool_message_content: Custom tool message content.
            handle_errors: Error handling configuration.
        """
```

**Lifecycle:**

1. At agent creation, schema is registered as `OutputToolBinding` (wraps schema in `StructuredTool`)
2. Tool is added to model's tool list with `tool_choice="any"`
3. Model invokes the tool with arguments matching the schema
4. Arguments parsed via `_parse_with_schema` using `TypeAdapter.validate_python()`
5. If validation succeeds: `structured_response` state field set, empty `ToolMessage` returned
6. If validation fails and `handle_errors` enabled: `ToolMessage` with error sent back for retry
7. If validation fails and `handle_errors=False`: `StructuredOutputValidationError` raised

**Supported Schemas:**

| Type | Validation | JSON Schema Gen | Union Support |
|------|-----------|-----------------|---------------|
| Pydantic `BaseModel` | Full (TypeAdapter) | Via `model_json_schema()` | Yes |
| `@dataclass` | Full (TypeAdapter) | Via `TypeAdapter.json_schema()` | Yes |
| `TypedDict` | Full (TypeAdapter) | Via `TypeAdapter.json_schema()` | Yes |
| `dict` (JSON schema) | None—returned as-is | Used directly | Yes (no validation) |

**Important Limitation:** Raw JSON schema dicts skip validation, so `handle_errors` has no effect. To enable validation and retry, use a Pydantic model, dataclass, or TypedDict.

**Union Type Handling:**

```python
from typing import Union
from pydantic import BaseModel

class Success(BaseModel):
    result: str

class Error(BaseModel):
    error: str

# Union expanded to multiple schema_specs
response_format = ToolStrategy(schema=Success | Error)
# Creates two OutputToolBindings, one for each type
```

#### ProviderStrategy

Sends the schema to the model provider's native structured output API, letting the provider enforce compliance.

```python
@dataclass(init=False)
class ProviderStrategy(Generic[SchemaT]):
    """Use the model provider's native structured output method."""
    
    schema: type[SchemaT] | dict[str, Any]
    """Schema for native structured output."""
    
    schema_spec: _SchemaSpec[SchemaT]
    """Pre-computed schema specification."""
    
    def __init__(
        self,
        schema: type[SchemaT] | dict[str, Any],
        *,
        strict: bool | None = None,
    ) -> None:
        """Initialize ProviderStrategy.
        
        Args:
            schema: Schema to enforce via provider API.
            strict: Request strict provider-side validation (if supported).
        """
    
    def to_model_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs to bind to a model.
        
        Returns:
            {"response_format": {"type": "json_schema", 
                "json_schema": {"name": ..., "schema": ..., "strict": ...}}}
        """
```

**Lifecycle:**

1. Schema converted to JSON schema via `TypeAdapter.json_schema()` or `model_json_schema()`
2. Wrapped in provider-specific format: `{"type": "json_schema", "json_schema": {...}}`
3. Passed to model via `model.bind(response_format={...})`
4. Provider returns valid JSON text (enforcement is provider-side)
5. Text parsed via `json.loads()` then validated via `TypeAdapter.validate_python()`
6. Parsed result stored in `structured_response` state field
7. No automatic retry—validation errors propagate as `StructuredOutputValidationError`

**Provider Bindings:**

The `to_model_kwargs()` method returns kwargs for OpenAI-style APIs. The format is provider-agnostic but currently follows OpenAI's `response_format` structure.

**Capability Requirement:**

A model must support provider-native structured output (checked via `model.profile` or model name matching) to use this strategy. Otherwise, `AutoStrategy` falls back to `ToolStrategy`.

#### AutoStrategy

Defers strategy selection until invocation time, automatically choosing the best strategy based on model capabilities.

```python
class AutoStrategy(Generic[SchemaT]):
    """Automatically select the best strategy for structured output."""
    
    schema: type[SchemaT] | dict[str, Any]
    """Schema for automatic mode."""
    
    def __init__(self, schema: type[SchemaT] | dict[str, Any]) -> None:
        """Initialize AutoStrategy with schema."""
```

**Selection Logic:**

1. At model invocation time, `_supports_provider_strategy(model, tools)` checks:
   - Model's `profile["structured_output"]` flag
   - Model name against fallback patterns (`gpt-4o`, `claude-opus`, `grok-4`, etc.)
   - Special handling: Gemini pre-3-series excluded if tools are present (cannot mix tools + structured output)

2. If supported → use `ProviderStrategy`
3. Otherwise → use `ToolStrategy` (pre-built at agent creation)

**Lifecycle:**

1. User passes raw schema or `AutoStrategy(schema=...)`
2. Factory wraps in `AutoStrategy`, then pre-builds `ToolStrategy` for setup
3. At each model invocation, detection runs and effective strategy chosen
4. If `ProviderStrategy` selected: bind response_format to model
5. If `ToolStrategy` selected: use pre-built tools

**Recommended Usage:**

```python
from langchain.agents import create_agent
from pydantic import BaseModel

class Response(BaseModel):
    """Structured response."""
    answer: str
    confidence: float

# AutoStrategy automatically selected (recommended)
agent = create_agent(
    model="openai:gpt-4o",
    response_format=Response,  # Implicitly wrapped in AutoStrategy
)
```

### Schema Specifications

#### _SchemaSpec

Internal class that normalizes and describes a schema for use by strategies.

```python
@dataclass(init=False)
class _SchemaSpec(Generic[SchemaT]):
    """Describes a structured output schema."""
    
    schema: type[SchemaT] | dict[str, Any]
    """Original schema (Pydantic, dataclass, TypedDict, or JSON schema dict)."""
    
    name: str
    """Name for tool calling. Derived from class name, 'title' field, 
    or generated UUID fragment if unavailable."""
    
    description: str
    """Description for tool calling. Extracted from docstring or 
    'description' field, defaults to empty string."""
    
    schema_kind: SchemaKind
    """Classification: 'pydantic', 'dataclass', 'typeddict', or 'json_schema'."""
    
    json_schema: dict[str, Any]
    """Computed JSON schema representation for tool binding."""
    
    strict: bool | None
    """Whether to enforce strict validation (provider-side only)."""
    
    def __init__(
        self,
        schema: type[SchemaT] | dict[str, Any],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> None:
        """Initialize SchemaSpec.
        
        Args:
            schema: Schema to describe.
            name: Optional override for schema name.
            description: Optional override for description.
            strict: Optional strict validation flag.
        
        Raises:
            ValueError: If schema type is unsupported.
        """
```

**Schema Type Detection:**

The constructor determines `schema_kind` by checking in order:

1. `isinstance(schema, dict)` → `"json_schema"`
2. `isinstance(schema, type) and issubclass(schema, BaseModel)` → `"pydantic"`
3. `is_dataclass(schema)` → `"dataclass"`
4. `is_typeddict(schema)` → `"typeddict"`
5. Otherwise → raises `ValueError`

**Name Resolution:**

- If `name` parameter provided: use it
- Elif schema is dict: extract from `schema["title"]` or generate `response_format_{uuid4_fragment}`
- Else: use `schema.__name__` or generate `response_format_{uuid4_fragment}`

**Description Resolution:**

- If `description` parameter provided: use it
- Elif schema is dict: extract from `schema["description"]`
- Else: extract from `schema.__doc__` (docstring)
- Fallback: empty string

**JSON Schema Generation:**

- Pydantic: `BaseModel.model_json_schema()`
- Dataclass/TypedDict: `TypeAdapter(schema).json_schema()`
- JSON schema dict: used as-is (assumed valid)

### Parsing and Validation

#### _parse_with_schema

Core function for parsing and validating data against any supported schema type.

```python
def _parse_with_schema(
    schema: type[SchemaT] | dict[str, Any],
    schema_kind: SchemaKind,
    data: dict[str, Any]
) -> SchemaT | dict[str, Any]:
    """Parse data according to schema type.
    
    Args:
        schema: The schema type (Pydantic, dataclass, TypedDict, or JSON schema dict).
        schema_kind: One of 'pydantic', 'dataclass', 'typeddict', or 'json_schema'.
        data: The data to parse (typically dict[str, Any] from tool args or JSON).
    
    Returns:
        Parsed instance of type SchemaT, or dict if schema_kind is 'json_schema'.
    
    Raises:
        ValueError: If validation fails (for non-dict schemas).
    """
```

**Behavior by Schema Kind:**

| Kind | Validation | Returns | Raises |
|------|-----------|---------|--------|
| `pydantic` | Full via `TypeAdapter` | Instance of SchemaT | `ValueError` on failure |
| `dataclass` | Full via `TypeAdapter` | Instance of SchemaT | `ValueError` on failure |
| `typeddict` | Full via `TypeAdapter` | Instance of SchemaT | `ValueError` on failure |
| `json_schema` | None | Input `data` dict as-is | Never |

**TypeAdapter Usage:**

```python
from pydantic import TypeAdapter

<!-- openwiki: broken internal link [Response] file "Response" does not exist. Fix the href or restore the target, then delete this comment. -->
adapter = TypeAdapter[Response](Response)
result = adapter.validate_python({"key": "value"})
```

The adapter performs type coercion, field validation, and raises detailed `PydanticErrorMixin` exceptions on failure.

#### OutputToolBinding

Bridges a schema specification and tool for ToolStrategy.

```python
@dataclass
class OutputToolBinding(Generic[SchemaT]):
    """Information for tracking structured output tool metadata."""
    
    schema: type[SchemaT] | dict[str, Any]
    """Original schema provided."""
    
    schema_kind: SchemaKind
    """Classification of the schema type."""
    
    tool: BaseTool
    """LangChain StructuredTool instance created from the schema."""
    
    @classmethod
    def from_schema_spec(cls, schema_spec: _SchemaSpec[SchemaT]) -> Self:
        """Create from a _SchemaSpec.
        
        Args:
            schema_spec: The _SchemaSpec to convert.
        
        Returns:
            OutputToolBinding with StructuredTool created from spec.
        """
    
    def parse(self, tool_args: dict[str, Any]) -> SchemaT | dict[str, Any]:
        """Parse tool call arguments according to schema.
        
        Args:
            tool_args: Arguments from the tool call.
        
        Returns:
            Parsed response according to schema type.
        
        Raises:
            ValueError: If parsing/validation fails.
        """
```

**Usage in Agent:**

```python
# At agent creation time (upfront setup)
for schema_spec in tool_strategy.schema_specs:
    binding = OutputToolBinding.from_schema_spec(schema_spec)
    structured_output_tools[binding.tool.name] = binding

# At model output time
tool_call = output.tool_calls[0]  # From AIMessage
binding = structured_output_tools[tool_call["name"]]
parsed = binding.parse(tool_call["args"])  # Validates and parses
```

#### ProviderStrategyBinding

Handles parsing of provider-native structured output from AIMessage.

```python
@dataclass
class ProviderStrategyBinding(Generic[SchemaT]):
    """Information for tracking native structured output metadata."""
    
    schema: type[SchemaT] | dict[str, Any]
    """Original schema provided."""
    
    schema_kind: SchemaKind
    """Classification of the schema type."""
    
    @classmethod
    def from_schema_spec(cls, schema_spec: _SchemaSpec[SchemaT]) -> Self:
        """Create from a _SchemaSpec.
        
        Returns:
            ProviderStrategyBinding for parsing provider responses.
        """
    
    def parse(self, response: AIMessage) -> SchemaT | dict[str, Any]:
        """Parse AIMessage content according to schema.
        
        Args:
            response: The AIMessage containing structured output.
        
        Returns:
            Parsed response according to schema.
        
        Raises:
            ValueError: If text extraction, JSON parsing, or validation fails.
        """
    
    @staticmethod
    def _extract_text_content_from_message(message: AIMessage) -> str:
        """Extract text content from AIMessage.
        
        Handles multiple message content formats:
        - str: direct text content
        - list of dicts: extract type=text entries
        - generic content field: stringify
        
        Returns:
            Concatenated text content.
        """
```

**Message Content Handling:**

The parser extracts text from multiple message content variants:

1. If `message.content` is `str`: use directly
2. If `message.content` is list of dicts:
   - Extract entries with `type="text"` and `text` field
   - Fallback to `content` field if present
3. Otherwise: stringify and use

This accommodates different provider response formats.

### Error Handling

#### Error Classes

```python
class StructuredOutputError(Exception):
    """Base class for structured output errors."""
    
    ai_message: AIMessage
    """The AIMessage that caused the error."""


class MultipleStructuredOutputsError(StructuredOutputError):
    """Raised when model returns multiple structured output tool calls 
    when only one is expected."""
    
    def __init__(self, tool_names: list[str], ai_message: AIMessage) -> None:
        """Initialize.
        
        Args:
            tool_names: Names of tools called for structured output.
            ai_message: The AI message containing the invalid calls.
        """
        self.tool_names = tool_names


class StructuredOutputValidationError(StructuredOutputError):
    """Raised when structured output tool call arguments fail to parse 
    according to the schema."""
    
    def __init__(
        self, 
        tool_name: str, 
        source: Exception, 
        ai_message: AIMessage
    ) -> None:
        """Initialize.
        
        Args:
            tool_name: Name of the tool that failed.
            source: The exception that occurred (from TypeAdapter).
            ai_message: The AI message containing the invalid output.
        """
        self.tool_name = tool_name
        self.source = source
```

**Error Context:**

All structured output errors retain the `ai_message` that caused them, enabling:
- Error context inspection (what did the model generate?)
- Retry decision-making (is this worth resending to the model?)
- Detailed debugging logs

#### Error Handling Logic

The `_handle_structured_output_error()` function interprets the `handle_errors` configuration and determines retry behavior.

```python
def _handle_structured_output_error(
    exception: Exception,
    response_format: ResponseFormat[Any],
) -> tuple[bool, str]:
    """Handle structured output error.
    
    Args:
        exception: The exception that occurred.
        response_format: The response format strategy (must be ToolStrategy).
    
    Returns:
        (should_retry: bool, retry_message: str)
    """
```

**Decision Matrix:**

| `handle_errors` | Behavior |
|-----------------|----------|
| `False` | No retry, `should_retry=False` |
| `True` | Retry with default error template |
| `str` | Retry with custom message |
| `type[Exception]` | Retry if exception is instance of type |
| `tuple[type[Exception], ...]` | Retry if exception is instance of any type |
| `Callable[[Exception], str]` | Call function to generate retry message |

**Default Error Template:**

```python
"An error occurred while processing your response: {error}\n\nPlease try again."
```

**Retry Flow:**

1. Exception raised during `_parse_with_schema()`
2. `_handle_structured_output_error()` consulted
3. If `should_retry=True`: `ToolMessage` returned with error message, model retries
4. If `should_retry=False`: exception re-raised, agent stops

**ProviderStrategy Limitation:**

`_handle_structured_output_error()` only applies to `ToolStrategy`. Provider-side validation is strict (no retry); errors propagate directly as `StructuredOutputValidationError`.

## Integration with Agent Factory

The agent factory (`create_agent()`) orchestrates structured output through several integration points.

### Upfront Schema Registration

At agent creation time, `ToolStrategy` schemas are pre-registered as `OutputToolBinding` instances:

```python
# At create_agent() time
if isinstance(initial_response_format, (AutoStrategy, ToolStrategy)):
    # Pre-build ToolStrategy (may be AutoStrategy converted)
    tool_strategy_for_setup: ToolStrategy[Any] | None = ...
    
    # Register each schema as a tool
    structured_output_tools: dict[str, OutputToolBinding] = {}
    for schema_spec in tool_strategy_for_setup.schema_specs:
        binding = OutputToolBinding.from_schema_spec(schema_spec)
        structured_output_tools[binding.tool.name] = binding
```

This decouples schema registration (once at creation) from strategy selection (at each invocation).

### Model Invocation and Auto-Detection

The `_get_bound_model()` function performs auto-detection and binding:

```python
def _get_bound_model(
    request: ModelRequest[ContextT],
) -> tuple[Runnable[Any, Any], ResponseFormat[Any] | None]:
    """Get model with appropriate tool bindings.
    
    Returns:
        (bound_model, effective_response_format)
    """
    # Determine effective strategy
    if isinstance(response_format, AutoStrategy):
        if _supports_provider_strategy(model, tools):
            effective_response_format = ProviderStrategy(schema=response_format.schema)
        else:
            effective_response_format = tool_strategy_for_setup
    else:
        effective_response_format = response_format
    
    # Bind model
    if isinstance(effective_response_format, ProviderStrategy):
        kwargs = effective_response_format.to_model_kwargs()
        model_ = model.bind(**kwargs)
    elif isinstance(effective_response_format, ToolStrategy):
        # Tools already bound to model at creation time
        model_ = model
    
    return model_, effective_response_format
```

### Output Parsing and State Storage

The `_handle_model_output()` dispatcher handles response parsing:

```python
def _handle_model_output(
    output: AIMessage,
    effective_response_format: ResponseFormat[Any] | None,
) -> dict[str, Any]:
    """Handle model output including structured responses.
    
    Returns:
        {"messages": [...], "structured_response": ...} or {"messages": [...]}
    """
    # ProviderStrategy path: extract and parse JSON from AIMessage text
    if isinstance(effective_response_format, ProviderStrategy):
        binding = ProviderStrategyBinding.from_schema_spec(...)
        structured_response = binding.parse(output)
        return {"messages": [output], "structured_response": structured_response}
    
    # ToolStrategy path: find and parse structured tool calls
    if isinstance(effective_response_format, ToolStrategy):
        structured_calls = [
            tc for tc in output.tool_calls 
            if tc["name"] in structured_output_tools
        ]
        
        if structured_calls and len(structured_calls) == 1:
            tool_call = structured_calls[0]
            binding = structured_output_tools[tool_call["name"]]
            try:
                structured_response = binding.parse(tool_call["args"])
                return {
                    "messages": [output, ToolMessage(...)],
                    "structured_response": structured_response
                }
            except Exception as exc:
                # Check if should retry
                should_retry, error_msg = _handle_structured_output_error(exc, ...)
                if should_retry:
                    return {"messages": [output, ToolMessage(content=error_msg, ...)]}
                else:
                    raise
    
    return {"messages": [output]}
```

**State Field:**

The `structured_response` field in `AgentState` stores the parsed result. It is:
- Populated only if `response_format` is specified
- Typed according to the schema (e.g., `ResponseT` in `AgentState[ResponseT]`)
- Accessible to downstream middleware and agents

## Configuration and Extension

### Response Format Type Hierarchy

```python
ResponseFormat[SchemaT] = (
    ToolStrategy[SchemaT] 
    | ProviderStrategy[SchemaT] 
    | AutoStrategy[SchemaT]
)
```

### Supported Schema Types Summary

| Type | Parsing | Tool Binding | Validation | Union Support |
|------|---------|--------------|-----------|---------------|
| Pydantic `BaseModel` | TypeAdapter | Full JSON schema | Full | Yes |
| `@dataclass` | TypeAdapter | Full JSON schema | Full | Yes |
| `TypedDict` | TypeAdapter | Full JSON schema | Full | Yes |
| `dict` (JSON schema) | Passthrough | Used directly | None | Yes (no validation) |

### Capability Detection

A model supports provider-native structured output if:

1. Its profile includes `structured_output: True`, **AND**
2. Either no tools are present, **OR**
3. The model is not a Gemini pre-3-series (which cannot mix tools + structured output), **OR**
4. Model name matches a fallback pattern (checked via regex)

Fallback patterns include: `gpt-4o`, `claude-opus`, `grok-4`, and others.

## Best Practices

### Choosing a Strategy

| Scenario | Strategy | Rationale |
|----------|----------|-----------|
| Need compatibility with any model | `ToolStrategy` | Works with all tool-supporting models |
| Want strict provider validation | `ProviderStrategy` | No local validation overhead |
| Uncertain about model capabilities | `AutoStrategy` (default) | Best of both worlds |
| Need automatic retry on validation | `ToolStrategy` with `handle_errors=True` | Only supported in tool strategy |
| Using Pydantic/dataclass/TypedDict | Any | Full validation available |
| Using raw JSON schema dict | `ToolStrategy` or `ProviderStrategy` | No validation, returned as-is |

### Error Handling Patterns

```python
from langchain.agents.structured_output import ToolStrategy, StructuredOutputValidationError

# Pattern 1: Strict (fail fast)
response_format = ToolStrategy(schema=MySchema, handle_errors=False)

# Pattern 2: Lenient (auto-retry with default message)
response_format = ToolStrategy(schema=MySchema, handle_errors=True)

# Pattern 3: Selective (retry only specific errors)
response_format = ToolStrategy(
    schema=MySchema,
    handle_errors=(ValueError, TypeError)
)

# Pattern 4: Custom retry message
response_format = ToolStrategy(
    schema=MySchema,
    handle_errors="Please correct the formatting and try again."
)

# Pattern 5: Custom decision function
def custom_error_handler(exc: Exception) -> str:
    if isinstance(exc, ValueError):
        return f"Value error: {exc}. Please fix this."
    return "Unknown error. Try again."

response_format = ToolStrategy(
    schema=MySchema,
    handle_errors=custom_error_handler
)
```

### Union Type Patterns

```python
from typing import Union
from pydantic import BaseModel

class Success(BaseModel):
    result: str

class Error(BaseModel):
    error_code: int
    message: str

# Union of multiple schema types
response_format = ToolStrategy(schema=Success | Error)
# or using Union syntax:
response_format = ToolStrategy(schema=Union[Success, Error])

# Client receives structured_response as one of the types
# (Type checking requires isinstance checks or discriminator fields)
```

### Validation and Coercion

Pydantic's `TypeAdapter.validate_python()` performs:

1. **Type coercion**: string "123" → int 123 (if field type is int)
2. **Field validation**: runs any `@field_validator` decorators
3. **Model validation**: runs any `@model_validator` decorators
4. **Detailed errors**: raises `ValidationError` with per-field messages

```python
from pydantic import BaseModel, Field, field_validator

class Response(BaseModel):
    temperature: float = Field(..., ge=0, le=100)
    
    @field_validator("temperature")
    @classmethod
    def validate_range(cls, v):
        if v < -50 or v > 50:
            raise ValueError("Extreme temperature outside expected range")
        return v
```

## Related Pages

- **[Structured Output and Response Formatting](/openwiki/structured-output.md)**: High-level overview of structured output system, strategy selection, and agent integration
<!-- openwiki: broken internal link [/openwiki/output-parsers.md] file "/openwiki/output-parsers.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- **[Output Parsers](/openwiki/output-parsers.md)** (if exists): Documentation of specific parser implementations (JSON, Pydantic, XML, etc.)
