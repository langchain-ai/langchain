---
type: API Reference
title: Structured Output API Reference
description: Complete API reference for structured output classes, error types, and validation mechanisms used by agents to enforce JSON schema compliance on language model responses.
tags: [structured-output, api-reference, schema, validation, error-handling]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-29T08:20:01.046Z
sources:
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
  - id: openwiki-source-ec30ab6256dd50cc670919f6
    resource: repo://libs/langchain_v1/langchain/agents/structured_output.py
generated: { by: "openwiki/0.5.0", at: "2026-10-29T08:20:01.046Z" }
---

## Overview

The structured output module provides classes and functions for enforcing JSON schema compliance on language model responses. It implements three strategy patterns—tool-based, provider-native, and automatic—each with distinct tradeoffs for compatibility, validation, and retry behavior.

This reference covers the primary classes and error types used by the factory to parse, validate, and manage structured responses.

## Response Format Strategies

The module defines three strategy classes that control how structured output is enforced:

### ToolStrategy

Tool-based structured output that wraps a schema as an artificial tool and uses tool calling to enforce compliance.

```python
@dataclass(init=False)
class ToolStrategy(Generic[SchemaT]):
    schema: type[SchemaT] | UnionType | dict[str, Any]
    """Schema for the tool calls."""
    
    schema_specs: list[_SchemaSpec[Any]]
    """Schema specs for the tool calls (split by Union variants)."""
    
    tool_message_content: str | None
    """Custom content for the tool message returned after parsing."""
    
    handle_errors: (
        bool | str | type[Exception] | tuple[type[Exception], ...] | Callable[[Exception], str]
    )
    """Error handling strategy:
    
    - `True`: Catch all errors with default template
    - `False`: Let exceptions propagate without retry
    - `str`: Custom error message for all errors
    - `type[Exception]` or `tuple[type[Exception], ...]`: Only retry specific types
    - `Callable[[Exception], str]`: Custom function returns error message per exception
    """
```

**Key Characteristics:**

- Works with any model supporting tool calling
- Supports Union types (multiple schema variants) by expanding to separate `schema_specs` entries
- `handle_errors` enables automatic retry on validation failures with configurable scope
- Raw JSON schema dicts bypass validation (see below)

**Important Limitation:** Raw JSON schema dictionaries (not Pydantic models, dataclasses, or TypedDicts) skip validation during parsing, making `handle_errors` ineffective for dict schemas. To enable validation and retry, use a Pydantic model, dataclass, or TypedDict instead.

### ProviderStrategy

Native structured output that delegates schema enforcement to the model provider's API.

```python
@dataclass(init=False)
class ProviderStrategy(Generic[SchemaT]):
    schema: type[SchemaT] | dict[str, Any]
    """Schema for native mode."""
    
    schema_spec: _SchemaSpec[SchemaT]
    """Schema spec for native mode."""
    
    def __init__(
        self,
        schema: type[SchemaT] | dict[str, Any],
        *,
        strict: bool | None = None,
    ) -> None:
        """Initialize with schema and optional strict mode flag."""
```

**Key Characteristics:**

- Delegates JSON schema generation and enforcement to the provider (e.g., OpenAI's `response_format` with `type: "json_schema"`)
- Does not consume tool slots; works alongside tool calling
- No automatic retry (provider side is strict)
- Converts schema to provider-specific format via `to_model_kwargs()`

**Provider Binding:**

The `to_model_kwargs()` method returns kwargs to bind to the model:

```python
def to_model_kwargs(self) -> dict[str, Any]:
    """Convert to kwargs for model binding.
    
    Returns:
        {"response_format": {"type": "json_schema", "json_schema": {...}}}
    """
```

### AutoStrategy

Automatically selects the best strategy based on model capabilities at invocation time.

```python
class AutoStrategy(Generic[SchemaT]):
    """Automatically select the best strategy for structured output."""
    
    schema: type[SchemaT] | dict[str, Any]
    """Schema for automatic mode."""
    
    def __init__(self, schema: type[SchemaT] | dict[str, Any]) -> None:
        """Initialize with schema."""
```

**Selection Logic:**

1. At model invocation time, `_supports_provider_strategy()` inspects the bound model
2. If model supports native structured output, uses `ProviderStrategy`
3. Otherwise falls back to `ToolStrategy`

**Support Detection:**

A model supports provider-native structured output if:

- Its profile includes `"structured_output": True`, AND
- Either no tools are used, OR the model is not a Gemini pre-3-series (which cannot mix tools with structured output), OR
- Model name matches fallback patterns (e.g., `gpt-4o`, `claude-opus`, `grok-4`)

## Schema Specification

### _SchemaSpec

Internal class describing a structured output schema, used by strategies to normalize schema types and generate JSON schemas.

```python
@dataclass(init=False)
class _SchemaSpec(Generic[SchemaT]):
    schema: type[SchemaT] | dict[str, Any]
    """The original schema (Pydantic, dataclass, TypedDict, or JSON schema dict)."""
    
    name: str
    """Schema name for tool calling (class name, 'title' field, or generated UUID fragment)."""
    
    description: str
    """Description extracted from docstring, 'description' field, or empty string."""
    
    schema_kind: SchemaKind
    """One of 'pydantic', 'dataclass', 'typeddict', or 'json_schema'."""
    
    json_schema: dict[str, Any]
    """JSON schema representation."""
    
    strict: bool | None = None
    """Whether to enforce strict validation (provider-side only)."""
```

**Supported Schema Types:**

| Type | Validation | JSON Schema | Tool Use |
|------|-----------|------------|----------|
| Pydantic BaseModel | Full (TypeAdapter) | Via `model_json_schema()` | Yes |
| `@dataclass` | Full (TypeAdapter) | Via `TypeAdapter.json_schema()` | Yes |
| `TypedDict` | Full (TypeAdapter) | Via `TypeAdapter.json_schema()` | Yes |
| `dict` (JSON Schema) | None (returned as-is) | Used directly | Yes (no validation) |

### Schema Type Detection

```python
schema_kind = Literal["pydantic", "dataclass", "typeddict", "json_schema"]
```

The `_SchemaSpec` constructor detects schema type by checking:

- `isinstance(schema, dict)` → `"json_schema"`
- `isinstance(schema, type) and issubclass(schema, BaseModel)` → `"pydantic"`
- `is_dataclass(schema)` → `"dataclass"`
- `is_typeddict(schema)` → `"typeddict"`
- Otherwise → raises `ValueError`

## Parsing and Validation

### _parse_with_schema

Core function for parsing and validating data against any supported schema type.

```python
def _parse_with_schema(
    schema: type[SchemaT] | dict[str, Any],
    schema_kind: SchemaKind,
    data: dict[str, Any]
) -> SchemaT | dict[str, Any]:
    """Parse data according to schema type.
    
    Args:
        schema: The schema type
        schema_kind: One of 'pydantic', 'dataclass', 'typeddict', 'json_schema'
        data: The data to parse
        
    Returns:
        Parsed instance matching the schema type
        
    Raises:
        ValueError: If parsing fails (wraps source exception)
    """
```

**Behavior by Schema Kind:**

- **json_schema**: Returns `data` as-is (no validation)
- **pydantic/dataclass/typeddict**: Uses Pydantic's `TypeAdapter[SchemaT]` to validate Python type, raising `ValueError` on failure

### OutputToolBinding

Bridge between a schema and a tool used by ToolStrategy for parsing tool call arguments.

```python
@dataclass
class OutputToolBinding(Generic[SchemaT]):
    """Information for tracking structured output tool metadata."""
    
    schema: type[SchemaT] | dict[str, Any]
    """The original schema provided."""
    
    schema_kind: SchemaKind
    """Classification of the schema type."""
    
    tool: BaseTool
    """LangChain tool instance created from the schema."""
    
    @classmethod
    def from_schema_spec(cls, schema_spec: _SchemaSpec[SchemaT]) -> Self:
        """Create from a SchemaSpec.
        
        Creates a StructuredTool with:
        - name: schema_spec.name
        - description: schema_spec.description
        - args_schema: schema_spec.json_schema
        """
    
    def parse(self, tool_args: dict[str, Any]) -> SchemaT | dict[str, Any]:
        """Parse tool call arguments.
        
        Args:
            tool_args: The arguments from the tool call
            
        Returns:
            Parsed response according to schema type
            
        Raises:
            ValueError: If parsing fails
        """
```

### ProviderStrategyBinding

Bridge for parsing native provider-enforced JSON responses.

```python
@dataclass
class ProviderStrategyBinding(Generic[SchemaT]):
    """Information for tracking native structured output metadata."""
    
    schema: type[SchemaT] | dict[str, Any]
    """The original schema provided."""
    
    schema_kind: SchemaKind
    """Classification of the schema type."""
    
    @classmethod
    def from_schema_spec(cls, schema_spec: _SchemaSpec[SchemaT]) -> Self:
        """Create from a SchemaSpec."""
    
    def parse(self, response: AIMessage) -> SchemaT | dict[str, Any]:
        """Parse AIMessage content as JSON according to schema.
        
        Args:
            response: The AIMessage containing the structured output
            
        Returns:
            Parsed response according to schema
            
        Raises:
            ValueError: If text extraction, JSON parsing, or validation fails
            
        Process:
            1. Extract text content from AIMessage (handles str, dict, and list formats)
            2. Parse text as JSON via json.loads()
            3. Validate against schema using _parse_with_schema()
        """
    
    @staticmethod
    def _extract_text_content_from_message(message: AIMessage) -> str:
        """Extract text from AIMessage.
        
        Handles:
        - Simple str content
        - Content with type='text'
        - Generic 'content' field
        - Falls back to str() for other types
        """
```

## Error Types

### StructuredOutputError

Base exception class for all structured output errors.

```python
class StructuredOutputError(Exception):
    """Base class for structured output errors."""
    
    ai_message: AIMessage
```

All structured output errors hold the `AIMessage` that caused the error.

### MultipleStructuredOutputsError

Raised when the model returns multiple structured output tool calls when only one is expected.

```python
class MultipleStructuredOutputsError(StructuredOutputError):
    """Raised when model returns multiple structured output tool calls."""
    
    def __init__(
        self,
        tool_names: list[str],
        ai_message: AIMessage
    ) -> None:
        """Initialize with tool names and AI message."""
        self.tool_names = tool_names
        self.ai_message = ai_message
```

**Lifecycle:**

- Raised by `_handle_model_output()` when `len(structured_tool_calls) > 1`
- Subject to retry logic via `_handle_structured_output_error()` if `handle_errors` is enabled
- If retry disabled, propagates to caller

### StructuredOutputValidationError

Raised when tool call arguments fail to parse according to the schema.

```python
class StructuredOutputValidationError(StructuredOutputError):
    """Raised when tool call arguments fail to parse."""
    
    def __init__(
        self,
        tool_name: str,
        source: Exception,
        ai_message: AIMessage
    ) -> None:
        """Initialize with tool name, source exception, and AI message."""
        self.tool_name = tool_name
        self.source = source
        self.ai_message = ai_message
```

**Lifecycle:**

- Raised by `_handle_model_output()` during `OutputToolBinding.parse()` or `ProviderStrategyBinding.parse()`
- For `ToolStrategy`, subject to retry logic via `_handle_structured_output_error()` if `handle_errors` is enabled
- For `ProviderStrategy`, always propagates (no retry possible)

## Error Handling Mechanics

### _handle_structured_output_error

Determines whether to retry on structured output validation failures and generates retry messages.

```python
def _handle_structured_output_error(
    exception: Exception,
    response_format: ResponseFormat[Any],
) -> tuple[bool, str]:
    """Handle structured output error.
    
    Args:
        exception: The exception that occurred
        response_format: The response format strategy
        
    Returns:
        (should_retry: bool, error_message: str)
        
    Only applies to ToolStrategy; returns (False, "") for other strategies.
    """
```

**Error Handling Dispatch:**

- `handle_errors is False` → `(False, "")`
- `handle_errors is True` → `(True, STRUCTURED_OUTPUT_ERROR_TEMPLATE.format(error=str(exception)))`
- `handle_errors: str` → `(True, handle_errors)`
- `handle_errors: type[Exception]` → Check `isinstance(exception, handle_errors)`, return default template if match
- `handle_errors: tuple[type[Exception], ...]` → Check any match, return default template if match
- `handle_errors: Callable[[Exception], str]` → `(True, handle_errors(exception))`

**Error Template:**

```python
STRUCTURED_OUTPUT_ERROR_TEMPLATE = "Error: {error}\n Please fix your mistakes."
```

### Retry Flow (ToolStrategy Only)

When validation fails and retry is enabled:

1. Exception caught in `_handle_model_output()`
2. `_handle_structured_output_error()` determines retry eligibility
3. If retry enabled:
   - Generate `ToolMessage` with error content and original tool call id
   - Return `{"messages": [output, error_tool_message]}`
   - Model receives error context in conversation and can correct
4. If retry disabled:
   - Raise `StructuredOutputValidationError` or `MultipleStructuredOutputsError`

## Response Parsing Pipeline

### _handle_model_output

Central dispatcher for structured response parsing after model invocation.

```python
def _handle_model_output(
    output: AIMessage,
    effective_response_format: ResponseFormat[Any] | None
) -> dict[str, Any]:
    """Handle model output including structured responses.
    
    Args:
        output: The AI message output from the model
        effective_response_format: The actual strategy used (may differ if auto-detected)
        
    Returns:
        {"messages": [...], "structured_response": ...}  # structured_response optional
        
    Raises:
        StructuredOutputValidationError: If parsing fails and retry disabled
        MultipleStructuredOutputsError: If multiple outputs and retry disabled
    """
```

**ProviderStrategy Path:**

1. Check if output has no tool calls (native structured output should not have tools)
2. Create `ProviderStrategyBinding` from strategy's schema spec
3. Call `binding.parse(output)`:
   - Extracts text from AIMessage
   - Parses JSON
   - Validates against schema
4. On success: return `{"messages": [output], "structured_response": parsed}`
5. On failure: raise `StructuredOutputValidationError`

**ToolStrategy Path:**

1. Filter output's tool calls to find structured output tools
2. If multiple structured tool calls:
   - Check retry eligibility
   - If retry: return error messages for model correction
   - If no retry: raise `MultipleStructuredOutputsError`
3. If single structured tool call:
   - Get `OutputToolBinding` for tool
   - Call `binding.parse(tool_call["args"])`:
     - Validates arguments against schema
   - On success: return `{"messages": [...], "structured_response": parsed}`
   - On failure:
     - Check retry eligibility
     - If retry: return error message
     - If no retry: raise `StructuredOutputValidationError`

## Integration Points

### Model Capability Detection

```python
def _supports_provider_strategy(
    model: str | BaseChatModel,
    tools: list[BaseTool | dict[str, Any]] | None = None
) -> bool:
    """Check if model supports provider-specific structured output.
    
    Args:
        model: Model name string or BaseChatModel instance
        tools: Optional tool list (needed for Gemini compatibility checks)
        
    Returns:
        True if model supports provider native structured output
        
    Detection:
        1. If BaseChatModel: check profile["structured_output"]
           - Exception: Gemini < 3-series with tools returns False
        2. If model name matches FALLBACK_MODELS_WITH_STRUCTURED_OUTPUT patterns
        3. Otherwise: False
    """
```

### Strategy Selection (AutoStrategy)

During model binding (`_get_bound_model`):

1. User passes raw schema or explicit `AutoStrategy` → factory wraps in `ToolStrategy` upfront
2. At invocation time, `_supports_provider_strategy()` checks model
3. If model supports provider: create `ProviderStrategy` and bind via `model.bind(..., response_format=...)`
4. Otherwise: use pre-built `ToolStrategy` and bind via `model.bind_tools(..., tool_choice="any")`

### Structured Output Tool Registration

At agent creation:

```python
# Upfront registration for ToolStrategy
if tool_strategy_for_setup:
    for response_schema in tool_strategy_for_setup.schema_specs:
        structured_tool_info = OutputToolBinding.from_schema_spec(response_schema)
        structured_output_tools[structured_tool_info.tool.name] = structured_tool_info
```

These bindings are stored in the agent's state and used throughout the lifetime for quick lookup during response parsing.

## Constants

### Response Format Union

```python
ResponseFormat = ToolStrategy[SchemaT] | ProviderStrategy[SchemaT] | AutoStrategy[SchemaT]
```

Union type for all supported response format strategies.

### Schema Kind Literal

```python
SchemaKind = Literal["pydantic", "dataclass", "typeddict", "json_schema"]
```

Classification of schema types.
