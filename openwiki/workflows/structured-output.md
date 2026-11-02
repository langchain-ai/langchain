---
type: Workflow
title: "Structured Output: Type-Safe Results from Language Models"
description: "Patterns for obtaining structured, validated output from language models using Pydantic schemas, JSON schemas, and output parsers. Bind schemas to chat models with with_structured_output() or parse raw model responses using BaseOutputParser implementations. For agents, use ToolStrategy, ProviderStrategy, or AutoStrategy to enforce structured output."
tags: [structured-output, output-parser, pydantic, json-schema, validation, type-safe, agent-framework]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-29T08:20:01.046Z
sources:
  - id: openwiki-source-c52037e7b642f7ac5a7642a8
    resource: repo://libs/core/langchain_core/language_models/chat_models.py
  - id: openwiki-source-e7908d069731cffec228727e
    resource: repo://libs/core/langchain_core/output_parsers/__init__.py
  - id: openwiki-source-3afd4d755887348d6b456045
    resource: repo://libs/core/langchain_core/output_parsers/base.py
  - id: openwiki-source-b6969337572a90d4b6262e7b
    resource: repo://libs/core/langchain_core/output_parsers/json.py
  - id: openwiki-source-d89c127dbba78772dc5369ed
    resource: repo://libs/core/langchain_core/output_parsers/openai_tools.py
  - id: openwiki-source-50275a07684ed251c972e512
    resource: repo://libs/core/langchain_core/output_parsers/pydantic.py
  - id: openwiki-source-464a5ad0b0c47b5c599ed9c2
    resource: repo://libs/core/langchain_core/output_parsers/transform.py
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
  - id: openwiki-source-ec30ab6256dd50cc670919f6
    resource: repo://libs/langchain_v1/langchain/agents/structured_output.py
generated: { by: "openwiki/0.5.0", at: "2026-10-29T08:20:01.046Z" }
---

## Overview

**Structured output** is the practice of obtaining data from a language model in a well-defined, type-safe format rather than as unstructured text. The LangChain ecosystem provides multiple complementary patterns:

1. **Native structured output** via `with_structured_output()` on chat models—leverages the model provider's native capabilities (function calling, JSON mode) to ensure the model returns data matching a schema.
2. **Output parsing** via `BaseOutputParser` subclasses—parses raw model text or tool calls into Python objects matching a schema, with validation and error handling.
3. **Streaming structured output**—progressively yields partial objects as the model generates them, enabling responsive UIs and early feedback.
4. **Agent framework strategies**—when building agents with `create_agent()`, use `ToolStrategy`, `ProviderStrategy`, or `AutoStrategy` to enforce structured responses with automatic retry and error handling.

These patterns combine to enable reliable data extraction, API response generation, entity classification, and form filling workflows where the model's output type and shape are known and enforced.

---

## Core Abstraction: Output Parsers

An **output parser** is a `Runnable[LanguageModelOutput, T]` that transforms raw model output into a structured object of type `T`. All output parsers inherit from **`BaseOutputParser`** (in `repo://libs/core/langchain_core/output_parsers/base.py#L140-L170`), which:

- Inherits from `RunnableSerializable`, enabling sync (`invoke`), async (`ainvoke`), batch, and streaming execution.
- Accepts language model outputs (strings, `AIMessage` objects, or `Generation` objects).
- Returns validated, structured data matching a schema.
- Raises `OutputParserException` on invalid or unparseable input.

### BaseOutputParser Interface

```python
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.outputs import Generation

class BaseOutputParser(Runnable[LanguageModelOutput, T]):
    """Base class for parsing LLM outputs."""

    @abstractmethod
    def parse_result(
        self, result: list[Generation], *, partial: bool = False
    ) -> T:
        """Parse a list of Generation objects into a structured output.
        
        Args:
            result: List of Generation objects from the LLM.
            partial: Whether to parse incomplete/streaming output.
        
        Returns:
            Parsed and validated object of type T.
        """

    def invoke(
        self,
        input: str | BaseMessage,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> T:
        """Synchronously parse a string or message into T."""

    async def ainvoke(
        self,
        input: str | BaseMessage,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> T:
        """Asynchronously parse a string or message into T."""
```

Every output parser must implement `parse_result()` to transform raw generations. The parser automatically integrates with chat models to form pipelines like `model | parser`.

---

## Schema Binding: with_structured_output()

**`with_structured_output()`** is a method on `BaseChatModel` (in `repo://libs/core/langchain_core/language_models/chat_models.py#L2385-L2565`) that wraps a model to enforce structured output by leveraging the provider's native capabilities. This is the **recommended approach** when your model supports structured output natively.

### Method Signature

```python
def with_structured_output(
    self,
    schema: dict[str, Any] | type,
    *,
    include_raw: bool = False,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
    """Model wrapper that returns outputs formatted to match the given schema.
    
    Args:
        schema: The output schema. Can be:
            - A Pydantic model class
            - A TypedDict class
            - A JSON Schema (as a dict)
            - An OpenAI function/tool schema
        
        include_raw: If False, return only the parsed output.
                    If True, return a dict with 'raw', 'parsed', and 'parsing_error'.
    
    Returns:
        A Runnable that takes the same inputs as the chat model and outputs
        the structured result (or None on error if include_raw=True).
    
    Raises:
        NotImplementedError: If the model does not implement this method.
        ValueError: If unsupported kwargs are passed.
    """
```

### Usage: Pydantic Schema

The most common pattern uses a **Pydantic model** to define the schema. The model automatically validates and returns instances of that class:

```python
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

class AnswerWithJustification(BaseModel):
    """An answer to the user question along with justification."""
    answer: str
    justification: str

model = ChatOpenAI(model="gpt-4o", temperature=0)
structured_model = model.with_structured_output(AnswerWithJustification)

# Returns an AnswerWithJustification instance
result = structured_model.invoke(
    "What weighs more: a pound of bricks or a pound of feathers?"
)
# -> AnswerWithJustification(
#     answer='They weigh the same',
#     justification='Both a pound of bricks and a pound of feathers weigh one pound...'
# )
```

### Usage: JSON Schema and Dictionary Schemas

For lower-overhead structured output without full Pydantic validation, pass a **JSON schema dict** or use `convert_to_openai_tool()` to create a function schema:

```python
from langchain_core.utils.function_calling import convert_to_openai_tool

# Using a Pydantic model converted to OpenAI function schema
class Answer(BaseModel):
    answer: str
    justification: str

schema_dict = convert_to_openai_tool(Answer)
structured_model = model.with_structured_output(schema_dict)

result = structured_model.invoke("Your question")
# -> Returns a dict (not validated): {'answer': '...', 'justification': '...'}
```

### Usage: include_raw for Error Handling

Set `include_raw=True` to capture both the raw model response and any parsing errors:

```python
structured_model = model.with_structured_output(
    AnswerWithJustification,
    include_raw=True
)

result = structured_model.invoke("Your question")
# -> {
#     'raw': AIMessage(...),  # The raw model response
#     'parsed': AnswerWithJustification(...) or None,
#     'parsing_error': None or BaseException
# }
```

This pattern is essential when you want to inspect what the model actually returned before validation, or to handle parsing failures gracefully without exceptions.

---

## Explicit Output Parsing

When you cannot use `with_structured_output()` (because your model doesn't support it, or you need custom parsing logic), explicitly compose a parser with the model using the `|` operator.

### PydanticOutputParser

**`PydanticOutputParser`** (in `repo://libs/core/langchain_core/output_parsers/pydantic.py#L19-L124`) parses **JSON-formatted text output** from a model into a Pydantic model instance:

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")

parser = PydanticOutputParser(pydantic_object=Person)

# Get format instructions to include in the prompt
instructions = parser.get_format_instructions()
# -> "The output should be formatted as a JSON instance that conforms to..."

model = ChatOpenAI(model="gpt-4o")
chain = model | parser

result = chain.invoke(
    f"Extract the person from this text: John is 30 years old.\n\n{instructions}"
)
# -> Person(name='John', age=30)
```

The parser:
1. Extracts JSON from the model's text (via `parse_json_markdown()`).
2. Validates the JSON against the Pydantic schema.
3. Returns a Pydantic instance or raises `OutputParserException`.

### JsonOutputParser

**`JsonOutputParser`** (in `repo://libs/core/langchain_core/output_parsers/json.py#L31-L139`) parses JSON output without strict schema validation (though it can accept an optional Pydantic model for validation):

```python
from langchain_core.output_parsers import JsonOutputParser

# Without validation
parser = JsonOutputParser()
result = parser.invoke('{"name": "Alice", "age": 25}')
# -> {'name': 'Alice', 'age': 25}

# With optional Pydantic validation
parser = JsonOutputParser(pydantic_object=Person)
result = parser.invoke('{"name": "Alice", "age": 25}')
# -> Person(name='Alice', age=25)
```

This parser supports **streaming** and can yield partial JSON objects as they are generated.

### Tool-Based Parsers

When using **function calling** (tool calls) instead of raw text output, use `JsonOutputToolsParser` or `PydanticToolsParser` to extract and validate the structured data from tool call arguments:

```python
from langchain_core.output_parsers import PydanticToolsParser
from pydantic import BaseModel

class ExtractedEntity(BaseModel):
    entity_type: str
    name: str
    confidence: float

# When the model has been bound with tools
parser = PydanticToolsParser(tools=[ExtractedEntity], first_tool_only=True)

# parser.parse_result() extracts the tool call arguments and validates them
# against the ExtractedEntity schema
```

---

## Validation and Error Handling

Structured output patterns include validation at multiple layers:

### Schema-Level Validation

**Pydantic** automatically validates field types and constraints:

```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str
    age: int = Field(gt=0, le=150)  # Age must be 1-150

# If the model returns {"name": "Alice", "age": -5}:
# Pydantic raises ValidationError: age must be greater than 0
```

### Output Parser Exception Handling

Parsers raise `OutputParserException` on invalid input. When composing with `with_fallbacks()`, the chain can automatically retry or provide a default:

```python
from langchain_core.exceptions import OutputParserException

model = ChatOpenAI(model="gpt-4o")
structured = model.with_structured_output(Person)

# Fallback: return None if parsing fails
safe_structured = structured.with_fallbacks(
    [lambda _: None],
    exception_key="parsing_error"
)
```

Alternatively, use `include_raw=True` to capture errors without raising:

```python
structured = model.with_structured_output(Person, include_raw=True)
result = structured.invoke("Your question")

if result['parsing_error']:
    # Handle gracefully
    print(f"Parsing failed: {result['parsing_error']}")
else:
    # Use the parsed object
    person = result['parsed']
```

### StructuredOutputError (Agent Framework)

The agent framework (in `repo://libs/langchain_v1/langchain/agents/structured_output.py#L35-L74`) provides specialized error types for structured output via tool strategies:

- **`StructuredOutputValidationError`**: Tool arguments failed schema validation.
- **`MultipleStructuredOutputsError`**: Model returned multiple tool calls when only one was expected.

These are automatically caught and can trigger retries with error context sent back to the model.

---

## Agent Framework: Structured Output Strategies

When building agents with `create_agent()` from LangChain, you can enforce structured responses using three complementary strategies defined in `repo://libs/langchain_v1/langchain/agents/structured_output.py`. Each strategy trades off between compatibility, validation, and control.

### Schema Types and Resolution

All strategies work with the same schema types, which are normalized through **`_SchemaSpec`** (in `repo://libs/langchain_v1/langchain/agents/structured_output.py#L106-L193`):

| Type | Example | Validation | Description |
|------|---------|-----------|-------------|
| **Pydantic model** | `class Response(BaseModel): ...` | Full validation | Type-safe, validated by Pydantic's TypeAdapter |
| **Dataclass** | `@dataclass class Response: ...` | Full validation | Standard Python dataclass, validated by TypeAdapter |
| **TypedDict** | `class Response(TypedDict): ...` | Full validation | Typed dictionary, validated by TypeAdapter |
| **JSON Schema dict** | `{"type": "object", "properties": {...}}` | None (dict schemas skip validation) | Raw schema, returned as-is without validation |
| **Union types** | `ToolStrategy(schema=A \| B)` | Full validation (each variant) | Multiple schema variants, each validated |

**Important:** Raw JSON schema dicts (plain `dict[str, Any]`) bypass validation and are returned as-is. To enable automatic retry on errors, use Pydantic, dataclass, or TypedDict schemas instead.

### ToolStrategy: Tool-Based Structured Output

Use the model's tool-calling capability to enforce a structured response by wrapping the schema as an artificial tool. The model calls the tool with the structured data as arguments.

**When to use:**
- You need maximum compatibility (all models with tool-calling support)
- You want automatic retry on validation errors
- You want full Pydantic validation

**Lifecycle:**

1. Schema is wrapped as a `StructuredTool` with the schema's JSON schema as `args_schema`
2. Tool is added to the model's tool list with `tool_choice="any"` to force use
3. Model generates a tool call with the schema name
4. Tool call arguments are parsed via `_parse_with_schema()` using Pydantic's `TypeAdapter`
5. Parsed result stored in `structured_response` state field
6. Empty `ToolMessage` returned (tool has no real execution)

**Configuration:**

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel

class WeatherResponse(BaseModel):
    """Weather forecast response."""
    location: str
    temperature: int
    condition: str

agent = create_agent(
    model="openai:gpt-4o",
    response_format=ToolStrategy(
        schema=WeatherResponse,
        handle_errors=True,  # Retry on validation failure
        tool_message_content="Successfully captured weather data."
    ),
)
```

**Error Handling:**

The `handle_errors` parameter controls retry behavior on validation failure:

- `True` (default): Catch all errors, retry with default error template
- `False`: Let exceptions propagate without retry
- `str`: Catch all errors, retry with custom message
- `type[Exception]` or `tuple[type[Exception], ...]`: Only retry specific exception types
- `Callable[[Exception], str]`: Custom function returns retry message per exception

Failed parses generate a `ToolMessage` with the error message, allowing the model to correct its output.

**Limitations:**
- Requires tool calling capability
- Adds tool to the tool list (may consume tool slot on some models)
- Raw JSON schema dicts skip validation, making `handle_errors` inert

### ProviderStrategy: Native Structured Output

Use the model provider's native structured output API (e.g., OpenAI's `response_format` with `type: "json_schema"`). The provider enforces schema compliance; the agent only parses the JSON response.

**When to use:**
- Your model supports native structured output (OpenAI, Claude 3.5, Gemini 2)
- You want provider-side schema enforcement
- You don't need automatic retry (provider guarantees valid JSON)
- You want to preserve tool slots for actual tools

**Lifecycle:**

1. Schema converted to JSON Schema via Pydantic's `model_json_schema()`
2. Wrapped in provider-specific format: `{"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": ...}}`
3. Passed to model via `model.bind(..., response_format={...})`
4. Model returns JSON text (guaranteed valid by provider)
5. Text parsed via `json.loads()` then validated against schema
6. Parsed result stored in `structured_response` state field

**Configuration:**

```python
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="openai:gpt-4o",
    response_format=ProviderStrategy(
        schema=WeatherResponse,
        strict=True  # Request strict schema enforcement (if supported)
    ),
)
```

**Advantages:**
- Provider enforces schema—no invalid JSON possible
- Doesn't consume tool slots
- Works alongside tool calling
- Strict mode available (supported providers only)

**Limitations:**
- Limited to models with native structured output
- No automatic retry on validation errors (provider side is strict)
- Requires explicit model capability testing

**Capability Detection:**

A model supports provider-native structured output if checked by `_supports_provider_strategy()` (in `repo://libs/langchain_v1/langchain/agents/factory.py#L578-L623`):

1. Its profile includes `"structured_output": True`, AND
2. Not a pre-3-series Gemini model (which cannot mix tools with structured output), OR
3. Model name matches fallback patterns like `gpt-4o`, `claude-opus`, `gemini-2`, etc.

### AutoStrategy: Automatic Strategy Selection

Defer strategy selection until runtime based on model capabilities. The framework inspects the bound model at invocation time and chooses the best strategy.

**When to use:**
- You want to use provider-native structured output when available, fall back to tools otherwise
- You're building portable agents that work across multiple model families
- **Recommended for most use cases**

**Selection logic:**

1. If model supports native structured output → **ProviderStrategy**
2. Otherwise → **ToolStrategy** (always available)

**Usage:**

```python
# Automatic detection (recommended)
agent = create_agent(
    model="openai:gpt-4o",
    response_format=WeatherResponse,  # Wrapped in AutoStrategy automatically
)

# Explicit AutoStrategy
from langchain.agents.structured_output import AutoStrategy

agent = create_agent(
    model="openai:gpt-4o",
    response_format=AutoStrategy(schema=WeatherResponse),
)
```

The agent framework automatically converts raw schemas to `AutoStrategy` to enable capability detection. At model invocation time (in `_get_bound_model()` at `repo://libs/langchain_v1/langchain/agents/factory.py#L1399-L1422`):

1. Checks if the bound model supports provider strategy via `_supports_provider_strategy()`
2. If yes, creates `ProviderStrategy` and binds model kwargs
3. If no, uses pre-built `ToolStrategy` (tools already prepared at setup)

---

## Streaming Structured Output

**Streaming structured output** progressively yields partial objects as the model generates tokens. This is especially useful for:

- Responsive UIs that display results incrementally.
- Early feedback before the full response is generated.
- Reducing latency for progressive data processing.

### BaseCumulativeTransformOutputParser

Streaming is implemented via `BaseCumulativeTransformOutputParser` (in `repo://libs/core/langchain_core/output_parsers/transform.py#L99-L180`), which accumulates streamed tokens and yields partial parsed objects:

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser(pydantic_object=Person)

# Stream from a model
model = ChatOpenAI(model="gpt-4o")

# Synchronous streaming
for partial_person in (model | parser).stream("Extract the person from: ..."):
    print(partial_person)
    # Yields progressively more complete Person objects as JSON is parsed

# Asynchronous streaming
async for partial_person in (model | parser).astream("Extract the person from: ..."):
    print(partial_person)
```

The parser:
1. Accumulates text chunks as they arrive.
2. Calls `parse_result(..., partial=True)` after each chunk.
3. Yields only if the parsed object changed from the previous iteration.
4. Supports `diff=True` to yield only the differences (JSONPatch format) rather than full objects.

### PydanticOutputParser Streaming

`PydanticOutputParser` extends `JsonOutputParser` and supports the same streaming interface. It validates each accumulated chunk:

```python
parser = PydanticOutputParser(pydantic_object=Person)

for partial_person in (model | parser).stream("Extract: ..."):
    # partial_person is None until the JSON is complete enough to validate
    if partial_person:
        print(f"So far: {partial_person}")
```

### JsonOutputParser with Partial Mode

`JsonOutputParser.parse_result(..., partial=True)` uses `parse_json_markdown()` to extract partial JSON and `parse_partial_json()` to handle incomplete structures:

```python
from langchain_core.utils.json import parse_partial_json

# Handles incomplete JSON
partial_json = parse_partial_json('{"name": "Alice", "age"')
# -> {'name': 'Alice'}  (incomplete object still extracted)
```

---

## Use Cases and Examples

### Entity Extraction

Extract structured entities from unstructured text:

```python
from pydantic import BaseModel, Field

class Entity(BaseModel):
    entity_type: str = Field(description="Person, Place, or Organization")
    name: str
    confidence: float = Field(ge=0, le=1)

model = ChatOpenAI(model="gpt-4o")
extractor = model.with_structured_output(Entity)

result = extractor.invoke(
    "Steve Jobs founded Apple Computer in Los Altos, California."
)
# -> Entity(entity_type='Person', name='Steve Jobs', confidence=0.95)
```

### API Response Generation

Generate structured API responses:

```python
class APIResponse(BaseModel):
    status: str = Field(description="success or error")
    code: int = Field(description="HTTP status code")
    message: str
    data: dict | None = None

model = ChatOpenAI(model="gpt-4o")
api_gen = model.with_structured_output(APIResponse)

result = api_gen.invoke(
    "Generate an API response for a successful user creation at /api/users with id=123"
)
# -> APIResponse(status='success', code=201, message='User created', data={'id': 123})
```

### Data Classification

Classify items into predefined categories:

```python
from enum import Enum
from pydantic import BaseModel

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class ClassifiedReview(BaseModel):
    text: str
    sentiment: Sentiment
    confidence: float

model = ChatOpenAI(model="gpt-4o")
classifier = model.with_structured_output(ClassifiedReview)

result = classifier.invoke("This product is amazing! Best purchase ever.")
# -> ClassifiedReview(
#     text='This product is amazing! Best purchase ever.',
#     sentiment=<Sentiment.POSITIVE: 'positive'>,
#     confidence=0.98
# )
```

### Form Filling

Extract structured data to populate forms:

```python
class AddressForm(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str
    country: str

model = ChatOpenAI(model="gpt-4o")
address_extractor = model.with_structured_output(AddressForm)

result = address_extractor.invoke(
    "Extract the address: 1600 Pennsylvania Avenue NW, Washington, DC 20500, USA"
)
# -> AddressForm(
#     street='1600 Pennsylvania Avenue NW',
#     city='Washington',
#     state='DC',
#     zip_code='20500',
#     country='USA'
# )
```

---

## Implementation Patterns

### Choosing an Approach

| Pattern | When to Use | Pros | Cons |
|---------|-----------|------|------|
| **`with_structured_output()`** | Model supports native structured output | Efficient, reliable, native validation | Limited to supported models |
| **`PydanticOutputParser`** | JSON-formatted output from any model | Full Pydantic validation, streaming | Requires prompt engineering for JSON |
| **`JsonOutputParser`** | Fast, unvalidated JSON parsing | Low overhead, streaming | No validation, JSON extraction fragile |
| **Tool-based (function calling)** | Using bind_tools() or agent frameworks | Model-native, structured | Requires tool definition, model support |
| **`ToolStrategy` (agents)** | Maximum agent compatibility, need retry | Works with all tool-capable models, auto-retry | Tool slot consumption |
| **`ProviderStrategy` (agents)** | Provider has native support, strict schema | Provider-enforced, efficient | Limited model support, no retry |
| **`AutoStrategy` (agents)** | Best of both worlds for portable agents | Automatic best-strategy selection | Requires testing model support |

### Composing Parsers with Chains

Parsers are `Runnable` objects, so they compose seamlessly with models via the `|` operator:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=Person)

prompt = ChatPromptTemplate.from_template(
    "Extract the person: {text}\n\n{format_instructions}"
)

model = ChatOpenAI(model="gpt-4o")

# Chain: prompt -> model -> parser
chain = (
    prompt
    | model
    | parser
)

result = chain.invoke({
    "text": "Alice is 30 years old",
    "format_instructions": parser.get_format_instructions()
})
# -> Person(name='Alice', age=30)
```

### Custom Output Parsers

Implement custom parsing logic by subclassing `BaseOutputParser` or `BaseTransformOutputParser`:

```python
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.outputs import Generation
import xml.etree.ElementTree as ET

class XMLOutputParser(BaseOutputParser[dict]):
    """Parse XML output."""

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> dict:
        text = result[0].text.strip()
        try:
            root = ET.fromstring(text)
            return ET.todict(root)  # Convert to dict
        except ET.ParseError as e:
            raise OutputParserException(f"Invalid XML: {text}") from e
```

### Batching and Async

Output parsers inherit batch and async methods from `Runnable`:

```python
model = ChatOpenAI(model="gpt-4o")
parser = PydanticOutputParser(pydantic_object=Person)
chain = model | parser

# Batch process multiple inputs
results = chain.batch([
    "Extract: Alice, 30",
    "Extract: Bob, 25",
    "Extract: Charlie, 35"
])

# Async process
result = await chain.ainvoke("Extract: Alice, 30")

# Async batch
results = await chain.abatch([...])
```

---

## Provider-Specific Considerations

### OpenAI / Compatible APIs

OpenAI models support native structured output via **JSON Schema**:

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")
structured = model.with_structured_output(Person)
# Internally uses response_format with JSON Schema
```

For agents, both `ProviderStrategy` and `AutoStrategy` automatically detect and use OpenAI's native structured output:

```python
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4o",
    response_format=Person,  # AutoStrategy detects native support
)
```

### Anthropic

Anthropic models support structured output via **tool use** (function calling):

```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-3-5-sonnet")
structured = model.with_structured_output(Person)
# Internally uses tool_choice="any"
```

Anthropic's native structured output is detected by agent framework via `profile` metadata. For maximum compatibility with agents, `AutoStrategy` is recommended.

### Other Providers

For models that don't have native structured output support, use `PydanticOutputParser` or `JsonOutputParser` with explicit prompting to generate JSON. Agent-based structured output falls back to `ToolStrategy` automatically:

```python
from langchain_community.chat_models import ChatOllama
from langchain.agents import create_agent

model = ChatOllama(model="mistral")

# On agents, AutoStrategy falls back to ToolStrategy
agent = create_agent(
    model="ollama:mistral",
    response_format=Person,  # AutoStrategy -> ToolStrategy fallback
)
```

---

## References

- **BaseOutputParser**: `repo://libs/core/langchain_core/output_parsers/base.py#L140-L170`
- **PydanticOutputParser**: `repo://libs/core/langchain_core/output_parsers/pydantic.py#L19-L124`
- **JsonOutputParser**: `repo://libs/core/langchain_core/output_parsers/json.py#L31-L139`
- **with_structured_output()**: `repo://libs/core/langchain_core/language_models/chat_models.py#L2385-L2565`
- **BaseCumulativeTransformOutputParser (Streaming)**: `repo://libs/core/langchain_core/output_parsers/transform.py#L99-L180`
- **Agent Framework Structured Output**: `repo://libs/langchain_v1/langchain/agents/structured_output.py`
  - `ToolStrategy`: `repo://libs/langchain_v1/langchain/agents/structured_output.py#L195-L268`
  - `ProviderStrategy`: `repo://libs/langchain_v1/langchain/agents/structured_output.py#L271-L315`
  - `AutoStrategy`: `repo://libs/langchain_v1/langchain/agents/structured_output.py#L458-L469`
  - Schema resolution and error handling: `repo://libs/langchain_v1/langchain/agents/structured_output.py#L35-L193`
- **Agent Framework Strategy Detection**: `repo://libs/langchain_v1/langchain/agents/factory.py#L578-L623` (_supports_provider_strategy), `repo://libs/langchain_v1/langchain/agents/factory.py#L1399-L1422` (strategy selection at invocation)
