---
type: "Reference"
title: "OpenAI Provider Implementation"
description: "Reference implementation of the partner pattern using OpenAI. Covers ChatModel implementation, message conversion, tool calling, structured output, vision support, and streaming."
tags: ["chat-model", "openai", "tool-calling", "structured-output", "vision", "streaming", "partner-pattern"]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-25T08:23:03.589Z
sources:
  - id: openwiki-source-738512768ef81ae009b097ac
    resource: repo://libs/partners/openai/langchain_openai/chat_models/base.py
  - id: openwiki-source-b65307f1d05bd74532af2eee
    resource: repo://libs/partners/openai/langchain_openai/decisions/base.py
generated: { by: "openwiki/0.5.0", at: "2026-10-25T08:23:03.589Z" }
---

# OpenAI Provider Implementation

The OpenAI integration (`langchain-openai`) provides production-ready implementations of LangChain chat models, embeddings, and decision systems that target the official [OpenAI API specifications](https://github.com/openai/openai-openapi). The integration demonstrates the partner pattern for adapting a complex, multi-API provider into LangChain's unified abstractions.

## Overview: Provider Architecture

The OpenAI package contains three main components:

1. **Chat Models** (`ChatOpenAI`, `BaseChatOpenAI`): The primary LLM interface implementing `BaseChatModel`.
2. **Decisions API** (`OpenAIDecisions`): A specialized runnable for typed question-answering.
3. **Embeddings** (`OpenAIEmbeddings`, `AzureOpenAIEmbeddings`): Text embedding implementations (not covered here).

The chat model hierarchy is:
- `BaseChatOpenAI`: Abstract base supporting both Chat Completions and Responses APIs
- `ChatOpenAI`: Concrete subclass routing between APIs based on model and parameters
- `AzureChatOpenAI`: Azure-specific variant with authentication and deployment handling

## Chat Model Inheritance and Required Methods

`BaseChatOpenAI` inherits from `BaseChatModel` (in `langchain_core`) and extends it with OpenAI-specific client management, payload construction, and API routing. Key required method implementations:

### `_generate(messages, stop, run_manager, **kwargs) → ChatResult`

The synchronous generation method that all chat models must implement. In OpenAI, this is the main workhorse:

```python
def _generate(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    run_manager: CallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> ChatResult:
    """Generate response using Chat Completions or Responses API."""
    payload = self._get_request_payload(messages, stop=stop, **kwargs)
    # Routes to either:
    # - Chat Completions for standard prompts
    # - Responses API for reasoning/mid-conversation tools/audio
    # - Structured output parsing for response_format
    response = self.client.create(**payload)  # or responses.create()
    return self._create_chat_result(response)
```

The method:
1. Constructs the request payload via `_get_request_payload()`, which converts LangChain messages to OpenAI format and applies API routing logic.
2. Selects the appropriate OpenAI API endpoint (Chat Completions, Responses, or structured output).
3. Invokes the OpenAI client synchronously.
4. Parses the response into a `ChatResult` containing generations, token usage, and metadata.

Error handling translates OpenAI exceptions to LangChain model errors (see Error Handling section below).

### `_stream(messages, stop, run_manager, **kwargs) → Iterator[ChatGenerationChunk]`

Streaming implementation that yields partial message chunks as they arrive:

```python
def _stream(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    run_manager: CallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> Iterator[ChatGenerationChunk]:
    """Stream response chunks."""
    kwargs["stream"] = True
    payload = self._get_request_payload(messages, stop=stop, **kwargs)
    with self.client.create(**payload) as response:
        for chunk in response:
            generation_chunk = self._convert_chunk_to_generation_chunk(chunk, ...)
            if run_manager:
                run_manager.on_llm_new_token(generation_chunk.text, chunk=...)
            yield generation_chunk
```

Streaming uses Server-Sent Events (SSE) over HTTP. OpenAI pushes delta updates that are assembled into complete messages. The wrapper:
- Sets `stream=True` in the payload.
- Optionally enables `stream_usage` to include token counts in streaming output.
- Applies per-chunk timeout (`stream_chunk_timeout`) for async streams to detect stalled connections.
- Converts raw OpenAI deltas into `ChatGenerationChunk` objects with response metadata.

### Async Methods (`_agenerate`, `_astream`)

Async variants use `self.async_client` (an `AsyncOpenAI` instance) and are invoked via the same routing logic. The async path is preferred when concurrency is needed.

### `_llm_type` Property

```python
@property
def _llm_type(self) -> str:
    return "openai-chat"
```

Returns a provider identifier for logging and serialization.

## API Key Management and Configuration

### API Key Resolution Order

The OpenAI integration resolves API keys through a multi-step process:

1. **Explicit `api_key` parameter** (highest priority): Can be a string, sync callable, or async callable returning a string.
2. **Environment variable `OPENAI_API_KEY`** (read at init time by LangChain).
3. **LangSmith Gateway** (if `LANGSMITH_GATEWAY` is set): Uses `LANGSMITH_API_KEY` or `LANGSMITH_GATEWAY_API_KEY`.
4. **Environment variable `OPENAI_API_BASE`** (read by the underlying SDK at runtime).

```python
from langchain_openai import ChatOpenAI

# Option 1: Environment variable (OPENAI_API_KEY)
model = ChatOpenAI(model="gpt-4.1-mini")

# Option 2: String parameter
model = ChatOpenAI(model="gpt-4.1-mini", api_key="sk-...")

# Option 3: Sync callable
def get_api_key() -> str:
    # Custom logic to retrieve key
    return "sk-..."

model = ChatOpenAI(model="gpt-4.1-mini", api_key=get_api_key)

# Option 4: Async callable
async def get_api_key_async() -> str:
    result = await some_async_fetch()
    return result.key

model = ChatOpenAI(model="gpt-4.1-mini", api_key=get_api_key_async)
```

### Client Initialization

During model validation (`validate_environment`), the integration:
1. Resolves sync and async API keys separately (callable keys may differ between sync/async).
2. Creates `openai.OpenAI` (sync) and `openai.AsyncOpenAI` (async) clients with parameters:
   - `api_key`: The resolved key
   - `base_url`: For proxies or service emulators (defaults to OpenAI's endpoint)
   - `organization`: OpenAI organization ID (from `OPENAI_ORG_ID`)
   - `timeout`: Request timeout (float or `httpx.Timeout`)
   - `max_retries`: Automatic retries on transient failures
   - `default_headers` / `default_query`: Custom request metadata
3. Stores references as `self.client` and `self.async_client`.

### Base URL and Proxy Configuration

**Resolution order:**
1. Explicit `base_url` parameter.
2. Environment variable `OPENAI_API_BASE` (read at init).
3. LangSmith Gateway (if `LANGSMITH_GATEWAY` is set).
4. Environment variable `OPENAI_BASE_URL` (read by openai SDK at runtime).

**Proxy handling:**
- `openai_proxy` parameter: Wraps HTTP clients with proxy configuration.
- Socket options: TCP keepalive and user timeout applied to httpx transports (configurable via `http_socket_options` and environment variables like `LANGCHAIN_OPENAI_TCP_KEEPALIVE`).
- Proxy environment bypass: If proxy env vars are set but no explicit custom transport is configured, the default httpx env-proxy auto-detection is preserved for compatibility.

## Message Conversion: Role and Content Mapping

### LangChain to OpenAI Message Format

The `_convert_message_to_dict()` function transforms LangChain `BaseMessage` objects into OpenAI API format:

```python
def _convert_message_to_dict(message: BaseMessage, api: Literal["chat/completions", "responses"] = "chat/completions") -> dict:
    """Convert LangChain message to OpenAI dict format."""
    message_dict: dict[str, Any] = {
        "content": _format_message_content(message.content, api=api, role=message.type)
    }
    # Role mapping
    if isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
        # Handle tool calls and function calls
        if message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = [_lc_tool_call_to_openai_tool_call(tc) for tc in message.tool_calls]
        elif "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
    elif isinstance(message, SystemMessage):
        # Support "developer" role for o-series models
        message_dict["role"] = message.additional_kwargs.get("__openai_role__", "system")
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id
    elif isinstance(message, FunctionMessage):
        message_dict["role"] = "function"
    else:
        message_dict["role"] = message.role  # ChatMessage
    return message_dict
```

**Role mappings:**
- `HumanMessage` → `"user"`
- `AIMessage` → `"assistant"`
- `SystemMessage` → `"system"` (or `"developer"` for o-series models when explicitly set)
- `ToolMessage` → `"tool"`
- `FunctionMessage` → `"function"` (legacy)
- `ChatMessage` → Custom role (passthrough)

### Content Formatting and Media Handling

The `_format_message_content()` function processes content blocks, handling:

**Text blocks:**
```python
{"type": "text", "text": "..."}
# Optional: {"prompt_cache_breakpoint": {...}}
```

**Image blocks:**
```python
# From content list
{"type": "image_url", "image_url": {"url": "..."}}

# Base64 data URLs
{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
```

Images can be provided as:
- Direct URLs (OpenAI fetches)
- Base64-encoded data URLs (for sensitive or offline images)
- Anthropic-style blocks with `source` field (automatically converted to OpenAI format)

**Tool results in responses:**
```python
{"type": "tool_result", "tool_use_id": "...", "content": "..."}
```

**Special handling:**
- **Prompt cache breakpoints**: Marked with `prompt_cache_breakpoint` key for OpenAI's prompt caching optimization.
- **Developer role for o-series**: System messages carry `__openai_role__="developer"` in `additional_kwargs` and are converted during payload construction.
- **Audio content**: Audio blocks carry an `id` field; when replayed in multi-turn, only the ID is sent (not the data).

### OpenAI to LangChain Message Parsing

The `_convert_dict_to_message()` function reverses the mapping:

```python
def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert OpenAI response dict to LangChain message."""
    role = _dict.get("role")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""), ...)
    elif role == "assistant":
        # Parse tool calls
        tool_calls, invalid_tool_calls = [], []
        for raw_tool_call in _dict.get("tool_calls", []):
            try:
                tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
            except Exception as e:
                invalid_tool_calls.append(make_invalid_tool_call(raw_tool_call, str(e)))
        return AIMessage(
            content=_dict.get("content", "") or "",
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            additional_kwargs={"audio": _dict.get("audio"), ...}
        )
    elif role in ("system", "developer"):
        additional_kwargs = {"__openai_role__": role} if role == "developer" else {}
        return SystemMessage(content=_dict.get("content", ""), additional_kwargs=additional_kwargs, ...)
    # ... ToolMessage, FunctionMessage, etc.
```

**Key behaviors:**
- **Tool call parsing**: Uses `parse_tool_call()` from LangChain core to convert OpenAI's function call dicts into structured `ToolCall` objects with `id`, `name`, and `args`.
- **Invalid tool calls**: If parsing fails, wrapped in `InvalidToolCall` for graceful error handling.
- **Empty content**: Content of `None` from OpenAI (typical for tool calls) is normalized to `""`.
- **Additional kwargs**: Preserves provider-specific fields like audio IDs and developer role markers.

## Vision/Media Support

### Image Input Handling

Vision support is transparent; images are passed as content blocks in messages:

```python
import base64
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# URL-based image
model = ChatOpenAI(model="gpt-4o")
response = model.invoke([
    HumanMessage(content=[
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
    ])
])

# Base64-encoded local image
image_data = base64.b64encode(httpx.get(url).content).decode("utf-8")
response = model.invoke([
    HumanMessage(content=[
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
    ])
])
```

### Image Format and Size Constraints

OpenAI API restrictions:
- **Supported formats**: JPEG, PNG, GIF, WebP
- **Max size**: 20 MB per image
- **Resolution**: Images are automatically rescaled if too large; OpenAI processes up to 768x768 pixels or applies smart cropping
- **Multiple images**: Supported in a single message; vision models automatically analyze all

The integration does not pre-process images; OpenAI's API handles format normalization and sizing.

### Audio Support

The Responses API supports audio inputs and outputs:

```python
model = ChatOpenAI(model="gpt-4o-audio-preview", output_version="responses/v1")

# Single-turn with audio
response = model.invoke("What does this audio say?", audio={"id": "file_abc123"})

# Multi-turn audio
from langchain_core.messages import AIMessage, HumanMessage
response = model.invoke([
    HumanMessage("Listen to this audio"),
    AIMessage(
        content="[Audio response]",
        additional_kwargs={"audio": {"id": "file_xyz789"}}
    ),
    HumanMessage("What was that about?")
])
```

Audio content is represented as `{"type": "audio", "id": "file_..."}` in content blocks and carried in `additional_kwargs["audio"]` for round-trip handling.

## Tool Calling

### Binding Tools and Schema Generation

The `bind_tools()` method binds function schemas to the model:

```python
def bind_tools(
    self,
    tools: Sequence[dict | type | Callable | BaseTool],
    tool_choice: dict | str | bool | None = None,
    strict: bool | None = None,
    parallel_tool_calls: bool | None = None,
    response_format: dict | type | None = None,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, AIMessage]:
```

**Tool format conversion:**
Tools are converted to OpenAI's function schema format via `convert_to_openai_tool()` (from `langchain_core.utils.function_calling`):

```python
from pydantic import BaseModel

class WeatherInput(BaseModel):
    """Input for get_weather tool."""
    location: str
    unit: str = "celsius"

def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the weather at a location."""
    return f"Sunny, 25°{unit}"

model = ChatOpenAI(model="gpt-4.1-mini")
model_with_tools = model.bind_tools([get_weather])

# Or from a Pydantic class directly
model_with_tools = model.bind_tools([WeatherInput])
```

Internally, the tool becomes:
```python
{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather at a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "default": "celsius"}
            },
            "required": ["location"]
        }
    }
}
```

### Tool Choice and Control

The `tool_choice` parameter controls model behavior:

```python
# Auto (default): model decides whether to call a tool
model.bind_tools(tools)

# Force a specific tool
model.bind_tools(tools, tool_choice="get_weather")

# Require at least one tool call
model.bind_tools(tools, tool_choice=True)  # or "required"

# Disable tool calling
model.bind_tools(tools, tool_choice="none")

# Dict format for explicit specification
model.bind_tools(tools, tool_choice={
    "type": "function",
    "function": {"name": "get_weather"}
})

# Well-known built-in tools
model.bind_tools([{"type": "web_search"}], tool_choice="web_search")
```

### Parallel Tool Calls

OpenAI supports parallel tool invocations within a single response:

```python
# Disable parallel (serial execution)
model_with_tools = model.bind_tools(tools, parallel_tool_calls=False)

response = model_with_tools.invoke("What's the weather in NYC and LA?")
# Returns AIMessage with multiple tool_calls in a single message
print(response.tool_calls)  # [ToolCall(...), ToolCall(...)]
```

**Default behavior:** Enabled for most models. Allows the model to call multiple tools at once, useful for answering questions that require multiple function invocations.

### Tool Call Parsing

When the model generates tool calls, they are parsed into structured `ToolCall` objects:

```python
response = model_with_tools.invoke("Get the weather in Boston")
# Returns AIMessage with:
# response.tool_calls = [
#     ToolCall(
#         name="get_weather",
#         args={"location": "Boston", "unit": "celsius"},
#         id="call_xyz123"
#     )
# ]
```

Tool call IDs are used to tie tool execution results back to the invocation:

```python
from langchain_core.messages import ToolMessage

# Execute tool
tool_result = "Sunny, 22°C"

# Create response message
follow_up = model_with_tools.invoke([
    HumanMessage("What's the weather in Boston?"),
    response,  # AIMessage with tool_calls
    ToolMessage(
        content=tool_result,
        tool_call_id="call_xyz123"
    ),
    HumanMessage("Great! What about tomorrow?")
])
```

### Strict Mode

When `strict=True`, the model output is guaranteed to exactly match the schema:

```python
model_with_tools = model.bind_tools(
    [WeatherInput],
    strict=True
)
```

Strict mode:
- Validates the schema at request time.
- OpenAI rejects requests with unsupported schema patterns.
- Model output is validated server-side; parsing errors are API errors.
- Requires Chat Completions API (structured output) or Responses API; does not work with legacy function calling.

### Response Format with Tools

When providing `response_format` along with tools, the Chat Completions API requires all function tools to be strict:

```python
model_with_format = model.bind_tools(
    [WeatherInput],
    response_format={"type": "json_schema", "json_schema": {...}},
    strict=True  # Implicit when response_format is provided
)
```

## Structured Output

The `with_structured_output()` method returns a wrapped model that parses responses into a specified schema:

```python
def with_structured_output(
    self,
    schema: dict | type | None = None,
    method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
    include_raw: bool = False,
    strict: bool | None = None,
    tools: list | None = None,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, dict | Any]:
```

### Method Comparison

**1. JSON Schema (Default, Recommended)**

Uses OpenAI's Structured Output API for guaranteed schema conformance:

```python
from pydantic import BaseModel, Field

class Joke(BaseModel):
    """A joke."""
    setup: str = Field(description="Setup of the joke")
    punchline: str = Field(description="Punchline")
    rating: int = Field(ge=1, le=10, description="Rating 1-10")

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
structured_model = model.with_structured_output(Joke, method="json_schema")

joke = structured_model.invoke("Tell me a joke about cats")
print(joke.setup)  # Returns Pydantic instance
```

**Characteristics:**
- Guarantees schema conformance (OpenAI validates model output).
- Requires a model supporting structured output (GPT-4 Turbo+, GPT-4o, gpt-5.x).
- Supports both Pydantic V2 and JSON schemas.
- Can include `strict=True` for additional validation.
- Supports `tools` parameter to allow tool invocation alongside structured output.

**2. Function Calling**

Uses tool-calling API; schema is treated as a function to "call":

```python
structured_model = model.with_structured_output(
    Joke,
    method="function_calling"
)

joke = structured_model.invoke("Tell me a joke")
```

**Characteristics:**
- Works with older models (GPT-3.5+).
- Less strict; schema is guidance, not a guarantee.
- Uses `bind_tools()` internally, then parses the tool call.
- Supports Pydantic V1 and V2.

**3. JSON Mode**

Tells the model to output valid JSON:

```python
structured_model = model.with_structured_output(
    Joke,
    method="json_mode"
)

joke = structured_model.invoke("Tell me a joke and format as JSON")
```

**Characteristics:**
- Simplest; just ensures output is JSON, no schema enforcement.
- Works with most models.
- Requires explicit prompt instructions to format correctly.
- No guarantee the JSON will match the schema (client-side parsing only).

### Include Raw Output

```python
structured_model = model.with_structured_output(
    Joke,
    method="json_schema",
    include_raw=True
)

result = structured_model.invoke("Tell me a joke")
# result = {
#     "raw": AIMessage(...),
#     "parsed": Joke(...),
#     "parsing_error": None
# }
```

When `include_raw=True`, returns a dict with:
- `"raw"`: The raw `AIMessage` from the model.
- `"parsed"`: The parsed schema instance (or `None` if parsing failed).
- `"parsing_error"`: The exception (if any) during parsing.

### Tools with Structured Output

When using `json_schema` method with `strict=True`, additional tools can be bound:

```python
def get_weather(location: str) -> str:
    """Get weather at location."""
    ...

structured_model = model.with_structured_output(
    Joke,
    method="json_schema",
    strict=True,
    include_raw=True,
    tools=[get_weather]
)

result = structured_model.invoke("Tell a joke or get the weather in Boston")
# If model chooses tool: raw.tool_calls = [ToolCall(...)], parsed = None
# If model chooses structured: raw.tool_calls = [], parsed = Joke(...)
```

## Streaming Implementation

### Stream Method

The `_stream()` method returns an iterator of `ChatGenerationChunk` objects:

```python
for chunk in model.stream("Hello, how are you?"):
    print(chunk.content, end="", flush=True)
```

**Underlying flow:**
1. Sets `stream=True` in the payload.
2. Opens an HTTP connection to OpenAI's streaming endpoint.
3. Iterates over Server-Sent Events (SSE).
4. Converts each delta to a `ChatGenerationChunk`.
5. Dispatches `on_llm_new_token()` callbacks.
6. Yields chunks to the caller.

### Stream Usage

By default, streaming does not include token usage (to avoid an extra event). Enable with:

```python
# Per-invocation
for chunk in model.stream(messages, stream_usage=True):
    ...

# Per-model
model = ChatOpenAI(model="...", stream_usage=True)

# Via model_kwargs
model = ChatOpenAI(model="...", model_kwargs={"stream_options": {"include_usage": True}})
```

When enabled, an additional final chunk with `usage_metadata` is yielded.

### Chunk Timeout (Async Streams)

Async streams have a per-chunk timeout to detect stalled connections:

```python
model = ChatOpenAI(model="...", stream_chunk_timeout=120.0)  # seconds

# Override per invocation
async for chunk in model.astream(messages, stream_chunk_timeout=60.0):
    ...

# Disable with None or 0
model = ChatOpenAI(model="...", stream_chunk_timeout=None)
```

If no chunk arrives within the timeout, a `StreamChunkTimeoutError` is raised with context about the stall.

### Structured Output Streaming

When using structured output with streaming, the final message includes the parsed output:

```python
structured_model = model.with_structured_output(Joke, method="json_schema")

for chunk in structured_model.stream("Tell me a joke"):
    # Text chunks arrive incrementally
    # Final chunk includes parsed Joke instance
    print(chunk)
```

## Decision Patterns and Model-Specific Features

The `/decisions/` submodule provides `OpenAIDecisions`, a specialized runnable for OpenAI's Decisions API. This API answers typed questions about text or images, returning probabilities, confidence scores, and token usage.

### OpenAIDecisions

```python
from langchain_openai.decisions import (
    OpenAIDecisions,
    Predicate,
    Choice,
    Score,
)

decisions = OpenAIDecisions(model="gpt-6-luna")
response = decisions.invoke({
    "input": "Stripe has failed for three days. Help ASAP.",
    "questions": {
        "department": Choice(
            instructions="Which team should handle this?",
            choices={
                "billing": "Payment or subscription issues",
                "technical": "Product bugs or integration failures"
            }
        ),
        "urgent": Predicate(instructions="Is this request urgent?"),
        "frustration": Score(
            instructions="Customer frustration level",
            levels=["Calm", "Concerned", "Angry"]
        )
    }
})

print(response.answers["department"].probabilities)  # {"billing": 0.15, "technical": 0.85}
print(response.answers["urgent"].probability)  # 0.92
print(response.answers["frustration"].score)  # 2 (Angry)
```

Configuration mirrors `ChatOpenAI`:
- `api_key`: Auto-loaded from `OPENAI_API_KEY` or provided explicitly
- `base_url`: Optional proxy or custom endpoint
- `organization`: OpenAI org ID
- `timeout`, `max_retries`: Client behavior tuning

## Disabled Parameters

Some models don't support certain parameters. Use `disabled_params` to prevent them:

```python
model = ChatOpenAI(
    model="old-model",
    disabled_params={"parallel_tool_calls": None}  # Disable entirely
)

# Or allow only specific values
model = ChatOpenAI(
    model="...",
    disabled_params={"parallel_tool_calls": [False]}  # Allow only True
)
```

When a parameter is disabled, `with_structured_output()` and `bind_tools()` skip it silently.

## Error Handling

The integration maps OpenAI exceptions to LangChain's error taxonomy:

```python
from langchain_openai.chat_models.base import (
    OpenAIAuthenticationError,
    OpenAIRateLimitError,
    OpenAIContextOverflowError,
    OpenAIModelNotFoundError,
    OpenAIPermissionDeniedError,
    OpenAIAPIError,
    OpenAIConnectionError,
    OpenAITimeoutError,
)
```

**Mapping table:**
| OpenAI Exception | LangChain Exception | Notes |
|---|---|---|
| `openai.AuthenticationError` | `OpenAIAuthenticationError` (→ `ModelAuthenticationError`) | Invalid/missing API key |
| `openai.RateLimitError` | `OpenAIRateLimitError` (→ `ModelRateLimitError`) | Rate limit exceeded |
| `openai.NotFoundError` | `OpenAIModelNotFoundError` (→ `ModelNotFoundError`) | Model doesn't exist |
| `openai.PermissionDeniedError` | `OpenAIPermissionDeniedError` (→ `ModelPermissionDeniedError`) | Org/billing issue |
| `openai.BadRequestError` (context_length) | `OpenAIContextOverflowError` (→ `ContextOverflowError`) | Prompt too long |
| `openai.BadRequestError` (other) | `OpenAIInvalidRequestError` (→ `ModelInvalidRequestError`) | Invalid schema, etc. |
| `openai.InternalServerError` | `OpenAIAPIError` (→ `ModelAPIError`) | OpenAI server error |
| `openai.APIConnectionError` | `OpenAIConnectionError` (→ `ModelConnectionError`) | Network failure |
| `openai.APITimeoutError` | `OpenAITimeoutError` (→ `ModelTimeoutError`) | Request timeout |

Error handling in `_generate()`:

```python
try:
    response = self.client.create(**payload)
except openai.BadRequestError as e:
    _handle_openai_bad_request(e)  # May raise OpenAIContextOverflowError, etc.
except openai.APIError as e:
    _handle_openai_api_error(e)  # Routes to appropriate LangChain exception
```

## Responses API and Chat Completions API Routing

OpenAI provides two generation APIs:

1. **Chat Completions API** (`/v1/chat/completions`): Traditional chat interface.
2. **Responses API** (`/v1/responses`): Advanced features (reasoning, vision, mid-conversation tools, conversation state).

The model automatically routes requests:

```python
def _use_responses_api(self, payload: dict) -> bool:
    """Determine if Responses API should be used."""
    # Routes based on:
    # 1. Explicit use_responses_api=True/False
    # 2. Model name (gpt-5+, codex models prefer Responses)
    # 3. Parameters like 'reasoning', 'include', 'store', 'truncation'
    # 4. OpenAI profile data
```

**Responses API is used when:**
- Model is GPT-5 series or "codex" series.
- `use_responses_api=True` is set.
- Advanced parameters (`reasoning`, `context_management`, `include`, `truncation`, `store`) are specified.
- Configuration explicitly requires it.

**Chat Completions API is default for:**
- GPT-4, GPT-4o, and older models.
- Basic chat scenarios.

Manual override:

```python
# Force Chat Completions API
model = ChatOpenAI(model="gpt-6", use_responses_api=False)

# Force Responses API
model = ChatOpenAI(model="gpt-3.5-turbo", use_responses_api=True)
```

## Configuration and Operations

### Key Constructor Parameters

```python
model = ChatOpenAI(
    # Model selection
    model="gpt-4.1-mini",  # Required
    
    # API authentication
    api_key="sk-...",  # Or callable; defaults to OPENAI_API_KEY
    base_url="https://api.openai.com/v1",  # Or custom endpoint
    organization="org-...",  # OpenAI org ID
    
    # Generation parameters
    temperature=0.7,  # 0=deterministic, 1=random
    max_tokens=100,  # Max response length
    top_p=1.0,  # Nucleus sampling
    frequency_penalty=0.0,  # Repetition penalty
    presence_penalty=0.0,  # Topic diversity
    seed=42,  # For reproducibility
    
    # Advanced features
    reasoning="medium",  # For Responses API reasoning models
    use_responses_api=True,  # Force Responses API
    output_version="responses/v1",  # Format output
    
    # Streaming
    stream_usage=True,  # Include token counts in stream
    stream_chunk_timeout=120.0,  # Per-chunk timeout (async)
    
    # Transport
    timeout=30.0,  # Request timeout
    max_retries=2,  # Automatic retries
    http_client=custom_httpx_client,  # Custom sync HTTP client
    http_async_client=custom_async_client,  # Custom async HTTP client
    http_socket_options=[(socket.SOL_TCP, socket.TCP_NODELAY, 1)],  # TCP tuning
    
    # Extras
    extra_body={"custom_param": "value"},  # Provider-specific params
    model_kwargs={"max_completion_tokens": 500},  # OpenAI params
)
```

### Prompt Caching

OpenAI's prompt cache reduces costs for repeated prompts:

```python
# Automatic caching (no config needed)
response = model.invoke(messages)  # Cached on second invocation

# Explicit cache key (for multi-tenant scenarios)
response = model.invoke(messages, prompt_cache_key="tenant:123:support-v1")

# Cache options
model = ChatOpenAI(
    model="gpt-5.6-sol",
    prompt_cache_options={
        "mode": "explicit",  # or "implicit"
        "ttl": "30m"
    }
)

# Cache breakpoints (explicit mode)
response = model.invoke([
    SystemMessage([
        {
            "type": "text",
            "text": "System instructions...",
            "prompt_cache_breakpoint": {"mode": "explicit"}
        }
    ]),
    HumanMessage("Current request")
], prompt_cache_options={"mode": "explicit", "ttl": "30m"})
```

Cache hit rates improve with `prompt_cache_key` stability (same key for related requests).

### Token Counting

Get token counts before invocation using `get_num_tokens_from_messages()`:

```python
tokens = model.get_num_tokens_from_messages([
    SystemMessage("You are a helpful assistant."),
    HumanMessage("What is 2+2?")
])
print(f"This will use ~{tokens} tokens")

# With tools
tokens = model.get_num_tokens_from_messages(
    messages,
    tools=[get_weather, search_docs]
)
```

Uses `tiktoken` internally with OpenAI's token encoding for the model.

## Architecture: Request Flow

The request lifecycle:

```
invoke(messages)
  ↓
_generate(messages, stop, run_manager, **kwargs)
  ↓
_get_request_payload(messages, stop, **kwargs)
  ├─ Convert LangChain messages → OpenAI dict format via _convert_message_to_dict()
  ├─ Apply API routing logic (_use_responses_api)
  └─ Construct payload with model params, tools, response_format, etc.
  ↓
Determine endpoint (Chat Completions vs Responses vs Structured)
  ↓
self.client.create(**payload)  [or self.root_client.responses.create]
  ↓
Parse response with _create_chat_result()
  ├─ Extract choices, token_usage, model_name
  ├─ Convert OpenAI message → LangChain AIMessage via _convert_dict_to_message()
  └─ Attach usage_metadata
  ↓
Return ChatResult(generations=[ChatGeneration(message=AIMessage(...), ...)])
```

## Security Considerations

1. **API Key Management**: Use environment variables or async callables; avoid hardcoding keys in code.
2. **SSRF Protection**: The integration uses `ssrf_safe_client` for SSRF mitigation when fetching external images.
3. **Proxy Handling**: Socket options and proxy configuration are transparent; custom httpx clients can be provided for full control.
4. **Credential Redaction**: MCP tool credentials are redacted from logs/tracing.
5. **LangSmith Gateway**: Routes requests through LangSmith when `LANGSMITH_GATEWAY` is set, enabling enterprise security policies.
