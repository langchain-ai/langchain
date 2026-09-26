---
type: "ChatModel Integration"
title: "OpenAI Integration: ChatOpenAI and Azure Support"
description: "ChatOpenAI integration for OpenAI's Chat Completions and Responses APIs, with support for tool calling, structured output, vision, streaming, and Azure deployment."
tags: ["openai", "chat-models", "tool-calling", "structured-output", "vision", "azure"]
sources:
  - id: openwiki-source-1e66a9da38565f8901e651f4
    resource: repo://libs/partners/openai/langchain_openai/__init__.py
  - id: openwiki-source-f32b395707eda97cd743f4e5
    resource: repo://libs/partners/openai/langchain_openai/chat_models/azure.py
  - id: openwiki-source-738512768ef81ae009b097ac
    resource: repo://libs/partners/openai/langchain_openai/chat_models/base.py
  - id: openwiki-source-74e5bef080f1af7da12371cf
    resource: repo://libs/partners/openai/langchain_openai/data/_profiles.py
generated: { by: "openwiki/0.5.0", at: "2026-09-21T08:30:16.745Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-09-26T08:25:01.631Z
---

## Overview

The OpenAI integration (`langchain-openai`) provides production-ready chat model support for OpenAI's API and OpenAI-compatible endpoints. `ChatOpenAI` is the primary class that wraps OpenAI's Chat Completions and Responses APIs, with full support for:

- **Chat Completions API** for standard generation and function calling
- **Responses API** for streaming, reasoning models, and enhanced features
- **Structured Output** via tool calling (`json_schema`), JSON mode, or function calling
- **Tool calling** with `bind_tools()` and `tool_choice` parameters
- **Vision** support for gpt-4-vision and gpt-4o models with image inputs
- **Streaming tokens** via callback integration with per-chunk timeouts
- **Model profiles** with capability metadata (input/output modalities, max tokens, tool support)

**Core Principle**: `ChatOpenAI` targets [official OpenAI API specifications](https://github.com/openai/openai-openapi) only. Non-standard response fields added by third-party providers (e.g., `reasoning_content` on vLLM, `reasoning_details` on DeepSeek) are **not** extracted or preserved. For provider-specific features, use the corresponding provider-specific LangChain package (e.g., `ChatDeepSeek`, `ChatOpenRouter`).

## Location

**Package**: `repo://libs/partners/openai/langchain_openai/`

**Main Class**: `repo://libs/partners/openai/langchain_openai/chat_models/base.py#L2829-L3750`

**Exports**: `repo://libs/partners/openai/langchain_openai/__init__.py`

Related classes:
- `BaseChatOpenAI`: Base implementation shared with Azure OpenAI
- `AzureChatOpenAI`: Azure-specific subclass in `repo://libs/partners/openai/langchain_openai/chat_models/azure.py`
- `OpenAI` (legacy): Completion-only model in `repo://libs/partners/openai/langchain_openai/llms/`

## ChatOpenAI Class

### Constructor Parameters

**API Configuration:**

- **`model`** (`str`, default `"gpt-3.5-turbo"`): OpenAI model identifier (e.g., `"gpt-4o"`, `"gpt-4-turbo"`, `"gpt-3.5-turbo"`).
- **`api_key`** (`str | Callable[[], str] | Callable[[], Awaitable[str]] | None`): API key for authentication. Can be:
  - A string value
  - A sync callable that returns a string
  - An async callable that returns a string
  - Inferred from `OPENAI_API_KEY` environment variable if not provided

  **Example:** Callable for dynamic key rotation
  ```python
  def get_api_key() -> str:
      return fetch_from_secrets_manager()
  
  model = ChatOpenAI(api_key=get_api_key)
  ```

- **`base_url`** (`str | None`): Custom API base URL for OpenAI-compatible endpoints. Resolution order (first match wins):
  1. Explicit `base_url` kwarg
  2. Environment variable `OPENAI_API_BASE` (read by LangChain at init)
  3. Environment variable `OPENAI_BASE_URL` (read by the underlying OpenAI SDK)
  
  When set, `stream_usage` is disabled by default since many non-OpenAI endpoints don't support streaming token usage.

- **`organization`** (`str | None`): OpenAI organization ID. Inferred from `OPENAI_ORG_ID` environment variable.

**Generation Parameters:**

- **`temperature`** (`float | None`): Sampling temperature (0–2, typically 0–1). Controls randomness; higher = more random.
- **`max_tokens`** (`int | None`): Maximum tokens to generate in the response.
- **`top_p`** (`float | None`): Nucleus sampling probability. Cumulative probability threshold for token selection.
- **`top_logprobs`** (`int | None`): Number of most-likely tokens to return with log probabilities at each position (requires `logprobs=True`).
- **`logprobs`** (`bool | None`): Whether to return token log probabilities in the response.
- **`seed`** (`int | None`): Deterministic generation seed (if supported by the model).
- **`presence_penalty`** (`float | None`): Penalizes already-mentioned tokens (−2 to 2).
- **`frequency_penalty`** (`float | None`): Penalizes tokens by frequency in the response (−2 to 2).
- **`logit_bias`** (`dict[int, int] | None`): Modify likelihood of specific token IDs appearing.
- **`n`** (`int | None`): Number of completions to generate for each prompt.

**Streaming & Latency:**

- **`streaming`** (`bool`, default `False`): Enable streaming output via `stream()` and `astream()`.
- **`stream_usage`** (`bool | None`): Include token usage metadata in streaming chunks.
  - `None` (default): Enabled for default OpenAI endpoint, disabled when `base_url` is set or custom client provided
  - Set to `True`/`False` to override
- **`stream_chunk_timeout`** (`float | None`, default `120.0`): Per-chunk wall-clock timeout (seconds) for async streaming. Fires on silence between parsed chunks (not affected by OpenAI keepalive SSE comments). Set to `None` or `0` to disable. Overridable via `LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S` environment variable.

**Request Handling:**

- **`timeout`** (`float | tuple[float, float] | None`): Request timeout in seconds or `(connect_timeout, read_timeout)` tuple.
- **`max_retries`** (`int | None`): Maximum retry attempts for transient failures.
- **`http_client`** (`httpx.Client | None`): Custom sync HTTP client. Must be paired with `http_async_client` for async use.
- **`http_async_client`** (`httpx.AsyncClient | None`): Custom async HTTP client.
- **`http_socket_options`** (`Sequence[tuple[int, int, int]] | None`): TCP socket options `(level, option, value)` applied to httpx transports. Defaults to conservative TCP-keepalive + `TCP_USER_TIMEOUT` profile (~2-minute hang bound). Set to `()` (empty) to disable. Overridable via environment variables: `LANGCHAIN_OPENAI_TCP_KEEPALIVE`, `LANGCHAIN_OPENAI_TCP_KEEPIDLE`, `LANGCHAIN_OPENAI_TCP_KEEPINTVL`, `LANGCHAIN_OPENAI_TCP_KEEPCNT`, `LANGCHAIN_OPENAI_TCP_USER_TIMEOUT_MS`.

**Advanced Features:**

- **`reasoning_effort`** (`str | None`): For reasoning models, constrains reasoning effort. Values: `'minimal'`, `'low'`, `'medium'`, `'high'`. (Chat Completions API only.)
- **`reasoning`** (`dict[str, Any] | None`): Reasoning parameters for reasoning models (Responses API only). Shape: `{"effort": None | "low" | "medium" | "high", "summary": "auto" | "concise" | "detailed"}`.
- **`verbosity`** (`str | None`): Verbosity level for reasoning models (Responses API). Values: `'low'`, `'medium'`, `'high'`.
- **`service_tier`** (`str | None`): Latency tier for requests. Options: `'auto'`, `'default'`, `'flex'`. For users of OpenAI's scale tier service.
- **`store`** (`bool | None`): Whether OpenAI may store response data. Defaults to `True` for Responses API, `False` for Chat Completions API.
- **`include_response_headers`** (`bool`, default `False`): Capture response headers in message `response_metadata`. Useful for capturing provider metadata (e.g., served model names from inference providers).
- **`extra_body`** (`dict[str, Any] | None`): Additional JSON properties for OpenAI-compatible APIs (vLLM, LM Studio, etc.). Recommended over `model_kwargs` for provider-specific parameters.
- **`prompt_cache_options`** (`dict[str, Any] | None`): Configuration for OpenAI prompt caching.
- **`include`** (`list[str] | None`): Additional fields to include in generations from Responses API. Examples: `'file_search_call.results'`, `'message.input_image.image_url'`, `'reasoning.encrypted_content'`.
- **`truncation`** (`str | None`): Truncation strategy for Responses API. `'auto'` (drop middle items) or `'disabled'` (default).
- **`context_management`** (`list[dict[str, Any]] | None`): Configuration for [context compaction](https://developers.openai.com/api/docs/guides/compaction).
- **`disabled_params`** (`dict[str, Any] | None`): Parameters to disable for the model. Shape: `{"param": None | ['val1', 'val2']}`. Used to prevent incompatible parameters (e.g., `{"parallel_tool_calls": None}` for older models).

**Other:**

- **`stop`** (`list[str] | str | None`): Default stop sequences.
- **`tiktoken_model_name`** (`str | None`): Model name for tiktoken token counting (if different from `model`).
- **`model_kwargs`** (`dict[str, Any]`): Additional parameters passed to the API (overridden by `extra_body` for provider-specific params).
- **`default_headers`** (`dict[str, str] | None`): Custom HTTP headers for requests.
- **`default_query`** (`dict[str, object] | None`): Custom query parameters.

### Initialization Examples

**Basic Usage (API key from environment):**

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")
response = model.invoke("What is 2 + 2?")
```

**Custom API Base (OpenAI-compatible endpoint):**

```python
model = ChatOpenAI(
    model="gpt-4-turbo",
    base_url="https://api.custom-openai-provider.com/v1",
    api_key="your-custom-api-key"
)
```

**With Streaming and Timeout:**

```python
model = ChatOpenAI(
    model="gpt-4o",
    streaming=True,
    timeout=30.0,
    stream_chunk_timeout=60.0
)

for chunk in model.stream("Hello, what is your name?"):
    print(chunk.content, end="", flush=True)
```

**Dynamic API Key:**

```python
async def get_api_key() -> str:
    return await fetch_from_secret_store()

model = ChatOpenAI(
    model="gpt-4o",
    api_key=get_api_key
)

# Use async methods (ainvoke, astream)
response = await model.ainvoke("Hi")
```

## BaseChatOpenAI and Initialization

`ChatOpenAI` inherits from `BaseChatOpenAI`, which is a base class shared with `AzureChatOpenAI`. On initialization, `BaseChatOpenAI`:

1. **Resolves API authentication** from parameters, environment variables, or callables
2. **Builds HTTP clients** (sync and async) with optional socket options for connection management
3. **Registers model profiles** for capability metadata
4. **Validates parameters** like `stream_chunk_timeout` (negative values fall back to defaults with warnings)
5. **Initializes OpenAI client instances** (`self.client`, `self.async_client`) using the OpenAI SDK

**Client Initialization Details:**

- **Sync client** (`self.client`): Built from sync `httpx.Client` or created internally. Required for sync `invoke()` and `stream()` methods.
- **Async client** (`self.async_client`): Built from async `httpx.AsyncClient` or created internally. Required for async `ainvoke()` and `astream()` methods.
- **Root clients** (`self.root_client`, `self.root_async_client`): Cached OpenAI client instances used for actual API calls.

If an **async callable** is provided for `api_key`, the sync client is not available, and sync methods raise `ValueError`. Use async methods instead:

```python
async def get_key() -> str:
    return await fetch_secret()

model = ChatOpenAI(model="gpt-4o", api_key=get_key)
# await model.ainvoke(...) works
# model.invoke(...) raises ValueError
```

## Model Profiles and Capabilities

Model profiles are auto-generated metadata that describe model capabilities. They are stored in `repo://libs/partners/openai/langchain_openai/data/_profiles.py` and retrieved via the `ModelProfileRegistry`.

**Profile Fields:**
- **`text_inputs` / `text_outputs`**: Text support.
- **`image_inputs`**: Vision support (gpt-4o, gpt-4-vision, gpt-4-turbo with vision).
- **`audio_inputs` / `audio_outputs`**: Audio support (gpt-4o, upcoming models).
- **`video_inputs`**: Video support (upcoming).
- **`tool_calling`**: Whether the model supports function/tool calling.
- **`structured_output`**: Whether the model supports JSON Schema structured output.
- **`max_input_tokens` / `max_output_tokens`**: Context window and generation limits.
- **`tool_call_streaming`**: Whether tool calls stream incrementally.
- **`tool_choice`**: Whether tool_choice parameter is supported.

**Accessing Profiles:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.language_models import ModelProfileRegistry

model = ChatOpenAI(model="gpt-4o")
# Profiles are used internally by LangChain for capability checks
```

## Responses API

`ChatOpenAI` automatically switches between the Chat Completions API and the Responses API based on the model, parameters, and configuration. The **Responses API** provides enhanced features including:

- **Streaming reasoning** for reasoning models (e.g., o1-preview)
- **Structured output with tools** alongside reasoning
- **Context management** (message compaction) via `context_management` parameter
- **Truncation strategy** control via `truncation` parameter
- **Reasoning parameters** (effort, summary) via `reasoning` dict
- **Previous response tracking** via `use_previous_response_id` parameter

**Automatic API Selection**: The Responses API is automatically used when:
- Model name starts with `gpt-5` (pro variants) or contains `codex`
- `use_responses_api=True` is explicitly set
- `reasoning` or `context_management` parameters are provided
- `truncation` or `include` parameters are set
- `use_previous_response_id=True` is set
- Model name starts with `gpt-6` and tools are provided

**Explicit Control:**

```python
# Force Responses API
model = ChatOpenAI(model="gpt-4o", use_responses_api=True)

# Force Chat Completions API
model = ChatOpenAI(model="gpt-4o", use_responses_api=False)

# Auto-detect (default)
model = ChatOpenAI(model="gpt-4o", use_responses_api=None)
```

**Responses API with Reasoning:**

```python
model = ChatOpenAI(
    model="o1-preview",
    use_responses_api=True,
    reasoning={
        "effort": "high",
        "summary": "detailed"
    }
)

response = model.invoke("Analyze this complex system design")
# Response includes reasoning content and analysis
```

**Context Management (Responses API only):**

```python
model = ChatOpenAI(
    model="gpt-4o",
    use_responses_api=True,
    context_management=[
        {"type": "auto", "min_tokens": 1000}
    ]
)
# Model will automatically drop older messages to fit context window
```

## Vision Support

Vision is supported on models like `gpt-4-vision`, `gpt-4o`, and `gpt-4-turbo`. Images can be provided as:

1. **URL-based (`image_url`):**
   ```python
   from langchain_core.messages import HumanMessage
   
   message = HumanMessage(
       content=[
           {"type": "text", "text": "What's in this image?"},
           {
               "type": "image_url",
               "image_url": {
                   "url": "https://example.com/image.jpg",
                   "detail": "low"  # or "high", "auto"
               }
           }
       ]
   )
   
   model = ChatOpenAI(model="gpt-4o")
   response = model.invoke(message)
   ```

2. **Base64-encoded:**
   ```python
   import base64
   
   with open("image.jpg", "rb") as f:
       image_data = base64.b64encode(f.read()).decode("utf-8")
   
   message = HumanMessage(
       content=[
           {"type": "text", "text": "Describe this image"},
           {
               "type": "image_url",
               "image_url": {
                   "url": f"data:image/jpeg;base64,{image_data}",
                   "detail": "auto"
               }
           }
       ]
   )
   ```

Token counting for images is approximated: `low` detail = 85 tokens, `high` detail = ~170 + 255 per image tile based on resolution.

## Function Calling

OpenAI's [function calling API](https://platform.openai.com/docs/guides/function-calling) (now called "tools" in the API) allows models to call functions you define.

### `bind_tools()` Method

Bind one or more tools to the model:

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Sunny in {location}"

model = ChatOpenAI(model="gpt-4o")
bound_model = model.bind_tools([get_weather])

response = bound_model.invoke("What's the weather in Boston?")
print(response.tool_calls)
# [ToolCall(id='call_123', name='get_weather', args={'location': 'Boston'}, type='tool_call')]
```

**`bind_tools()` Signature:**

```python
def bind_tools(
    self,
    tools: Sequence[dict | type | Callable | BaseTool],
    *,
    tool_choice: dict | str | bool | None = None,
    strict: bool | None = None,
    parallel_tool_calls: bool | None = None,
    response_format: dict | type | None = None,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, AIMessage]
```

**Parameters:**

- **`tools`**: List of tools. Supports:
  - `BaseTool` instances (from `@tool` decorator)
  - Pydantic `BaseModel` classes
  - Callables with type hints
  - Dicts (OpenAI tool schema)

- **`tool_choice`** (`dict | str | bool | None`): Which tool(s) to force:
  - `str` (tool name): Forces that specific tool (e.g., `"get_weather"`)
  - `'auto'`: Auto-select tool or none (default)
  - `'none'`: Prevent tool calling
  - `'any'` / `'required'` / `True`: Force at least one tool call
  - `dict`: OpenAI tool choice dict `{"type": "function", "function": {"name": "tool_name"}}`
  - `False` / `None`: No effect, default behavior
  - `WellKnownTools` strings (`'file_search'`, `'web_search'`, `'tool_search'`, etc.): Built-in tools

  **Example:**
  ```python
  # Force specific tool
  bound = model.bind_tools([get_weather, get_time], tool_choice="get_weather")
  
  # Force any tool
  bound = model.bind_tools([get_weather, get_time], tool_choice=True)
  
  # Prevent tool use
  bound = model.bind_tools([get_weather, get_time], tool_choice="none")
  
  # Allow web search alongside tools
  bound = model.bind_tools([get_weather], tool_choice="web_search")
  ```

- **`parallel_tool_calls`** (`bool | None`): Allow the model to call multiple tools in one response. Default: `None` (allow parallel). Set to `False` to disable.

  ```python
  # Disable parallel tool calls (one at a time)
  bound = model.bind_tools([get_weather, get_time], parallel_tool_calls=False)
  ```

- **`strict`** (`bool | None`): If `True`, model output matches tool schema exactly. Schema is validated per OpenAI's [supported schemas](https://platform.openai.com/docs/guides/structured-outputs/supported-schemas). If `False`, no validation. If `None`, no strict requirement. When `response_format` is provided via Chat Completions API, strict defaults to `True` unless explicitly set to `False`.

- **`response_format`** (`dict | type | None`): Optional response schema for Chat Completions API. When set with tools, requires `strict=True` (exception: Responses API does not require this).

### Tool Call Processing

When a model calls tools, the response includes `AIMessage.tool_calls`:

```python
response = bound_model.invoke("What's the weather in Boston and New York?")

# response.tool_calls:
# [
#   ToolCall(id='call_1', name='get_weather', args={'location': 'Boston'}),
#   ToolCall(id='call_2', name='get_weather', args={'location': 'New York'})
# ]
```

**Process tool calls in an agentic loop:**

```python
from langchain_core.messages import ToolMessage

messages = [HumanMessage("What's the weather in Boston?")]

while True:
    response = model.invoke(messages)
    
    if not response.tool_calls:
        print("Final response:", response.content)
        break
    
    messages.append(response)
    
    for tool_call in response.tool_calls:
        tool_result = get_weather(location=tool_call.args["location"])
        messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call.id))
```

## Structured Output

The `with_structured_output()` method constrains model outputs to a specific schema. Three methods are available:

### Method: `'function_calling'` (Default)

Uses OpenAI's [tool-calling API](https://platform.openai.com/docs/guides/function-calling). The model must call a specific tool with arguments matching the schema.

**Pros**: Supported on most models (gpt-3.5-turbo, gpt-4, etc.).

**Cons**: Requires tool calling support. Less strict than `json_schema`.

**Usage:**

```python
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

class Joke(BaseModel):
    setup: str
    punchline: str

model = ChatOpenAI(model="gpt-4o")
structured = model.with_structured_output(Joke, method="function_calling")

result = structured.invoke("Tell me a joke")
print(result)
# Joke(setup='...', punchline='...')
```

### Method: `'json_schema'`

Uses OpenAI's [Structured Output API](https://platform.openai.com/docs/guides/structured-outputs). The model generates JSON strictly matching the schema.

**Pros**: Guaranteed strict output conformance. Supported on modern models (gpt-4o-2024-08-06+, gpt-4-turbo-2024-04-09+).

**Cons**: Only for models with `structured_output=True` in profile. Requires valid JSON Schema.

**Usage:**

```python
structured = model.with_structured_output(
    Joke, 
    method="json_schema",
    strict=True  # Validate schema and output
)

result = structured.invoke("Tell me a joke")
print(result)  # Pydantic instance if schema is BaseModel, else dict
```

### Method: `'json_mode'`

Uses OpenAI's [JSON mode](https://platform.openai.com/docs/guides/structured-outputs/json-mode). The model generates JSON but without strict schema validation.

**Pros**: Works on more models. Simpler than `json_schema`.

**Cons**: Output may not strictly match schema. Manual prompt engineering required.

**Usage:**

```python
structured = model.with_structured_output(
    Joke,
    method="json_mode"
)

# Must include instructions in your prompt
result = structured.invoke(
    "Tell me a joke. Return as JSON: {setup: ..., punchline: ...}"
)
```

### Common Parameters

```python
def with_structured_output(
    self,
    schema: dict | BaseModel | type | None = None,
    *,
    method: Literal["function_calling", "json_mode", "json_schema"] = "function_calling",
    include_raw: bool = False,
    strict: bool | None = None,
    tools: list | None = None,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, dict | BaseModel]
```

- **`schema`**: Output schema. Accepts:
  - Pydantic `BaseModel` (output is instance of this class)
  - JSON Schema dict
  - `TypedDict`
  - OpenAI tool schema dict

- **`method`**: Approach for constraining output. Defaults to `"function_calling"`. Override incompatible methods:
  ```python
  # For older models, auto-downgrade json_schema to function_calling
  structured = model.with_structured_output(
      Joke,
      method="json_schema"  # Auto-downgrades to function_calling if model doesn't support it
  )
  ```

- **`include_raw`** (`bool`, default `False`): Return both raw model response and parsed output in a dict:
  ```python
  structured = model.with_structured_output(
      Joke,
      include_raw=True
  )
  
  result = structured.invoke("Tell me a joke")
  # {
  #   'raw': AIMessage(...),
  #   'parsed': Joke(...),
  #   'parsing_error': None
  # }
  ```

  If parsing fails, `parsed` is `None` and `parsing_error` is the exception.

- **`strict`** (`bool | None`): Validate schema and enforce exact output matching. Default: `None` (not enforced). Only applies to `json_schema` and `function_calling` methods.

- **`tools`** (`list | None`): Additional tools the model can call (alongside structured output). Requires:
  - `method="json_schema"`
  - `strict=True`
  - `include_raw=True`
  
  When the model calls a tool instead of generating structured output:
  ```python
  structured = model.with_structured_output(
      ResponseSchema,
      method="json_schema",
      tools=[get_weather, search_web],
      strict=True,
      include_raw=True
  )
  
  result = structured.invoke("Should I bring an umbrella to Boston?")
  # {
  #   'raw': AIMessage(tool_calls=[ToolCall(name='get_weather', ...)]),
  #   'parsed': None,
  #   'parsing_error': None
  # }
  ```

## Streaming and Callbacks

### Basic Streaming

```python
model = ChatOpenAI(model="gpt-4o", streaming=True)

for chunk in model.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

### Token Callback Integration

Streaming callbacks fire on each chunk via `run_manager.on_llm_new_token()`:

```python
from langchain_core.callbacks import StreamingStdOutCallbackHandler

model = ChatOpenAI(model="gpt-4o", streaming=True)

# Callbacks are invoked during stream
for chunk in model.stream(
    "Hello",
    config={"callbacks": [StreamingStdOutCallbackHandler()]}
):
    pass  # Callback prints tokens as they arrive
```

**Custom Streaming Callback:**

```python
from langchain_core.callbacks import BaseCallbackHandler

class CustomTokenCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"[TOKEN] {token}")

model = ChatOpenAI(model="gpt-4o", streaming=True)
model.invoke(
    "Hi",
    config={"callbacks": [CustomTokenCallback()]}
)
```

### Async Streaming with Chunk Timeout

Async streaming operations apply `stream_chunk_timeout` (default 120s):

```python
async def stream_response():
    model = ChatOpenAI(
        model="gpt-4o",
        streaming=True,
        stream_chunk_timeout=30.0  # 30-second per-chunk timeout
    )
    
    async for chunk in model.astream("Tell me a long story"):
        print(chunk.content, end="", flush=True)

import asyncio
asyncio.run(stream_response())
```

If a chunk doesn't arrive within the timeout, `StreamChunkTimeoutError` is raised. This is distinct from `httpx` read timeout—it measures silence between *parsed chunks*, not inter-byte silence.

## Error Handling and Retries

`ChatOpenAI` maps OpenAI SDK exceptions to LangChain's standardized error hierarchy:

| OpenAI Exception | LangChain Class | Meaning |
|---|---|---|
| `AuthenticationError` | `ModelAuthenticationError` | Invalid API key |
| `PermissionDeniedError` | `ModelPermissionDeniedError` | API key lacks permissions |
| `BadRequestError` (context_length_exceeded) | `ContextOverflowError` | Input exceeds model's context window |
| `BadRequestError` (response_format validation) | `ModelInvalidRequestError` | Invalid schema for structured output |
| `RateLimitError` | `ModelRateLimitError` | Rate limit exceeded |
| `NotFoundError` | `ModelNotFoundError` | Model doesn't exist or isn't available |
| `APIError` / `InternalServerError` | `ModelAPIError` | OpenAI server error |
| `APIConnectionError` | `ModelConnectionError` | Network connectivity issue |
| `APITimeoutError` | `ModelTimeoutError` | Request timeout |

**Error Handling Example:**

```python
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAuthenticationError,
    ModelRateLimitError,
    ModelTimeoutError,
)

model = ChatOpenAI(model="gpt-4o")

try:
    response = model.invoke(messages)
except ContextOverflowError as e:
    print(f"Message too long: {e}")
except ModelAuthenticationError as e:
    print(f"Auth failed: {e}")
except ModelRateLimitError as e:
    print(f"Rate limited, retry later")
except ModelTimeoutError as e:
    print(f"Request timed out")
```

**Retry Configuration:**

Automatic retries for transient failures are configured via `max_retries` (default: None). The OpenAI SDK automatically retries on certain transient errors (429, 500-599 status codes):

```python
model = ChatOpenAI(
    model="gpt-4o",
    max_retries=3,  # Retry up to 3 times on transient failures
    timeout=30.0   # Request timeout in seconds
)

# Or with tuple for separate connect/read timeouts
model = ChatOpenAI(
    model="gpt-4o",
    timeout=(10.0, 30.0)  # (connect_timeout, read_timeout)
)
```

**Stream Chunk Timeout (Async Streaming):**

When async streaming stalls between parsed chunks (not keepalive), a `StreamChunkTimeoutError` is raised:

```python
from langchain_openai import StreamChunkTimeoutError

model = ChatOpenAI(
    model="gpt-4o",
    stream_chunk_timeout=60.0  # Timeout per chunk
)

try:
    async for chunk in model.astream("Hello"):
        print(chunk.content, end="")
except StreamChunkTimeoutError as e:
    print(f"Stream stalled: {e}")
```

## Advanced Configuration

### Proxy and Network

```python
# Explicit proxy
model = ChatOpenAI(
    model="gpt-4o",
    openai_proxy="http://proxy.example.com:8080"
)

# Or via environment: OPENAI_PROXY=...
```

### Custom HTTP Client

```python
import httpx

http_client = httpx.Client(
    timeout=30.0,
    limits=httpx.Limits(max_connections=10)
)

model = ChatOpenAI(
    model="gpt-4o",
    http_client=http_client
)
```

### Prompt Caching

```python
# Cache long system prompts or large context
model = ChatOpenAI(
    model="gpt-4o",
    prompt_cache_options={
        "type": "ephemeral"
    }
)
```

### Logit Bias

```python
# Encourage specific tokens
model = ChatOpenAI(
    model="gpt-4o",
    logit_bias={
        20: 50,    # Boost token ID 20
        100: -100  # Suppress token ID 100
    }
)
```

## Message Handling and Generation

### Message Conversion

`ChatOpenAI` converts LangChain message types to OpenAI's API format and back:

**Input message types** (converted to OpenAI format):
- `HumanMessage`: user role
- `AIMessage`: assistant role (with tool_calls and additional_kwargs)
- `SystemMessage`: system role (or "developer" if marked with `__openai_role__`)
- `ToolMessage`: tool role (with tool_call_id)
- `FunctionMessage`: function role (legacy)

**Output**: `AIMessage` with:
- `content`: Text response
- `tool_calls`: List of `ToolCall` objects if model called tools
- `invalid_tool_calls`: Malformed tool calls that couldn't be parsed
- `additional_kwargs`: Audio data (if audio output enabled), function_call (legacy), etc.
- `response_metadata`: token usage, finish reason, system fingerprint, logprobs, etc.
- `usage_metadata`: Standardized usage counts (input_tokens, output_tokens, total_tokens)

### Generation Flow

1. **Input normalization**: Convert string or message list to `ChatPromptValue`
2. **Message formatting**: Format content blocks (text, images, tool use markers) per API requirements
3. **Payload construction**: Build request dict with model, messages, parameters, tools, response_format, etc.
4. **API selection**: Determine Chat Completions vs Responses API based on model and parameters
5. **API call**: Invoke OpenAI SDK (sync or async)
6. **Response parsing**: Extract message content, tool calls, usage, metadata
7. **Message creation**: Wrap in `AIMessage` with all metadata
8. **Callback firing**: Invoke LLM callbacks for logging, streaming, etc.

### Content Block Handling

When messages contain multi-modal content (text + images, text + tool references), `ChatOpenAI` formats them per API requirements:

```python
from langchain_core.messages import HumanMessage

# Multi-modal message
message = HumanMessage(
    content=[
        {"type": "text", "text": "Analyze this chart"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/chart.png", "detail": "high"}
        }
    ]
)

response = model.invoke([message])
```

For **Chat Completions API**, certain content block types are filtered (e.g., `thinking`, `tool_use`).
For **Responses API**, content blocks are expanded to support reasoning, computer use, file search, etc.

## Azure OpenAI Integration

`AzureChatOpenAI` is a specialized subclass for Azure OpenAI deployments. It inherits all `ChatOpenAI` functionality (tool calling, structured output, streaming, vision) but with Azure-specific authentication, endpoint routing, and response metadata handling.

### Azure Setup

First, create an Azure OpenAI deployment using the [quickstart guide](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/chatgpt-quickstart).

Install the package and set environment variables:

```bash
pip install -U langchain-openai

export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
export OPENAI_API_VERSION="2024-05-01-preview"  # Optional; can be passed to constructor
```

### Basic Usage

```python
from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    azure_deployment="my-deployment",
    api_version="2024-05-01-preview",
    temperature=0,
    max_tokens=None,
)

response = model.invoke("What is 2 + 2?")
print(response.usage_metadata)  # Token counts
```

### Key Azure Parameters

**Authentication & Endpoint:**

- **`azure_deployment`** (`str`): Name of the Azure OpenAI deployment. Sets the request URL to `/deployments/{azure_deployment}`.
- **`azure_endpoint`** (`str`): Full Azure endpoint URL (e.g., `https://resource-name.openai.azure.com/`). Auto-inferred from `AZURE_OPENAI_ENDPOINT` env var.
- **`api_key`** (`str | Callable`): Azure API key. Auto-inferred from `AZURE_OPENAI_API_KEY` env var.
- **`azure_ad_token`** (`str`): Azure Active Directory token (alternative to API key).
- **`api_version`** (`str`): Azure OpenAI REST API version (distinct from model version). Examples: `"2024-05-01-preview"`, `"2024-02-15-preview"`. See [API versions](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning).

**Model Configuration (for tracing & token counting only):**

- **`model`** (`str`): Underlying OpenAI model name (e.g., `"gpt-4o"`, `"gpt-35-turbo"`). Does **not** affect completion; uses `azure_deployment` instead.
- **`model_version`** (`str`): Model version (e.g., `'0125'`, `'0125-preview'`) for token counting.

**Other Parameters:**

All standard `ChatOpenAI` parameters are supported: `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, `timeout`, `max_retries`, `streaming`, `logprobs`, etc.

### Azure Response Example

Azure includes additional metadata in responses:

```python
model = AzureChatOpenAI(azure_deployment="my-deployment", api_version="2024-05-01-preview")
response = model.invoke("Translate to French: Hello")

print(response.usage_metadata)
# {'input_tokens': 28, 'output_tokens': 6, 'total_tokens': 34}

print(response.response_metadata)
# {
#   'token_usage': {
#     'completion_tokens': 6, 'prompt_tokens': 28, 'total_tokens': 34
#   },
#   'model_name': 'gpt-4o',
#   'system_fingerprint': 'fp_...',
#   'prompt_filter_results': [...],      # Content safety filtering
#   'content_filter_results': {...},     # Safety categorization
#   'finish_reason': 'stop',
# }
```

**Content Safety Filtering**: Azure includes `prompt_filter_results` and `content_filter_results` in `response_metadata`, detailing filtering for hate speech, self-harm, sexual content, and violence.

### Azure Tool Calling

Tool calling with `AzureChatOpenAI` works identically to `ChatOpenAI`:

```python
from pydantic import BaseModel, Field

class GetWeather(BaseModel):
    '''Get current weather'''
    location: str = Field(description="City and state, e.g. Boston, MA")

model = AzureChatOpenAI(azure_deployment="my-deployment", api_version="2024-05-01-preview")
model_with_tools = model.bind_tools([GetWeather])
response = model_with_tools.invoke("What's the weather in Boston?")
print(response.tool_calls)
```

### Azure Streaming

Streaming with `AzureChatOpenAI` includes all standard features (callbacks, chunk timeouts, token usage in chunks):

```python
model = AzureChatOpenAI(
    azure_deployment="my-deployment",
    api_version="2024-05-01-preview",
    streaming=True,
    stream_chunk_timeout=60.0
)

for chunk in model.stream("Translate to French: Hello"):
    print(chunk.content, end="")
```

### Azure Structured Output

All `with_structured_output()` methods are supported:

```python
from pydantic import BaseModel

class Translation(BaseModel):
    french: str
    confidence: float

model = AzureChatOpenAI(azure_deployment="my-deployment", api_version="2024-05-01-preview")
structured = model.with_structured_output(Translation, method="json_schema")
result = structured.invoke("Translate to French: Hello world")
print(result.french)
```

## Model Name Examples

**Current recommended models:**
- **`gpt-4o`**: Latest, multimodal, fastest (recommended for most use cases)
- **`gpt-4o-mini`**: Lightweight, cheaper variant
- **`gpt-4-turbo`**: Powerful, older than gpt-4o
- **`gpt-4`**: Original GPT-4 (deprecated)
- **`gpt-3.5-turbo`**: Legacy, still cheap (deprecated)

Check [OpenAI models page](https://platform.openai.com/docs/models) for current list.

## Testing

Unit and integration tests are located in `repo://libs/partners/openai/tests/`.

### Unit Tests

Key unit test files:
- `repo://libs/partners/openai/tests/unit_tests/chat_models/test_base.py`: Main ChatOpenAI tests including:
  - API initialization and parameter validation
  - Message conversion and content block handling
  - Error handling and exception mapping
  - Tool calling and structured output methods
  - Streaming with callbacks
- `repo://libs/partners/openai/tests/unit_tests/chat_models/test_base_standard.py`: Standard test suite for ChatOpenAI (Chat Completions API)
- `repo://libs/partners/openai/tests/unit_tests/chat_models/test_responses_standard.py`: Standard test suite for Responses API
- `repo://libs/partners/openai/tests/unit_tests/chat_models/test_azure.py`: Azure-specific tests
- `repo://libs/partners/openai/tests/unit_tests/chat_models/test_client_utils.py`: Client utilities (socket options, proxies, HTTP clients)

### Integration Tests

Integration tests with real API calls are in `repo://libs/partners/openai/tests/integration_tests/chat_models/`.

### Standard Test Suite

Both `ChatOpenAI` (Chat Completions) and Responses API inherit standard test suites from `langchain-tests` to validate:
- Basic invoke and streaming
- Tool calling semantics
- Structured output conformance
- Callback integration
- Token counting accuracy

**Example unit test:**

```python
import pytest
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class TestSchema(BaseModel):
    name: str
    value: int

@pytest.mark.asyncio
async def test_structured_output_function_calling():
    model = ChatOpenAI(model="gpt-4o")
    structured = model.with_structured_output(TestSchema, method="function_calling")
    result = await structured.ainvoke("Return {name: 'test', value: 42}")
    assert isinstance(result, TestSchema)
    assert result.name == "test"
    assert result.value == 42

@pytest.mark.asyncio
async def test_streaming_with_callback():
    from langchain_core.callbacks import StreamingStdOutCallbackHandler
    
    model = ChatOpenAI(model="gpt-4o", streaming=True)
    chunks = []
    async for chunk in model.astream("Hello", config={"callbacks": []}):
        chunks.append(chunk)
    assert len(chunks) > 0
```

## Extension and Customization

### Subclassing BaseChatOpenAI

Advanced use cases can subclass `BaseChatOpenAI` to customize behavior:

```python
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.outputs import ChatResult

class CustomChatOpenAI(BaseChatOpenAI):
    """Custom OpenAI wrapper with additional logging."""
    
    custom_param: str = "default"
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Custom pre-processing
        print(f"Custom param: {self.custom_param}")
        
        # Call parent
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        
        # Custom post-processing
        result.llm_output["custom_field"] = "custom_value"
        
        return result

# Use custom class
model = CustomChatOpenAI(model="gpt-4o", custom_param="my_value")
response = model.invoke("Hello")
```

### Middleware and Hooks

Custom middleware can be added via `RunnablePassthrough`, `RunnableLambda`, or decorator patterns:

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def log_input(input_val):
    print(f"User input: {input_val}")
    return input_val

def log_output(output):
    print(f"Model output: {output.content}")
    return output

model = ChatOpenAI(model="gpt-4o")
chain = (
    RunnableLambda(log_input)
    | model
    | RunnableLambda(log_output)
)

response = chain.invoke("What is 2+2?")
```

### Custom Client Configuration

For advanced network control, provide fully configured httpx clients:

```python
import httpx
from langchain_openai import ChatOpenAI

http_client = httpx.Client(
    timeout=httpx.Timeout(30.0),
    limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
    verify=certifi.where(),
)

http_async_client = httpx.AsyncClient(
    timeout=httpx.Timeout(30.0),
    limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
)

model = ChatOpenAI(
    model="gpt-4o",
    http_client=http_client,
    http_async_client=http_async_client,
)
```

## Known Limitations and Considerations

1. **Sync callable API keys**: If `api_key` is a sync callable, async methods still work, but they resolve the key in an executor thread.
2. **Provider-specific fields**: Non-OpenAI fields in responses (e.g., from vLLM, DeepSeek) are not preserved. Use provider-specific packages instead.
3. **Responses API limitations**: Not all Chat Completions parameters are supported in Responses API (e.g., `n` is not supported).
4. **Structured output schema validation**: The `json_schema` method requires schemas to meet OpenAI's supported-schemas constraints.
5. **Azure API version coupling**: Azure requires explicit `api_version` and ties it to feature availability (e.g., structured output only in newer versions).

## Related Pages

- `/openwiki/model-initialization.md`: Factory function `init_chat_model()` for provider-agnostic model selection
- `/openwiki/chat-models.md`: Core `BaseChatModel` interface and lifecycle
- `/openwiki/messages.md`: Message types and content blocks (text, images, tool calls)
