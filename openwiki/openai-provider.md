---
type: "Reference"
title: "Use async methods (ainvoke, astream)"
openwiki_generated: true
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:18:34.589Z
sources:
  - id: openwiki-source-1e66a9da38565f8901e651f4
    resource: repo://libs/partners/openai/langchain_openai/__init__.py
  - id: openwiki-source-738512768ef81ae009b097ac
    resource: repo://libs/partners/openai/langchain_openai/chat_models/base.py
  - id: openwiki-source-74e5bef080f1af7da12371cf
    resource: repo://libs/partners/openai/langchain_openai/data/_profiles.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:18:34.589Z" }
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

**Main Class**: `repo://libs/partners/openai/langchain_openai/chat_models/base.py#L2799-L2900`

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

  **Example:**
  ```python
  # Force specific tool
  bound = model.bind_tools([get_weather, get_time], tool_choice="get_weather")
  
  # Force any tool
  bound = model.bind_tools([get_weather, get_time], tool_choice=True)
  
  # Prevent tool use
  bound = model.bind_tools([get_weather, get_time], tool_choice="none")
  ```

- **`parallel_tool_calls`** (`bool | None`): Allow the model to call multiple tools in one response. Default: `None` (allow parallel). Set to `False` to disable.

  ```python
  # Disable parallel tool calls (one at a time)
  bound = model.bind_tools([get_weather, get_time], parallel_tool_calls=False)
  ```

- **`strict`** (`bool | None`): If `True`, model output matches tool schema exactly. Schema is validated per OpenAI's [supported schemas](https://platform.openai.com/docs/guides/structured-outputs/supported-schemas). If `False`, no validation. If `None`, no strict requirement.

- **`response_format`** (`dict | type | None`): Optional response schema for Chat Completions API. When set with tools, requires `strict=True` (exception: Responses API).

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

## Error Handling

`ChatOpenAI` maps OpenAI SDK exceptions to LangChain's standardized error hierarchy:

| OpenAI Exception | LangChain Class | Meaning |
|---|---|---|
| `AuthenticationError` | `ModelAuthenticationError` | Invalid API key |
| `PermissionDeniedError` | `ModelPermissionDeniedError` | API key lacks permissions |
| `BadRequestError` (context_length_exceeded) | `ContextOverflowError` | Input exceeds model's context window |
| `RateLimitError` | `ModelRateLimitError` | Rate limit exceeded |
| `NotFoundError` | `ModelNotFoundError` | Model doesn't exist or isn't available |
| `APIError` / `InternalServerError` | `ModelAPIError` | OpenAI server error |
| `APIConnectionError` | `ModelConnectionError` | Network connectivity issue |
| `APITimeoutError` | `ModelTimeoutError` | Request timeout |

**Example:**

```python
from langchain_core.exceptions import ContextOverflowError, ModelAuthenticationError

try:
    response = model.invoke(very_long_message)
except ContextOverflowError as e:
    print(f"Message too long: {e}")
except ModelAuthenticationError as e:
    print(f"Auth failed: {e}")
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

## Model Name Examples

**Current recommended models:**
- **`gpt-4o`**: Latest, multimodal, fastest (recommended for most use cases)
- **`gpt-4o-mini`**: Lightweight, cheaper variant
- **`gpt-4-turbo`**: Powerful, older than gpt-4o
- **`gpt-4`**: Original GPT-4 (deprecated)
- **`gpt-3.5-turbo`**: Legacy, still cheap (deprecated)

Check [OpenAI models page](https://platform.openai.com/docs/models) for current list.

## Testing

Unit tests are located in `repo://libs/partners/openai/tests/unit_tests/chat_models/`.

Key test files:
- `repo://libs/partners/openai/tests/unit_tests/chat_models/test_base.py`: Main ChatOpenAI tests
- `repo://libs/partners/openai/tests/unit_tests/chat_models/test_client_utils.py`: Client utilities
- `repo://libs/partners/openai/tests/unit_tests/chat_models/test_azure.py`: Azure-specific tests

**Test structured output:**

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class TestSchema(BaseModel):
    name: str
    value: int

def test_with_structured_output():
    model = ChatOpenAI(model="gpt-4o")
    structured = model.with_structured_output(TestSchema, method="function_calling")
    # Invoke and verify output is TestSchema instance
```

## Related Pages

- `/openwiki/model-initialization.md`: Factory function `init_chat_model()` for provider-agnostic model selection
- `/openwiki/chat-models.md`: Core `BaseChatModel` interface and lifecycle
- `/openwiki/messages.md`: Message types and content blocks (text, images, tool calls)
