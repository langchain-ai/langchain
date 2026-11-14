---
type: "API Reference"
title: "ChatOpenAI API Reference"
description: "Reference documentation for ChatOpenAI and BaseChatOpenAI class signatures, configuration parameters, and core behaviors for OpenAI chat model integration."
tags: ["openai", "chat-models", "api-reference", "langchain-openai"]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-13T08:16:35.902Z
sources:
  - id: openwiki-source-738512768ef81ae009b097ac
    resource: repo://libs/partners/openai/langchain_openai/chat_models/base.py
generated: { by: "openwiki/0.5.0", at: "2026-11-13T08:16:35.902Z" }
---

# ChatOpenAI API Reference

This reference documents the core OpenAI chat model integration provided by `langchain-openai`, specifically the `ChatOpenAI` class and its configuration parameters.

## Overview

`ChatOpenAI` provides a LangChain wrapper around OpenAI's Chat Completions and Responses APIs. It targets [official OpenAI API specifications](https://github.com/openai/openai-openapi) and does not extract non-standard response fields added by third-party providers. For OpenAI-compatible endpoints with custom fields (e.g., reasoning_content from DeepSeek or vLLM), use provider-specific LangChain packages instead.

### Scope

- **Supported**: Official OpenAI APIs (Chat Completions and Responses)
- **Not Supported**: Non-standard provider fields (e.g., reasoning_content, reasoning_details from OpenRouter, vLLM, or DeepSeek)
- **Alternative**: Use provider-specific packages (e.g., `ChatDeepSeek`, `ChatOpenRouter`)

## Class Hierarchy

```
BaseChatModel (from langchain_core)
  └── BaseChatOpenAI
      └── ChatOpenAI
```

`BaseChatOpenAI` contains shared implementation for OpenAI-based models; `ChatOpenAI` is the standard entry point for OpenAI's official APIs.

## Core Configuration Parameters

### Model & API Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` (or `model_name`) | `str` | `"gpt-3.5-turbo"` | Name of the OpenAI model to use (e.g., `"gpt-4-turbo"`, `"gpt-4o"`, `"o1"`). |
| `use_responses_api` | `bool \| None` | `None` | Whether to use the Responses API instead of Chat Completions. When `None`, automatically set to `True` if model or configuration requires it. |
| `output_version` | `str \| None` | `None` | Version of `AIMessage` output format. Supported: `'v0'` (legacy), `'responses/v1'` (Responses API with content blocks), `'v1'` (LangChain v1 standard). |

### Sampling & Generation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | `float \| None` | `None` | Sampling temperature (0-2). Controls randomness; 0 = deterministic, higher = more random. For `o1` models, defaults to 1 if unset. For `gpt-5` non-chat models, only 1 or unset is allowed. |
| `top_p` | `float \| None` | `None` | Nucleus sampling parameter (0-1). Cumulative probability threshold for token selection. |
| `top_logprobs` | `int \| None` | `None` | Number of most-likely tokens to return at each position with log probabilities. Requires `logprobs=True`. |
| `max_tokens` (alias: `max_completion_tokens`) | `int \| None` | `None` | Maximum number of tokens to generate in the completion. |
| `logprobs` | `bool \| None` | `None` | Whether to return log probabilities of the output tokens. |
| `seed` | `int \| None` | `None` | Seed for deterministic output (when supported by the model). |
| `n` | `int \| None` | `None` | Number of chat completions to generate for each prompt. Must be 1 when streaming. |

### Penalties & Bias

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `presence_penalty` | `float \| None` | `None` | Penalizes repeated tokens; range: -2.0 to 2.0. Positive values encourage new tokens. |
| `frequency_penalty` | `float \| None` | `None` | Penalizes tokens by frequency; range: -2.0 to 2.0. Positive values discourage frequent tokens. |
| `logit_bias` | `dict[int, int] \| None` | `None` | Modify the likelihood of specified tokens. Maps token IDs to bias values (-100 to 100). |

### Reasoning & Output (Responses API)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reasoning_effort` (Chat Completions) | `str \| None` | `None` | Reasoning effort for reasoning models. Values: `'minimal'`, `'low'`, `'medium'`, `'high'`. For Chat Completions API only. |
| `reasoning` (Responses API) | `dict \| None` | `None` | Reasoning configuration for Responses API. Structure: `{"effort": "low\|medium\|high", "summary": "auto\|concise\|detailed"}`. |
| `verbosity` | `str \| None` | `None` | Verbosity level for reasoning models (Responses API). Values: `'low'`, `'medium'`, `'high'`. |

### Streaming & Token Usage

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `streaming` | `bool` | `False` | Whether to stream results. When `True`, enables streaming via `stream()` or `astream()`. |
| `stream_usage` | `bool \| None` | `None` | Include usage metadata in streaming chunks. Auto-enabled for default base URL and OpenAI clients; disabled when `openai_api_base` or custom client is set. |
| `stream_chunk_timeout` | `float \| None` | `120.0` | Per-chunk wall-clock timeout (seconds) for async streaming. Fires between SSE chunks. Set to `None` or `0` to disable. Configurable via `LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S` env var. |

### Authentication & Connection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` (or `openai_api_key`) | `str \| None \| Callable[[], str] \| Callable[[], Awaitable[str]]` | From `OPENAI_API_KEY` env var | OpenAI API key. Can be a string, sync callable, or async callable. LangSmith gateway keys (prefix: `lsv2_`) are detected for gateway routing. |
| `base_url` (or `openai_api_base`) | `str \| None` | From `OPENAI_API_BASE` or `OPENAI_BASE_URL` env vars | Base URL for API requests. Use for proxies, service emulators, or OpenAI-compatible endpoints. Resolution order: explicit param → `OPENAI_API_BASE` → `OPENAI_BASE_URL` (SDK). |
| `organization` (or `openai_organization`) | `str \| None` | From `OPENAI_ORG_ID` env var | OpenAI organization ID for multi-organization accounts. |
| `timeout` (or `request_timeout`) | `float \| tuple[float, float] \| httpx.Timeout \| None` | `None` | Request timeout(s). Can be a single float (connect & read), tuple (connect, read), or httpx.Timeout object. |
| `max_retries` | `int \| None` | `None` | Maximum number of retries for transient failures. |

### HTTP Client & Networking

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `http_client` | `httpx.Client \| None` | `None` | Custom synchronous HTTP client. Must be paired with `http_async_client` for async operations. |
| `http_async_client` | `httpx.AsyncClient \| None` | `None` | Custom asynchronous HTTP client. Must be paired with `http_client` for sync operations. |
| `http_socket_options` | `Sequence[tuple[int, int, int]] \| None` | From env vars or defaults | TCP socket options `(level, option, value)` applied to httpx transports. Defaults to conservative keepalive + `TCP_USER_TIMEOUT`. Set to `()` to disable. Configurable via `LANGCHAIN_OPENAI_TCP_KEEPALIVE*` env vars. |
| `openai_proxy` | `str \| None` | From `OPENAI_PROXY` env var | Explicit HTTP proxy for OpenAI requests. Cannot be combined with custom `http_client`/`http_async_client`. |
| `default_headers` | `Mapping[str, str] \| None` | `None` | Default HTTP headers for all requests. |
| `default_query` | `Mapping[str, object] \| None` | `None` | Default query parameters for all requests. |

### Advanced Request Handling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_kwargs` | `dict[str, Any]` | `{}` | Additional OpenAI API parameters not explicitly defined. Parameters are merged into the top-level request payload. Use for standard OpenAI params like `stream_options`, `max_completion_tokens`, `modalities`, `audio`. |
| `extra_body` | `Mapping[str, Any] \| None` | `None` | Custom parameters for OpenAI-compatible providers (vLLM, LM Studio, OpenRouter). Parameters are nested under `extra_body` in requests. **Do not use `model_kwargs` for non-OpenAI parameters.** |
| `stop` (or `stop_sequences`) | `list[str] \| str \| None` | `None` | Default stop sequences. Can be overridden per invocation. |

### Responses API & Conversation State

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_previous_response_id` | `bool` | `False` | If `True`, automatically pass `previous_response_id` from the most recent response. Responses API only. Input messages up to that response are dropped from payloads. |
| `store` | `bool \| None` | `None` | Whether OpenAI may store response data. Defaults to `True` for Responses API, `False` for Chat Completions. |
| `truncation` | `str \| None` | `None` | Truncation strategy for Responses API. Values: `'auto'` (model drops middle items), `'disabled'` (default, no truncation). |
| `context_management` | `list[dict[str, Any]] \| None` | `None` | Configuration for context management / compaction (Responses API). |
| `include` | `list[str] \| None` | `None` | Additional fields to include in Responses API output. Values: `'file_search_call.results'`, `'message.input_image.image_url'`, `'computer_call_output.output.image_url'`, `'reasoning.encrypted_content'`, `'code_interpreter_call.outputs'`. |

### Prompt Caching

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt_cache_options` | `dict[str, Any] \| None` | `None` | Options for OpenAI prompt cache control. Structure: `{"mode": "implicit" or "explicit", "ttl": "30m"}`. Can be set per-model or per-invocation. |

### Service Tiers & Other

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `service_tier` | `str \| None` | `None` | Latency tier for requests. Values: `'auto'`, `'default'`, `'flex'`. For OpenAI scale tier users. |
| `tiktoken_model_name` | `str \| None` | `None` | Model name for tiktoken token counting. Defaults to the chat model name. Use for Azure endpoints or unsupported models. |
| `disabled_params` | `dict[str, Any] \| None` | `None` | Parameters to disable for a model. Format: `{"param": None}` (disable entirely) or `{"param": ['val1', 'val2']}` (disable specific values). Example: `{"parallel_tool_calls": None}` for models not supporting parallel tool calls. |
| `include_response_headers` | `bool` | `False` | Include response headers in `response_metadata`. Useful for capturing provider-specific metadata (e.g., served model names). |

## Key Methods

### Primary Invocation

```python
def invoke(input: LanguageModelInput, **kwargs) -> AIMessage
```

Generate a single response from the model. Returns an `AIMessage` with content, usage metadata, and response metadata.

```python
async def ainvoke(input: LanguageModelInput, **kwargs) -> AIMessage
```

Async version of `invoke`.

### Streaming

```python
def stream(input: LanguageModelInput, **kwargs) -> Iterator[AIMessageChunk]
```

Stream response chunks from the model. Each chunk is an `AIMessageChunk` with partial content. Set `stream_usage=True` or `stream_options={"include_usage": True}` to include final usage metadata.

```python
async def astream(input: LanguageModelInput, **kwargs) -> AsyncIterator[AIMessageChunk]
```

Async version of `stream`.

### Batch Processing

```python
def batch(inputs: List[LanguageModelInput], **kwargs) -> List[AIMessage]
```

Process multiple inputs in parallel batches.

```python
async def abatch(inputs: List[LanguageModelInput], **kwargs) -> List[AIMessage]
```

Async version of `batch`.

### Tool Binding & Structured Output

```python
def bind_tools(tools: Sequence[BaseTool | dict | type], **kwargs) -> Runnable
```

Bind tools to the model. Returns a `Runnable` that extracts tool calls from model output. Supports `strict=True` for strict schema enforcement and `parallel_tool_calls=False` to disable parallel invocation.

```python
def with_structured_output(
    schema: _DictOrPydanticClass | None = None,
    *,
    method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
    include_raw: bool = False,
    strict: bool | None = None,
    **kwargs
) -> Runnable
```

Wrap the model to return structured output matching a schema. Methods:

- `'json_schema'`: OpenAI Structured Output API (most reliable)
- `'function_calling'`: Tool-calling API (legacy)
- `'json_mode'`: JSON mode (least reliable; requires prompt instructions)

## Response Metadata

Response metadata is returned in `AIMessage.response_metadata`:

```python
{
    "token_usage": {
        "prompt_tokens": int,
        "completion_tokens": int,
        "total_tokens": int,
        # Responses API may include:
        "input_token_details": {...},
        "output_token_details": {...},
        "cache_creation_input_tokens": int,  # prompt cache write
        "cache_read_input_tokens": int,      # prompt cache read
    },
    "model_name": str,
    "system_fingerprint": str | None,  # Unique identifier for model version
    "finish_reason": str,  # "stop", "length", "tool_calls", "content_filter", "error", ...
    "logprobs": dict | None,  # When logprobs=True
    # Responses API:
    "id": str,  # Response ID (use with use_previous_response_id or previous_response_id param)
}
```

## Usage Metadata

Token usage is also available in `AIMessage.usage_metadata`:

```python
{
    "input_tokens": int,
    "output_tokens": int,
    "total_tokens": int,
}
```

## Message Conversion

The class handles bidirectional conversion between LangChain and OpenAI message formats:

### LangChain → OpenAI

- `HumanMessage` → `{"role": "user", ...}`
- `AIMessage` → `{"role": "assistant", ...}` (includes tool calls as OpenAI format)
- `SystemMessage` → `{"role": "system" or "developer", ...}` (developer role for o-series models)
- `ToolMessage` → `{"role": "tool", ...}`
- `FunctionMessage` → `{"role": "function", ...}`

### OpenAI → LangChain

- `role: "user"` → `HumanMessage`
- `role: "assistant"` → `AIMessage` (parses tool calls; collects invalid calls)
- `role: "system" or "developer"` → `SystemMessage` (preserves role in `additional_kwargs`)
- `role: "tool"` → `ToolMessage`
- `role: "function"` → `FunctionMessage`

## Error Handling

The class maps OpenAI API errors to LangChain exceptions:

| OpenAI Error | LangChain Exception |
|--------------|-------------------|
| `AuthenticationError` | `ModelAuthenticationError` |
| `PermissionDeniedError` | `ModelPermissionDeniedError` |
| `RateLimitError` | `ModelRateLimitError` |
| `APIConnectionError` | `ModelConnectionError` |
| `APITimeoutError` | `ModelTimeoutError` |
| `APIStatusError` (4xx) | `ModelInvalidRequestError` or `ModelNotFoundError` |
| `APIStatusError` (5xx) | `ModelAPIError` |
| Context overflow | `ContextOverflowError` |
| Model refusal | `ModelRefusalError` |

## Environment Variables

Resolution order (first match wins):

```
API Key:
  1. Explicit `api_key` parameter
  2. `OPENAI_API_KEY`

Base URL:
  1. Explicit `base_url` parameter
  2. `OPENAI_API_BASE`
  3. `OPENAI_BASE_URL` (read by underlying SDK)

Organization:
  1. Explicit `organization` parameter
  2. `OPENAI_ORG_ID` or `OPENAI_ORGANIZATION`

Proxy:
  1. Explicit `openai_proxy` parameter
  2. `OPENAI_PROXY`
  3. `HTTP_PROXY` / `HTTPS_PROXY` / system proxies (httpx auto-detection)

Output Version:
  1. `LC_OUTPUT_VERSION` env var
```

## Advanced Behaviors

### API Routing

The model choice between Chat Completions and Responses APIs is inferred from:

1. Explicit `use_responses_api` parameter
2. Model name (models with specific prefixes default to Responses API)
3. Configuration (e.g., `reasoning`, `verbosity`, `use_previous_response_id`)

Use `use_responses_api=False` explicitly to force Chat Completions for Responses-API-capable models.

### Temperature Constraints

- **o1 models**: Default to `temperature=1` if not specified; required to be 1.
- **gpt-5 non-chat models**: Only allow `temperature=1` unless `reasoning_effort="none"` or `reasoning={"effort": "none"}`.

### System Role Evolution

For **o-series models** (matching `^o\d`), system messages are automatically converted from `role: "system"` to `role: "developer"` in the request payload, as required by the model's API.

### Prompt Caching

Automatic for eligible prompts. For high-volume applications:

- Set `prompt_cache_key` per invocation to route requests to the same machine for better cache hit rates.
- Modes: `'implicit'` (default, automatic + explicit breakpoints) or `'explicit'` (only explicit breakpoints).
- Cache breakpoints on content blocks: `{"type": "text", "text": "...", "prompt_cache_breakpoint": {...}}`.
- Set `prompt_cache_options` per-model or per-invocation.

## OpenAI-Compatible Endpoints

`ChatOpenAI` works with OpenAI-compatible APIs (LM Studio, vLLM, Ollama, etc.):

- Use `base_url` to point to the endpoint
- Use `extra_body` for provider-specific parameters (not `model_kwargs`)
- Set `use_responses_api=False` explicitly to avoid ambiguity
- Example (LM Studio with TTL):

```python
ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="mlx-community/QwQ-32B-4bit",
    extra_body={"ttl": 300},  # Auto-evict after 5 min
)
```

## Initialization Examples

### Basic Usage

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o", temperature=0)
response = model.invoke("What is 2 + 2?")
```

### With Structured Output

```python
from pydantic import BaseModel

class Joke(BaseModel):
    setup: str
    punchline: str

model = ChatOpenAI(model="gpt-4o")
structured = model.with_structured_output(Joke)
result = structured.invoke("Tell me a joke")
# result is a Joke instance
```

### With Tool Binding

```python
from pydantic import BaseModel

class GetWeather(BaseModel):
    location: str

model = ChatOpenAI(model="gpt-4o")
model_with_tools = model.bind_tools([GetWeather])
response = model_with_tools.invoke("What's the weather in SF?")
# response.tool_calls contains [{"name": "GetWeather", "args": {...}, "id": "..."}]
```

### Streaming

```python
for chunk in model.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

### Async

```python
response = await model.ainvoke("What is 2 + 2?")
async for chunk in await model.astream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

### Custom HTTP Client

```python
import httpx
from langchain_openai import ChatOpenAI

client = httpx.Client(limits=httpx.Limits(max_connections=10))
async_client = httpx.AsyncClient(limits=httpx.Limits(max_connections=10))

model = ChatOpenAI(
    model="gpt-4o",
    http_client=client,
    http_async_client=async_client,
)
```

### Responses API with Reasoning

```python
model = ChatOpenAI(
    model="gpt-6",
    use_responses_api=True,
    reasoning={
        "effort": "medium",
        "summary": "auto",
    },
    output_version="responses/v1",
)
response = model.invoke("What is 3^3?")
# response.content includes reasoning blocks
```

## Serialization & Configuration

`ChatOpenAI` is LangChain-serializable:

```python
@classmethod
def is_lc_serializable(cls) -> bool:
    return True

@classmethod
def get_lc_namespace(cls) -> list[str]:
    return ["langchain", "chat_models", "openai"]
```

Configuration can be loaded from YAML/JSON using standard LangChain loaders.

## See Also

- Related page: [/openwiki/openai-provider.md](/openwiki/openai-provider.md) — Integration documentation and usage examples
- OpenAI API Reference: [https://platform.openai.com/docs/api-reference](https://platform.openai.com/docs/api-reference)
- LangChain Docs: [https://python.langchain.com/docs/integrations/chat/openai](https://python.langchain.com/docs/integrations/chat/openai)
