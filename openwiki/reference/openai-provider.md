---
type: "Class Reference"
title: "OpenAI Chat Provider API Reference"
description: "Technical reference for ChatOpenAI and AzureChatOpenAI classes, covering class structure, constructor parameters, initialization behavior, and Azure integration."
tags: ["openai", "chat-models", "api-reference", "azure"]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-07T08:15:07.286Z
sources:
  - id: openwiki-source-f32b395707eda97cd743f4e5
    resource: repo://libs/partners/openai/langchain_openai/chat_models/azure.py
  - id: openwiki-source-738512768ef81ae009b097ac
    resource: repo://libs/partners/openai/langchain_openai/chat_models/base.py
generated: { by: "openwiki/0.5.0", at: "2026-11-07T08:15:07.286Z" }
---

## Overview

The OpenAI integration provides two main chat model classes:

- **`ChatOpenAI`**: Wraps OpenAI's Chat Completions and Responses APIs for general OpenAI deployments.
- **`AzureChatOpenAI`**: Specialized subclass for Azure OpenAI deployments with Azure authentication and endpoint management.

Both inherit from **`BaseChatOpenAI`**, which is the shared base class implementing the core LangChain chat model interface and OpenAI API integration logic.

## Class Hierarchy

```
BaseChatModel (langchain_core)
  ↑
  BaseChatOpenAI
    ↑
    ├─ ChatOpenAI
    └─ AzureChatOpenAI
```

**Design principle**: `ChatOpenAI` targets [official OpenAI API specifications](https://github.com/openai/openai-openapi) only. Non-standard response fields added by third-party providers (e.g., `reasoning_content`, `reasoning_details`) are not extracted or preserved. For provider-specific features, use the corresponding provider-specific LangChain package.

## Location

- **Base class**: `repo://libs/partners/openai/langchain_openai/chat_models/base.py#L787`
- **ChatOpenAI class**: `repo://libs/partners/openai/langchain_openai/chat_models/base.py#L2928`
- **AzureChatOpenAI class**: `repo://libs/partners/openai/langchain_openai/chat_models/azure.py#L65`
- **Module exports**: `repo://libs/partners/openai/langchain_openai/__init__.py`

## BaseChatOpenAI Class

`BaseChatOpenAI` is the core implementation shared by `ChatOpenAI` and `AzureChatOpenAI`. It provides:

- HTTP client management (sync/async httpx clients)
- OpenAI SDK client initialization and lifecycle
- Message conversion between LangChain and OpenAI formats
- API selection logic (Chat Completions vs Responses API)
- Error mapping to LangChain exception hierarchy
- Streaming and callback integration
- Tool calling and structured output handling

### Core Fields

**Client Management** (`repo://libs/partners/openai/langchain_openai/chat_models/base.py#L797-L803`):

```python
client: Any  # Sync OpenAI client (httpx.Client)
async_client: Any  # Async OpenAI client (httpx.AsyncClient)
root_client: Any  # Cached sync OpenAI() instance
root_async_client: Any  # Cached async AsyncOpenAI() instance
```

These are created during `__init__` or lazily on first use. If an async-only API key is provided, `client` remains `None` and sync methods raise `ValueError`.

**Model & API Parameters**:

```python
model_name: str = "gpt-3.5-turbo"  # OpenAI model identifier
temperature: float | None = None
max_tokens: int | None = None
streaming: bool = False
logprobs: bool | None = None
top_logprobs: int | None = None
seed: int | None = None
presence_penalty: float | None = None
frequency_penalty: float | None = None
top_p: float | None = None
n: int | None = None
```

**Authentication & Endpoint** (`repo://libs/partners/openai/langchain_openai/chat_models/base.py#L814-L881`):

```python
openai_api_key: SecretStr | None | Callable[[], str] | Callable[[], Awaitable[str]]
  # Alias: api_key
  # Can be string, sync callable, or async callable
  # Inferred from OPENAI_API_KEY env var

openai_api_base: str | None  # Alias: base_url
  # Resolution order: explicit kwarg → OPENAI_API_BASE env → OPENAI_BASE_URL env

openai_organization: str | None  # Alias: organization
  # Inferred from OPENAI_ORG_ID env var

openai_proxy: str | None
  # Inferred from OPENAI_PROXY env var
```

**Streaming & Timeout**:

```python
stream_usage: bool | None = None
  # Include usage metadata in streaming chunks
  # Defaults to True for default endpoint, False when base_url is set

stream_chunk_timeout: float | None = 120.0
  # Per-chunk timeout for async streaming (seconds)
  # Overridable via LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S

request_timeout: float | tuple[float, float] | Any | None = None
  # Alias: timeout
  # Can be single float (both connect/read) or (connect, read) tuple

max_retries: int | None = None
  # Automatic retries for transient failures (429, 500-599)
```

**HTTP Client Configuration** (`repo://libs/partners/openai/langchain_openai/chat_models/base.py#L1026-L1103`):

```python
http_client: Any | None = None
  # Custom httpx.Client for sync requests
  # Must pair with http_async_client for async use

http_async_client: Any | None = None
  # Custom httpx.AsyncClient for async requests

http_socket_options: Sequence[tuple[int, int, int]] | None = None
  # TCP socket options: (level, option, value) tuples
  # Defaults to conservative TCP-keepalive profile
  # Set to () to disable; env-var overridable via
  # LANGCHAIN_OPENAI_TCP_KEEPALIVE, TCP_KEEPIDLE, TCP_KEEPINTVL, TCP_KEEPCNT,
  # TCP_USER_TIMEOUT_MS
```

**Advanced Features**:

```python
reasoning_effort: str | None = None
  # For Chat Completions API reasoning models
  # Values: 'minimal', 'low', 'medium', 'high'

reasoning: dict[str, Any] | None = None
  # For Responses API reasoning models
  # Shape: {"effort": None | "low" | "medium" | "high",
  #         "summary": "auto" | "concise" | "detailed"}

verbosity: str | None = None
  # Responses API only: 'low', 'medium', 'high'

service_tier: str | None = None
  # Latency tier: 'auto', 'default', 'flex'

store: bool | None = None
  # Whether OpenAI may store response data
  # Defaults to True for Responses API, False for Chat Completions

include_response_headers: bool = False
  # Capture response headers in message.response_metadata
  # Useful for provider metadata (e.g., served model names)

extra_body: Mapping[str, Any] | None = None
  # Additional JSON properties for OpenAI-compatible APIs
  # Recommended over model_kwargs for provider-specific parameters
  # Examples: vLLM, LM Studio TTL, beam search settings

prompt_cache_options: dict[str, Any] | None = None
  # OpenAI prompt caching configuration

include: list[str] | None = None
  # Responses API: additional fields to include in generations
  # Examples: 'file_search_call.results', 'message.input_image.image_url'

truncation: str | None = None
  # Responses API truncation strategy: 'auto' or 'disabled'

context_management: list[dict[str, Any]] | None = None
  # Context compaction configuration

disabled_params: dict[str, Any] | None = None
  # Parameters to disable for the model
  # Shape: {"param": None | ['val1', 'val2']}
  # Example: {"parallel_tool_calls": None} for older models
```

**Other Parameters**:

```python
stop: list[str] | str | None = None  # Alias: stop_sequences
logit_bias: dict[int, int] | None = None  # Token likelihood modifications
tiktoken_model_name: str | None = None  # For token counting
model_kwargs: dict[str, Any] = {}  # Additional API parameters
default_headers: Mapping[str, str] | None = None
default_query: Mapping[str, object] | None = None
```

## Initialization Flow

When `ChatOpenAI.__init__()` or `AzureChatOpenAI.__init__()` is called:

1. **Pydantic model validation**: All field validators run, env vars are resolved (e.g., `OPENAI_API_KEY`)
2. **API key resolution** (`repo://libs/partners/openai/langchain_openai/chat_models/_client_utils.py`):
   - Sync and async key providers are separated
   - If `api_key` is an async callable, `client` (sync) is not available
   - If `api_key` is a sync callable, it's stored for lazy evaluation
3. **HTTP client building**:
   - If `http_client`/`http_async_client` provided, use them as-is
   - Otherwise, build from defaults with socket options applied
   - Socket options include TCP-keepalive and keepidle configuration per platform
4. **OpenAI client initialization**:
   - Create OpenAI SDK client instances (lazy or immediate)
   - Pass timeout, max_retries, organization, proxy settings
5. **Model profile registration**:
   - Load capability metadata from `_PROFILES` registry
6. **Parameter validation**:
   - Validate `stream_chunk_timeout`: negative values fall back to default with warning
   - Verify model compatibility with requested features

## API Selection Logic

`ChatOpenAI` automatically selects between two OpenAI APIs based on model name and parameters:

**Chat Completions API** (default, most models):
- Standard request/response flow
- Supports: tool calling, structured output (function_calling, json_mode), streaming
- Parameters: `temperature`, `max_tokens`, `top_p`, `presence_penalty`, `frequency_penalty`, `logprobs`, `seed`, `logit_bias`, `n`
- Reasoning: via `reasoning_effort` (minimal, low, medium, high)

**Responses API** (newer models, advanced features):
- Streaming reasoning, context management, truncation control
- Automatically selected when:
  - Model name starts with `gpt-5` (pro), `gpt-6`, or contains `codex`
  - `use_responses_api=True` is explicitly set
  - `reasoning` or `context_management` parameters provided
  - `truncation` or `include` parameters set
  - `use_previous_response_id=True` is set

Method: `_should_use_responses_api()` determines selection at call time.

## Message Conversion

### Input Message Conversion

LangChain message types → OpenAI API format via `_convert_message_to_dict()` (`repo://libs/partners/openai/langchain_openai/chat_models/base.py#L476`):

| LangChain Type | OpenAI Role | Handling |
|---|---|---|
| `HumanMessage` | `"user"` | Content directly |
| `AIMessage` | `"assistant"` | Content + tool_calls + tool_call_id + audio |
| `SystemMessage` | `"system"` or `"developer"` | Marked with `__openai_role__` key if developer |
| `ToolMessage` | `"tool"` | Requires `tool_call_id` |
| `FunctionMessage` | `"function"` | Legacy support |
| `ChatMessage` | Custom role | Role passed through |

**Content formatting** (`repo://libs/partners/openai/langchain_openai/chat_models/base.py#L385`):
- `_format_message_content()` filters content blocks per API
- Chat Completions: removes `tool_use`, `thinking`, `reasoning_content`, `function_call`, `code_interpreter_call` blocks
- Responses API: preserves all block types
- Supports image, audio, text blocks with prompt cache breakpoints

### Output Message Conversion

OpenAI response → LangChain `AIMessage` via `_create_chat_result()`:

```python
AIMessage(
    content: str  # Text response
    tool_calls: list[ToolCall]  # Parsed tool calls with id, name, args
    invalid_tool_calls: list[InvalidToolCall]  # Malformed tool calls
    additional_kwargs: dict  # audio, function_call (legacy), refusal
    response_metadata: dict  # token_usage, model_name, system_fingerprint, logprobs, finish_reason
    usage_metadata: UsageMetadata  # Standard counts: input_tokens, output_tokens, total_tokens
)
```

## Error Mapping

OpenAI SDK exceptions are mapped to LangChain standardized hierarchy (`repo://libs/partners/openai/langchain_openai/chat_models/base.py#L642-L681`):

| OpenAI Exception | LangChain Class | Condition |
|---|---|---|
| `AuthenticationError` | `ModelAuthenticationError` | Invalid API key |
| `PermissionDeniedError` | `ModelPermissionDeniedError` | API key lacks permissions |
| `BadRequestError` | `ContextOverflowError` | context_length_exceeded, Input tokens exceed limit |
| `BadRequestError` | `ModelInvalidRequestError` | response_format validation, unsupported schema |
| `RateLimitError` | `ModelRateLimitError` | Rate limit exceeded |
| `NotFoundError` | `ModelNotFoundError` | Model doesn't exist/unavailable |
| `InternalServerError` | `ModelAPIError` | OpenAI server error (5xx) |
| `APIConnectionError` | `ModelConnectionError` | Network connectivity issue |
| `APITimeoutError` | `ModelTimeoutError` | Request timeout |

Custom error classes inherit from both OpenAI SDK and LangChain exception types for exception filtering.

## ChatOpenAI Class

`ChatOpenAI` is the standard class for OpenAI API deployments. It inherits all `BaseChatOpenAI` fields and methods without additional parameters.

### Instantiation

```python
model = ChatOpenAI(
    model="gpt-4o",  # Required
    temperature=0.7,  # Optional
    api_key="sk-...",  # Optional, from OPENAI_API_KEY if omitted
    base_url="https://api.openai.com/v1",  # Optional, for proxy/custom endpoint
    max_tokens=1000,
    timeout=30.0,
    stream_chunk_timeout=120.0,
    streaming=False,
)
```

### Core Methods

**Sync invocation**:
```python
def invoke(
    self,
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> AIMessage
```

**Async invocation**:
```python
async def ainvoke(
    self,
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> AIMessage
```

**Streaming**:
```python
def stream(
    self,
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> Iterator[AIMessageChunk]

async def astream(
    self,
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> AsyncIterator[AIMessageChunk]
```

**Batch processing**:
```python
def batch(
    self,
    inputs: list[LanguageModelInput],
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> list[AIMessage]

async def abatch(
    self,
    inputs: list[LanguageModelInput],
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> list[AIMessage]
```

## AzureChatOpenAI Class

`AzureChatOpenAI` is the Azure-specific subclass. It inherits all `BaseChatOpenAI` functionality and adds Azure-specific fields.

### Azure-Specific Fields

**Endpoint & Deployment**:

```python
azure_endpoint: str | None = None
  # Full Azure endpoint URL (e.g., https://resource-name.openai.azure.com/)
  # Inferred from AZURE_OPENAI_ENDPOINT env var
  # Required unless overridden in __init__

azure_deployment: str | None = None
  # Name of Azure OpenAI deployment
  # Used to construct request URL: /deployments/{azure_deployment}/chat/completions
  # Required parameter

api_version: str | None = None
  # Azure OpenAI REST API version (distinct from model version)
  # Examples: "2024-05-01-preview", "2024-02-15-preview"
  # Inferred from OPENAI_API_VERSION env var
  # Controls feature availability (structured output, reasoning, etc.)
```

**Authentication**:

```python
api_key: str | Callable[[], str] | Callable[[], Awaitable[str]] | None = None
  # Azure API key
  # Inferred from AZURE_OPENAI_API_KEY env var

azure_ad_token: str | None = None
  # Azure Active Directory token (alternative to API key)
  # For workload identity/managed identity authentication

azure_ad_token_provider: Callable[[], str] | Callable[[], Awaitable[str]] | None = None
  # Callable that returns an Azure AD token
  # Auto-configured from workload identity env vars if available:
  # AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_FEDERATED_TOKEN_FILE
```

**Model Identification** (for tracing & token counting only):

```python
model: str | None = None
  # Underlying OpenAI model name for tracing and token counting
  # Does NOT affect completion; azure_deployment determines routed endpoint
  # Examples: "gpt-4o", "gpt-4-turbo", "gpt-35-turbo"

model_version: str | None = None
  # Model version for token counting (e.g., '0125', '0125-preview')
  # Does NOT affect completion; documentation/tracing only
```

### Initialization Behavior

Workload identity detection (`repo://libs/partners/openai/langchain_openai/chat_models/azure.py#L40-L58`):

If all of these env vars are set, automatic token providers are created:
- `AZURE_CLIENT_ID`
- `AZURE_TENANT_ID`
- `AZURE_FEDERATED_TOKEN_FILE`

When present, sync and async token providers are auto-configured and used for `azure_ad_token_provider`.

### Usage Differences from ChatOpenAI

1. **Endpoint routing**: Uses `azure_deployment` in URL path instead of model name
2. **Response metadata**: Includes Azure-specific fields:
   - `prompt_filter_results`: Content safety filtering (hate speech, self-harm, sexual, violence)
   - `content_filter_results`: Safety categorization details
3. **API version coupling**: Feature availability tied to `api_version` (e.g., structured output only in newer versions)
4. **Model parameter semantics**: `model` and `model_version` are documentation-only; affect tracing/token counting, not API calls

### Azure Example

```python
model = AzureChatOpenAI(
    azure_endpoint="https://my-resource.openai.azure.com/",
    azure_deployment="my-gpt-4o-deployment",
    api_version="2024-05-01-preview",
    api_key="your-api-key",  # or use AZURE_OPENAI_API_KEY
    temperature=0.7,
    max_tokens=1000,
)

response = model.invoke("Translate to French: Hello")
print(response.response_metadata)  # Includes Azure safety filtering metadata
```

## Model Profiles

Model capabilities are described via `ModelProfile` objects stored in `_PROFILES` registry (`repo://libs/partners/openai/langchain_openai/data/_profiles.py`).

**Profile fields**:

```python
text_inputs: bool  # Text support
text_outputs: bool  # Text output support
image_inputs: bool  # Vision support (gpt-4o, gpt-4-vision, gpt-4-turbo)
audio_inputs: bool  # Audio input support (gpt-4o, upcoming)
audio_outputs: bool  # Audio output support (gpt-4o, upcoming)
video_inputs: bool  # Video input support (future)
tool_calling: bool  # Function/tool calling support
structured_output: bool  # JSON Schema structured output support
max_input_tokens: int | None  # Context window size
max_output_tokens: int | None  # Generation limit
tool_call_streaming: bool  # Whether tool calls stream incrementally
tool_choice: bool  # Whether tool_choice parameter is supported
```

**Access**:

```python
from langchain_openai import ChatOpenAI
from langchain_core.language_models import ModelProfileRegistry

model = ChatOpenAI(model="gpt-4o")
# Profiles are used internally for capability checks
```

The registry is used by `bind_tools()` and `with_structured_output()` to validate compatibility.

## Streaming Mechanics

### Sync Streaming

```python
model = ChatOpenAI(model="gpt-4o", streaming=True)
for chunk in model.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

**Chunk types**:
- `AIMessageChunk` with `content` (text delta)
- `AIMessageChunk` with `tool_call_chunks` (incremental tool calls)
- Final chunk may include `usage_metadata` if `stream_usage=True`

### Async Streaming with Chunk Timeout

```python
async def stream_response():
    model = ChatOpenAI(
        model="gpt-4o",
        streaming=True,
        stream_chunk_timeout=30.0
    )
    async for chunk in model.astream("Tell me a long story"):
        print(chunk.content, end="", flush=True)
```

**Timeout behavior** (`repo://libs/partners/openai/langchain_openai/chat_models/base.py#L1105-L1145`):

- Measures silence between *parsed chunks*, not inter-byte silence
- Distinct from httpx read timeout (which resets on keepalive SSE comments)
- Fires `StreamChunkTimeoutError` (subclass of `asyncio.TimeoutError`)
- Default: 120s; set to `None` or `0` to disable
- Overridable via `LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S` env var

### Callback Integration

Streaming callbacks fire on each chunk via `run_manager.on_llm_new_token()`:

```python
from langchain_core.callbacks import StreamingStdOutCallbackHandler

model = ChatOpenAI(model="gpt-4o", streaming=True)
for chunk in model.stream(
    "Hello",
    config={"callbacks": [StreamingStdOutCallbackHandler()]}
):
    pass  # Callback prints tokens as they arrive
```

## Tool Calling

### bind_tools() Method

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

**Tool format support**:
- `BaseTool` instances (from `@tool` decorator)
- Pydantic `BaseModel` classes
- Callables with type hints
- OpenAI tool schema dicts

**tool_choice parameter**:
- `str` (tool name): Force specific tool
- `'auto'`: Auto-select tool or none (default)
- `'none'`: Prevent tool calling
- `'any'` / `'required'` / `True`: Force at least one tool call
- `dict`: OpenAI tool choice dict `{"type": "function", "function": {"name": "..."}}`
- `'file_search'`, `'web_search'`, etc.: WellKnownTools built-in tools
- `False` / `None`: No effect

**parallel_tool_calls**:
- `None` (default): Allow parallel
- `False`: Disable (one tool at a time)

**strict**:
- `True`: Output matches tool schema exactly
- `False`: No validation
- `None`: No strict requirement

**response_format**:
- Optional response schema for Chat Completions API
- When set with tools, requires `strict=True` (except Responses API)

### Tool Call Extraction

Response includes `AIMessage.tool_calls`:

```python
ToolCall(
    id: str  # Unique ID from API
    name: str  # Tool name
    args: dict  # Parsed arguments
    type: Literal["tool_call"] = "tool_call"
)
```

Invalid tool calls go in `AIMessage.invalid_tool_calls`:

```python
InvalidToolCall(
    id: str
    name: str | None
    args: str  # Unparsed argument string
    error: str  # Parse error message
)
```

## Structured Output

### with_structured_output() Method

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

**Methods**:

1. **`function_calling`** (default):
   - Uses tool-calling API
   - Supported on most models
   - Less strict than `json_schema`

2. **`json_schema`**:
   - Uses Structured Output API
   - Strict conformance guaranteed
   - Requires model with `structured_output=True` in profile
   - Validates schema per OpenAI's supported-schemas constraints

3. **`json_mode`**:
   - Model generates JSON without schema validation
   - Works on more models
   - Requires manual prompt engineering

**Schema formats**:
- Pydantic `BaseModel` (output is instance)
- JSON Schema dict
- `TypedDict`
- OpenAI tool schema dict

**include_raw**:
When `True`, returns dict with keys:
- `'raw'`: Original `AIMessage`
- `'parsed'`: Parsed schema instance or None if parsing failed
- `'parsing_error'`: Exception if parsing failed

**tools** (Responses API only with `json_schema`):
Additional tools the model can call alongside structured output. Requires `method="json_schema"`, `strict=True`, `include_raw=True`.

## Token Counting

`BaseChatOpenAI` integrates `tiktoken` for token counting (`repo://libs/partners/openai/langchain_openai/chat_models/base.py#L1006-L1018`):

```python
tiktoken_model_name: str | None = None
  # Model name for tiktoken (defaults to model_name)
  # Useful when using Azure or provider-specific models not in tiktoken
```

Usage counts are tracked in response metadata:

```python
response.usage_metadata = UsageMetadata(
    input_tokens: int,
    output_tokens: int,
    total_tokens: int
)
```

Image token estimates:
- `low` detail: ~85 tokens
- `high` detail: ~170 + 255 per image tile (based on resolution)

## Known Limitations

1. **Sync callable API keys**: If `api_key` is a sync callable, async methods still work but resolve the key in executor thread.
2. **Provider-specific fields**: Non-OpenAI response fields (e.g., from vLLM, DeepSeek) are not preserved.
3. **Responses API parameter support**: Not all Chat Completions parameters are supported in Responses API (e.g., `n`).
4. **Schema validation**: `json_schema` method requires schemas to meet OpenAI's supported-schemas constraints.
5. **Azure API version coupling**: Feature availability tied to `api_version` selection.
6. **Proxy auto-detection bypass**: Custom socket options may shadow env-proxy settings; explicit configuration recommended when using both.

## Testing

Unit tests are located in `repo://libs/partners/openai/tests/unit_tests/`:

- `chat_models/test_base.py`: Main ChatOpenAI tests
- `chat_models/test_base_standard.py`: Standard test suite (Chat Completions)
- `chat_models/test_responses_standard.py`: Standard test suite (Responses API)
- `chat_models/test_azure.py`: Azure-specific tests
- `chat_models/test_client_utils.py`: Client utilities tests

Integration tests in `repo://libs/partners/openai/tests/integration_tests/chat_models/`.
