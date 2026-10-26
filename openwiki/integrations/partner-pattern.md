---
type: "Integration Pattern"
title: "Partner Pattern: Provider Integration Framework"
description: "Architecture and implementation patterns for adding new provider integrations to LangChain. Covers ChatModel subclassing, message conversion, tool calling, and structured output support."
tags: [partner, integration, chat-model, tool-calling, structured-output, message-conversion, package-structure]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-25T08:23:03.589Z
sources:
  - id: openwiki-source-c52037e7b642f7ac5a7642a8
    resource: repo://libs/core/langchain_core/language_models/chat_models.py
  - id: openwiki-source-a12395e96060f062d8ec1ca2
    resource: repo://libs/partners/anthropic/langchain_anthropic/__init__.py
  - id: openwiki-source-1e66a9da38565f8901e651f4
    resource: repo://libs/partners/openai/langchain_openai/__init__.py
  - id: openwiki-source-738512768ef81ae009b097ac
    resource: repo://libs/partners/openai/langchain_openai/chat_models/base.py
  - id: openwiki-source-df762860acfcc6abf0ce804b
    resource: repo://libs/partners/openai/pyproject.toml
  - id: openwiki-source-86b6689572ac828885d7d4b0
    resource: repo://libs/partners/README.md
generated: { by: "openwiki/0.5.0", at: "2026-10-25T08:23:03.589Z" }
---

# Partner Pattern: Provider Integration Framework

LangChain's partner pattern defines how provider-specific integrations (like OpenAI, Anthropic, Ollama) implement support for chat models, embeddings, and other abstractions. Partner packages live in `libs/partners/<provider>/` and implement the core abstractions defined in `langchain-core` by adapting each provider's API to LangChain's unified message and runnable interfaces.

## Partner Package Structure

Each partner package follows a standardized layout. The OpenAI partner (`langchain-openai`) exemplifies this structure:

```
libs/partners/openai/
  langchain_openai/
    __init__.py              # Public API exports
    chat_models/
      base.py               # BaseChatOpenAI and ChatOpenAI implementations
      azure.py              # Azure-specific variant
      _client_utils.py      # Helper functions for client setup
      _compat.py            # Version/compatibility handling
    embeddings/
      base.py               # OpenAI embeddings integration
    llms/                   # Legacy completion API support
    tools/                  # Tool-specific helpers
    output_parsers/         # Custom output parsing logic
    data/
      _profiles.py          # Model capability profiles
  pyproject.toml            # Package metadata and dependencies
  README.md                 # Integration documentation
  Makefile                  # Development targets
  tests/                    # Test suite
```

### pyproject.toml Dependencies

Partner packages declare dependencies on:
- **`langchain-core`**: Provides base abstractions (`BaseChatModel`, `BaseRunnable`, messages, tools).
- **Provider SDK**: The official client library (e.g., `openai>=2.45.0`, `anthropic>=...`).
- **Optional extras**: For advanced features (e.g., `azure-identity` for Azure OpenAI).

Example (from langchain-openai):

```toml
[project]
name = "langchain-openai"
dependencies = [
    "langchain-core>=1.6.10,<2.0.0",
    "openai>=2.45.0",
    "tiktoken>=0.7.0",
]

[project.optional-dependencies]
azure-identity = ["azure-identity>=1.25.0"]
```

### Public API (_\_init\_\_.py)

Partners export a curated set of classes, typically:
- `ChatOpenAI`, `AzureChatOpenAI`: Chat model implementations
- `OpenAI`, `AzureOpenAI`: Legacy completion models (if supported)
- `OpenAIEmbeddings`: Embeddings integration
- Utility classes: `OpenAIDecisions`, `custom_tool`

Only stable, user-facing classes should be exported; internal helpers remain private.

## ChatModel Subclassing

The central pattern: every partner implements one or more **`BaseChatModel` subclasses** that wrap the provider's API. `ChatOpenAI` serves as the canonical example.

### Required Method Implementations

All `BaseChatModel` subclasses **must** implement:

#### `_generate(messages, stop, run_manager, **kwargs) → ChatResult`

The core sync generation method called by `invoke`, `generate`, and streaming operations. Returns a `ChatResult` wrapping `ChatGeneration` objects with `AIMessage` and optional metadata.

```python
def _generate(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    run_manager: CallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> ChatResult:
    """Generate chat response from messages."""
    # 1. Convert LangChain messages to provider format
    # 2. Build request payload from model params and kwargs
    # 3. Call provider API
    # 4. Parse provider response to AIMessage
    # 5. Extract and attach metadata (tokens, finish reason, etc.)
    # 6. Return ChatResult
```

For OpenAI, this method orchestrates:
1. Message normalization and conversion to OpenAI role/content format
2. Payload construction via `_get_request_payload()`
3. API invocation via the OpenAI SDK
4. Result parsing via `_create_chat_result()`

#### `_llm_type` (property)

A string identifier for logging, serialization, and tracing (e.g., `"openai"`, `"anthropic"`, `"ollama"`). Must be unique within the ecosystem.

```python
@property
def _llm_type(self) -> str:
    """Return type of llm."""
    return "openai"
```

### Optional Streaming Support

Implementing these methods enables optimized streaming paths:

#### `_stream(messages, stop, run_manager, **kwargs) → Iterator[ChatGenerationChunk]`

If not implemented, `stream()` falls back to `invoke()`. Streaming is essential for real-time UX. OpenAI's `_stream` method:
- Sets `stream=True` on the request
- Iterates chunks from the provider's streaming API
- Converts each chunk to a `ChatGenerationChunk` with incremental content
- Fires callbacks (`run_manager.on_llm_new_token()`) for each chunk
- Handles provider errors and maps them to LangChain error classes

#### `_agenerate(messages, stop, run_manager, **kwargs) → ChatResult`

Native async generation. If absent, `ainvoke` runs `_generate` in an executor. OpenAI implements this to use the `AsyncOpenAI` client for non-blocking I/O.

#### `_astream(messages, stop, run_manager, **kwargs) → AsyncIterator[ChatGenerationChunk]`

Async streaming. OpenAI wraps its async SDK's streaming with per-chunk timeouts to prevent indefinite hangs.

## Message Conversion

The boundary where provider APIs differ most: each partner must convert between LangChain's unified `BaseMessage` format and the provider's role/content schema.

### LangChain Message Types

Core message types in `langchain_core.messages`:
- **`HumanMessage`**: User input
- **`AIMessage`**: Model output; carries `tool_calls` (structured as `ToolCall` objects) and `additional_kwargs` (provider-specific fields like token counts, reasoning content)
- **`SystemMessage`**: Model instructions
- **`ToolMessage`**: Result from tool execution
- **`FunctionMessage`**: Legacy OpenAI function invocation result
- **Content**: String or list of content blocks (text, image, audio, file) for multimodal support

### Provider-Specific Conversion

OpenAI message conversion (in `_convert_message_to_dict`):
- `HumanMessage` → `{"role": "user", "content": ...}`
- `AIMessage` → `{"role": "assistant", "content": ..., "tool_calls": [...]}` (if tool calls present)
- `SystemMessage` → `{"role": "system" | "developer", "content": ...}` (respects `__openai_role__` field for developer role)
- `ToolMessage` → `{"role": "tool", "tool_call_id": ..., "content": ...}`
- `FunctionMessage` → `{"role": "function", "name": ..., "content": ...}` (legacy)

Tool calls are parsed from the provider's response format (OpenAI uses `{"function": {"name": ..., "arguments": ...}}`) and wrapped as LangChain `ToolCall` objects carrying `id`, `name`, and `args`.

Reverse conversion (provider response → `AIMessage`) handles:
- Extracting text content and tool invocation data
- Parsing provider-specific metadata (token counts, finish reason, reasoning content, audio)
- Storing non-standard fields in `additional_kwargs` or content blocks for later retrieval

### Streaming Delta Conversion

Streaming requires incremental parsing. OpenAI's `_convert_delta_to_message_chunk` parses chunks containing partial text, tool call deltas (partial JSON), and token usage, accumulating them into `ChatGenerationChunk` objects that carry `tool_call_chunks` for efficient merging.

## Tool Calling Integration

`bind_tools(tools, tool_choice, **kwargs)` binds tool schemas to a model, returning a wrapped `Runnable` that encodes tools in the API request.

### Schema Binding

OpenAI's `bind_tools` accepts tools as:
- `BaseTool` instances (from `langchain_core.tools`)
- Pydantic models (with docstrings and field descriptions)
- Functions (analyzed for signature and docstring)
- Dicts (raw tool schemas)

Each is converted to the provider's format via `convert_to_openai_tool()` (from `langchain_core.utils.function_calling`), which produces JSON Schema representations compatible with the provider's API.

### Function Calling Flow

1. **Binding**: User calls `model.bind_tools([GetWeather, GetPopulation])`
2. **Schema conversion**: Each tool is converted to OpenAI's `{"type": "function", "function": {...}}` format
3. **Request encoding**: Tools are passed to the provider API alongside the message
4. **Response parsing**: Provider returns tool calls; OpenAI parser extracts them into `ToolCall` objects
5. **Tool execution**: Framework executes each tool; results become `ToolMessage` objects
6. **Feedback**: `ToolMessage` objects are added to the message history for next model call

### Tool Choice Control

`tool_choice` parameter steers the model:
- `"none"`: Never call a tool
- `"auto"`: Model decides
- `"any"` / `"required"` / `True`: Force at least one tool call
- `{"type": "function", "function": {"name": "GetWeather"}}`: Force a specific tool
- `None` / `False`: Use provider default

OpenAI normalizes aliases (e.g., `"any"` → `"required"`) and handles both string shortcuts and dicts.

### Strict Mode

When `strict=True`, the model is constrained to output exactly matching the JSON Schema. This is enforced server-side by compatible providers and requires schema validation. Enabling strict mode on OpenAI also requires all tools to be strict-compatible (relevant for Chat Completions API with response formats).

## Structured Output Support

`with_structured_output(schema, method="function_calling")` wraps a model to return outputs conforming to a schema, without requiring explicit tool definitions.

### Methods

OpenAI supports three methods:

#### `"function_calling"` (Default)

Treats the schema as a single tool and forces the model to call it. The `PydanticToolsParser` extracts the tool argument, yielding the parsed Pydantic instance (if schema is Pydantic) or a dict (if schema is JSON).

Advantages: Broad model support, simpler error handling.
Limitations: Requires schema compatibility with tool calling.

#### `"json_schema"`

Uses OpenAI's Structured Outputs API (JSON Schema mode), which enforces exact schema matching server-side. Requires `langchain-openai >= 0.3.0` and GPT models supporting the feature (not gpt-3.5, not gpt-4, yes to gpt-4o and newer).

Advantages: Strongest guarantee of schema compliance; supports complex schemas.
Limitations: Newer models only; higher API overhead.

#### `"json_mode"`

Requests JSON-formatted output via `response_format={"type": "json_object"}`. The model generates valid JSON but is **not** constrained to match the schema. Parsing is client-side via `JsonOutputParser`.

Advantages: Works with all chat models.
Limitations: Schema is a hint, not enforced; more parsing errors.

### Pydantic Binding

If schema is a Pydantic class:
- OpenAI generates a `with_structured_output` wrapper that applies `PydanticToolsParser` (for function_calling/json_schema) or `PydanticOutputParser` (for json_mode)
- Output is a Pydantic instance, fully validated
- Errors during parsing are raised or caught (if `include_raw=True`)

If schema is a dict:
- Output is a dict (no Pydantic validation)
- Parsing still uses JSON parsers but yields dicts

### Error Handling & Fallback

When `include_raw=True`, the wrapper returns a dict:
```python
{
    "raw": AIMessage(...),
    "parsed": <schema_instance> | None,
    "parsing_error": <exception> | None,
}
```

Parsing errors are caught and attached rather than raised, allowing graceful fallbacks.

## Configuration Patterns

Partners expose configuration across three layers:

### Constructor Parameters

Core model parameters are declared as Pydantic fields. OpenAI's example:

```python
class BaseChatOpenAI(BaseChatModel):
    model_name: str = Field(default="gpt-3.5-turbo", alias="model")
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    api_key: SecretStr | None = Field(alias="api_key", default=None)
    base_url: str | None = Field(alias="base_url", default=None)
    timeout: float | tuple | None = Field(alias="timeout", default=None)
    max_retries: int | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
```

Key patterns:
- **`model_kwargs`**: Catch-all dict for unsupported parameters; passed directly to the provider API
- **Aliases**: Support alternative field names (e.g., `model_name` aliased to `"model"`)
- **Defaults**: Reasonable defaults inferred from environment or provider (e.g., `OPENAI_API_KEY`)
- **SecretStr**: Sensitive fields like API keys are automatically redacted in logs

### Environment Variable Fallback

Constructors read environment variables if explicit values are not provided:

```python
openai_api_key = Field(default=None, ...)  # Env: OPENAI_API_KEY
openai_api_base = Field(default=None, ...)  # Env: OPENAI_API_BASE, then OPENAI_BASE_URL
openai_organization = Field(default=None, ...) # Env: OPENAI_ORG_ID
```

The `from_env()` utility and Pydantic validators orchestrate this fallback.

### Endpoint Overrides

Partners often support custom endpoints for proxies, fine-tuned models, or on-prem deployments:

```python
model = ChatOpenAI(
    base_url="http://localhost:8000/v1",  # vLLM, LM Studio, etc.
    api_key="fake-key"
)
```

Partners validate that overridden endpoints match expected schemas (e.g., OpenAI format compliance checks).

### Client Configuration

Advanced configuration for HTTP clients, proxies, and timeouts:

```python
model = ChatOpenAI(
    timeout=30.0,
    max_retries=2,
    http_client=my_httpx_client,  # Custom httpx.Client
    openai_proxy="http://proxy:8080",
)
```

OpenAI's implementation handles proxy auto-detection, socket options, and SSL verification, with fallback to sane defaults.

## Testing Patterns

### StandardTests Conformance

Partner packages use `langchain-tests` (in `libs/standard-tests/`) to run a standard test suite verifying:
- **Chat model contract**: `invoke`, `stream`, `batch`, async variants all work
- **Tool calling**: `bind_tools` produces correct schemas and parses responses
- **Structured output**: `with_structured_output` works with Pydantic/dict schemas
- **Serialization**: Models can be dumped/loaded via `to_dict()`/`from_dict()`
- **Callbacks**: Streaming callbacks fire at the right times

Example test registration (from OpenAI's conftest):

```python
def test_standard_base_model_params():
    """Test standard parameters work."""
    model = ChatOpenAI(model="gpt-4.1-mini")
    assert model.invoke("hello") is not None
```

### Real vs. Mock API Testing

- **Real API tests**: Hit the actual provider (slow, require API key, counted against quota). Used for integration tests.
- **Mock/recorded tests**: Use `vcrpy` to record HTTP interactions and replay them. Fast, deterministic, no API key required.

OpenAI uses recorded cassettes in `tests/cassettes/` containing fixture responses for common scenarios.

### Test Categories

1. **Unit tests**: Isolated functionality (message conversion, schema validation, error mapping)
2. **Integration tests**: Full workflows with mocked API responses
3. **Snapshot tests** (via `syrupy`): Capture and diff model output, catching regressions in response parsing

## Error Mapping

Partner packages should map provider-specific errors to LangChain's standardized error hierarchy (from `langchain_core.exceptions`):

```python
# OpenAI example
openai.BadRequestError → OpenAIInvalidRequestError(ModelInvalidRequestError)
openai.AuthenticationError → OpenAIAuthenticationError(ModelAuthenticationError)
openai.RateLimitError → OpenAIRateLimitError(ModelRateLimitError)
openai.APIConnectionError → OpenAIConnectionError(ModelConnectionError)
openai.APITimeoutError → OpenAITimeoutError(ModelTimeoutError)
openai.NotFoundError → OpenAIModelNotFoundError(ModelNotFoundError)
```

Subclasses inherit from both the provider's exception and LangChain's abstract class, allowing framework code to catch errors generically while preserving provider context.

## Model Profiles

Partners can declare model capabilities (context window, supported modalities, tool calling support) in a **model profile registry**. OpenAI stores profiles in `data/_profiles.py`:

```python
_PROFILES = {
    "gpt-4o": {
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "supported_modalities": {"text", "image", "audio"},
        "tool_calling": True,
        "structured_output": True,
    },
    # ...
}
```

Profiles are loaded at model init via `_resolve_model_profile()` and exposed on the instance, enabling runtime capability checks:

```python
if model.profile and model.profile.get("tool_calling"):
    # Use bind_tools
```

## Async Support

Partner implementations should provide native async variants where possible:

- **`_agenerate()`**: Async generation using the provider's async client
- **`_astream()`**: Async streaming with proper timeout handling
- **Callable API keys**: Support `Callable[[], Awaitable[str]]` for async key fetching

OpenAI's async implementation uses `openai.AsyncOpenAI` and includes per-chunk streaming timeouts (`stream_chunk_timeout`) to prevent indefinite hangs on stalled streams.

## Version & Compatibility Management

Partners manage compatibility across:
1. **langchain-core versions**: Ensure dependency bounds match breaking changes
2. **Provider SDK versions**: Pin to avoid upstream API changes
3. **Python versions**: Declare supported Python ranges (e.g., `>=3.10`)
4. **Model evolution**: Detect model capabilities and adapt behavior (e.g., strict mode support, Responses API availability)

OpenAI uses validators to detect model capabilities (e.g., `validate_temperature` for o1/gpt-5 models) and warnings to guide users toward compatible alternatives.

## Extension Boundary

Partners are the extension point for:
- **Provider-specific features**: Built-in tools (web search), reasoning modes, audio I/O
- **Vendor variants**: Azure OpenAI, DeepSeek, OpenRouter (inheriting from base OpenAI or building independently)
- **Custom message formats**: Non-standard blocks (Anthropic thinking, reasoning content) stored in `additional_kwargs` or content lists

The core framework abstractions (BaseChatModel, BaseMessage, Runnable) remain stable; partners handle diversity at the integration boundary.

---

## See Also

- [Architecture](/openwiki/architecture.md): Three-layer design where partners implement core abstractions
- [Chat Models](/openwiki/concepts/chat-models.md): BaseChatModel contract and streaming semantics
- [Messages](/openwiki/concepts/messages.md): Message types, content blocks, and conversion
- [OpenAI Provider](/openwiki/integrations/openai-provider.md): Detailed OpenAI integration walkthrough
