---
type: "Reference"
title: "Chat Models: LLM Integration Points"
openwiki_generated: true
verified:
  - by: openwiki/0.5.0
    at: 2026-10-25T08:23:03.589Z
sources:
  - id: openwiki-source-c52037e7b642f7ac5a7642a8
    resource: repo://libs/core/langchain_core/language_models/chat_models.py
  - id: openwiki-source-b32b84365d17276620c41ebc
    resource: repo://libs/core/langchain_core/messages/base.py
  - id: openwiki-source-c479d4fffee5cf62576699e4
    resource: repo://libs/langchain_v1/langchain/chat_models/base.py
  - id: openwiki-source-7de1ace618efbdfd8bacb5cb
    resource: repo://libs/partners/anthropic/langchain_anthropic/chat_models.py
  - id: openwiki-source-738512768ef81ae009b097ac
    resource: repo://libs/partners/openai/langchain_openai/chat_models/base.py
generated: { by: "openwiki/0.5.0", at: "2026-10-25T08:23:03.589Z" }
---


# Chat Models: LLM Integration Points

Chat models form the core abstraction for conversational AI in LangChain, operating as **message-in, message-out** interfaces that accept lists of messages and return AI responses. The ecosystem consists of a unified base contract (`BaseChatModel` in core), a universal factory function (`init_chat_model`), and dozens of provider-specific implementations that adapt each LLM API to the common interface.

## Core Architecture: BaseChatModel

**BaseChatModel** (in `langchain_core.language_models.chat_models`) is the abstract base class that all chat implementations inherit from. It extends `BaseLanguageModel` with message-specific semantics and serves three responsibilities:

1. **Interface Contract**: Defines which methods subclasses must implement (`_generate`, `_llm_type`) and which are optional (`_stream`, `_agenerate`, `_astream`).

2. **Runnable Integration**: Acts as a `Runnable` that accepts `LanguageModelInput` (strings, lists of messages, or prompt values) and produces `AIMessage` outputs, enabling use in chains.

3. **Behavior Implementation**: Provides imperative methods (`invoke`, `ainvoke`, `stream`, `astream`, `generate`, `agenerate`) and declarative methods (`bind_tools`, `with_structured_output`) that compose chat models into larger workflows.

### Minimal Implementation Contract

A custom chat model **must** override:

- **`_generate(messages, stop, run_manager, **kwargs) → ChatResult`**: The core sync generation method; called by `invoke`, `generate`, and stream operations. Returns a `ChatResult` containing `ChatGeneration` objects with `AIMessage` and optional generation metadata.

- **`_llm_type` (property)**: A string identifier for logging and serialization (e.g., `"openai"`, `"anthropic"`).

Optional overrides enable optimized paths:

- **`_stream(messages, stop, run_manager, **kwargs) → Iterator[ChatGenerationChunk]`**: Yields streaming chunks; if not provided, `stream()` falls back to `invoke()`.
- **`_agenerate(messages, stop, run_manager, **kwargs) → ChatResult`**: Native async generation; if not provided, runs `_generate` in an executor.
- **`_astream(messages, stop, run_manager, **kwargs) → AsyncIterator[ChatGenerationChunk]`**: Native async streaming.

### Message Conversion and Normalization

**Input normalization** happens in `_convert_input()`, which accepts:
- A bare string (converted to `StringPromptValue`)
- A list of `BaseMessage` objects or message-like tuples/dicts (converted to `ChatPromptValue`)
- Existing `PromptValue` objects (passed through)

All inputs are normalized to a list of `BaseMessage` objects before reaching `_generate`. Implementations typically call `_normalize_messages()` to ensure consistent internal representation.

**Message types** include:
- `HumanMessage`: User input.
- `AIMessage`: Model response, potentially containing `tool_calls` (structured as `ToolCall` objects) and `additional_kwargs` (provider-specific fields like token counts).
- `SystemMessage`: Model instructions.
- `ToolMessage`: Results from tool execution.
- `FunctionMessage`: Legacy function invocation result (OpenAI-specific).
- Content can be a single string or a list of content blocks (text, image, audio, file) for multimodal support.

**Provider-specific conversion** occurs at the boundary:
- OpenAI models convert `BaseMessage` to OpenAI's role/content format, handling image URL transformation and tool call parsing.
- Anthropic models map `HumanMessage` → `"user"`, `AIMessage` → `"assistant"`, normalizing tool call IDs to match Anthropic's strict requirements.
- Each provider may adjust block structure, extract reasoning content, or hoist metadata fields.

## Streaming Architecture

Chat models support two streaming modes:

### V1 Streaming (Legacy Callback-Based)

When a `_StreamingCallbackHandler` is attached or `stream=True` is explicit, `BaseChatModel.stream()` and `.astream()` call the underlying `_stream()` / `_astream()` methods, yielding `ChatGenerationChunk` objects that carry incremental content. Each chunk is:
- Assigned a run ID if missing.
- Decorated with `response_metadata` (token counts, finish reason, etc.).
- Dispatched to `on_llm_new_token()` callbacks for real-time processing.

### V2 Streaming (Protocol Events)

When a `_V2StreamingCallbackHandler` is attached, `_generate_with_cache()` routes through `_iter_v2_events()` / `_aiter_v2_events()`, which:
- Prefer native `_stream_chat_model_events()` / `_astream_chat_model_events()` hooks if implemented.
- Otherwise, bridge `_stream()` chunks into standardized `MessagesData` events via `chunks_to_events()`.
- Fire `on_stream_event()` for each event, enabling LangGraph integration and structured tracing.

### Streaming Bypass

The `disable_streaming` field controls when streaming is suppressed:
- `True`: Always use non-streaming (sync `invoke`).
- `"tool_calling"`: Use non-streaming only when `tools` are bound.
- `False` (default): Use streaming if available and requested.

## Tool Calling Integration

### Binding and Schema Conversion

`bind_tools(tools, tool_choice=None, **kwargs)` binds tool schemas to a model, returning a wrapped `Runnable` that encodes tool information into the API call. Tools are specified as:
- `BaseTool` instances (have `.invoke()` and introspectable `args_schema`).
- Pydantic models (with docstrings and field descriptions).
- Plain functions (analyzed for signature and docstring).
- Dictionaries (raw tool schemas).

The base implementation raises `NotImplementedError`; each provider's subclass converts tools to its format:
- **OpenAI**: Uses `convert_to_openai_tool()` to create function schema dictionaries.
- **Anthropic**: Converts to `AnthropicTool` with `input_schema` (JSON Schema) and optional beta headers for built-in tools.

### Tool Calling in Responses

When bound tools are used, the model returns `AIMessage` objects with `tool_calls` (list of `ToolCall` objects) in place of or alongside text content. Each `ToolCall` contains:
- `id`: Unique identifier for the call.
- `name`: Tool/function name.
- `args`: Dictionary of argument values (parsed from JSON if necessary).
- `type`: Always `"tool_call"`.

`additional_kwargs` may carry provider-specific tool metadata (OpenAI includes full `tool_calls` array; Anthropic includes Anthropic-specific fields like `stop_reason`).

### Structured Output Pattern

`with_structured_output(schema, include_raw=False, **kwargs)` wraps a model to enforce structured responses. The flow is:

1. Bind the schema as a single tool with `tool_choice="any"`.
2. Invoke the bound model (tool calling is guaranteed).
3. Parse tool arguments via `JsonOutputToolsParser` (dict) or `PydanticToolsParser` (Pydantic instance).
4. If `include_raw=True`, return `{"raw": message, "parsed": result, "parsing_error": None|exception}`.

Works with:
- Pydantic models (output is validated `BaseModel` instance).
- TypedDict classes (output is dict).
- JSON schema dicts (output is dict).
- Raw OpenAI function schema dicts.

## Message Output and Response Metadata

### AIMessage Structure

Generated `AIMessage` objects carry:

- **`content`**: String (most common) or list of content blocks (multimodal). String content can be empty when model returns only tool calls.
- **`tool_calls`**: List of `ToolCall` objects if model performed tool use.
- **`additional_kwargs`**: Provider-specific fields (e.g., `{"tool_calls": [...], "refusal": "..."`}).
- **`response_metadata`**: Response headers, token usage, model name, finish reason, logprobs—extracted from the API response or provider wrapper.
- **`usage_metadata`**: Normalized `UsageMetadata` with `input_tokens`, `output_tokens`, `total_tokens`, and optional token cost breakdown (`cache_read_tokens`, `cache_creation_tokens`).
- **`id`**: Unique message identifier (typically a run ID prefixed with `"lc-"` in LangChain traces).

### Content Blocks and Output Version

The `output_version` field controls how multimodal content is stored:
- **`None` / `"v0"` (default)**: Provider-specific format in `content`; `content_blocks` property lazily parses into standard format.
- **`"v1"`**: Standard format written to `content` directly, making `content_blocks` and `content` equivalent.

This enables gradual migration to a unified content format across providers.

## Caching and Generation Lifecycle

### Cache Lookup and Hit Handling

Before calling `_generate`, the internal `_generate_with_cache()` method:
1. Serializes message inputs and call parameters into a cache key.
2. Queries the configured cache (instance, global, or none).
3. On **cache hit**, returns cached `ChatGeneration` objects and replays v2 events if a v2 handler is attached (ensuring consistent event streams).
4. On **cache miss** (or no cache), applies rate limiting and proceeds to generation.

Cache entries store full `ChatGeneration` objects, including `AIMessage` with all metadata. On hit, token cost is zeroed to reflect no API consumption.

### Rate Limiting

If `rate_limiter` (a `BaseRateLimiter`) is set, `acquire()` or `aacquire()` is called before generation, enforcing request pacing at the model level.

### Token Counting Integration

Response metadata is augmented with gateway metadata (if routed through LangSmith) and token details:
- `InputTokenDetails` and `OutputTokenDetails` for fine-grained token breakdown.
- `UsageMetadata` aggregates totals and optional cost.

## Multi-Batch Generation

### generate() and agenerate()

These methods process lists of message lists in batched or parallel fashion:
- `generate(messages: list[list[BaseMessage]], ...)` → `LLMResult` with generations for each message list.
- `agenerate(...)` concurrently invokes `_agenerate_with_cache()` for each message list.

Both accept the same kwargs as `invoke` (temperature, max_tokens, stop sequences, etc.), which are forwarded to `_generate`. Callbacks are invoked per message list (via `on_chat_model_start` with batch size).

### LLMResult and Metadata Combination

The returned `LLMResult` contains:
- `generations`: List of lists of `ChatGeneration` objects (one list per input message list).
- `llm_output`: Combined output from all generations (via `_combine_llm_outputs`, usually `None`).
- `run`: List of `RunInfo` objects (if callbacks were used) linking each generation to its run ID.

## Universal Model Factory: init_chat_model()

The `init_chat_model()` function (in `langchain_v1.langchain.chat_models.base`) provides a unified interface to instantiate any supported chat model without hardcoding provider imports. It handles:

### Provider Selection

Models are specified as:
- **Prefixed form**: `"openai:gpt-4"`, `"anthropic:claude-opus-4-7"` → provider inferred from prefix.
- **Bare model name**: `"gpt-4"` → provider inferred via heuristics (OpenAI models start with `gpt-`, `o1`, `o3`; Anthropic with `claude`, etc.).
- **Explicit provider**: `init_chat_model("gpt-4", model_provider="openai")`.

Supported providers (with example models):
- `openai` (gpt-4, gpt-4-turbo, gpt-4o, o1, o3)
- `anthropic` (claude-opus, claude-sonnet, claude-haiku)
- `azure_openai`, `azure_ai`
- `google_vertexai`, `google_genai`, `google_anthropic_vertex`
- `mistralai`, `groq`, `deepseek`, `xai`, `cohere`, `perplexity`, `upstage`
- `bedrock`, `bedrock_converse`, `anthropic_bedrock` (AWS)
- `ollama`, `together`, `fireworks`, `huggingface`, `ibm`, `nvidia`, `litellm`, `openrouter`, `baseten`, `meta`, `langsmith`

### Fixed vs. Configurable Models

- **Fixed model** (`model` specified, `configurable_fields=None`): Returns a ready-to-use `BaseChatModel`.
- **Configurable model** (`configurable_fields` set): Returns a `_ConfigurableModel` proxy that defers instantiation until a config with runtime parameters is passed to `invoke()`.

For configurable models:
- `configurable_fields="any"`: All init params (including `api_key`, `base_url`) become runtime-configurable via `config["configurable"][...]`.
- `configurable_fields=["model", "temperature"]`: Only listed fields are configurable.
- `config_prefix="foo"`: Fields are prefixed in config keys (e.g., `config["configurable"]["foo_model"]`).

Queued declarative operations (`.bind_tools()`, `.with_structured_output()`) are applied when the underlying model is instantiated, enabling declarative tool binding on configurable models.

## Provider Implementations

### Core Provider Pattern

All provider packages follow a common structure:

```
langchain_<provider>/
├── chat_models.py          # Main Chat<Provider> class
├── _compat.py              # Message conversion
├── _client_utils.py        # API client setup
└── data/
    └── _profiles.py        # ModelProfile registry
```

Each provider's chat model:
1. Extends `BaseChatModel` (or an intermediate base like `BaseChatOpenAI`).
2. Implements `_generate()` (required) and optionally `_stream()`, `_agenerate()`, `_astream()`.
3. Overrides `bind_tools()` to convert schemas to provider format.
4. Optionally overrides `with_structured_output()` if using provider-specific structured output features.
5. Implements `_resolve_model_profile()` to load model metadata from `_profiles.py`.

### Major Implementations

**OpenAI (`langchain_openai.ChatOpenAI`)**:
- Routes to `/v1/chat/completions` endpoint.
- Converts `BaseMessage` to role/content dicts; extracts tool calls from `function` and `tool_calls` fields.
- Supports vision (image URLs and base64), audio, GPT-4 Turbo functions, and structured outputs via both tool calling and native modes.
- Uses `_compat` to map between OpenAI's v0/v03 message formats and LangChain's unified format.

**Anthropic (`langchain_anthropic.ChatAnthropic`)**:
- Routes to `/v1/messages` endpoint.
- Converts `BaseMessage` to `content` (list of text/image/tool_use blocks); normalizes tool call IDs to match Anthropic's alphanumeric requirement.
- Supports vision (base64 and URL), extended thinking (with `budget_tokens`), and computer use / web fetch (built-in tools).
- Handles tool result merging: sequences of `ToolMessage` are merged into single `HumanMessage` with `tool_result` blocks.

**Ollama (`langchain_ollama.ChatOllama`)**:
- Routes to local or remote Ollama server (default `http://localhost:11434`).
- Supports open-source models (llama2, mistral, neural-chat, etc.).
- Limited tool calling (depends on model); reasoning content extraction from `additional_kwargs`.

**Specialized Providers**:
- `ChatFireworks`, `ChatGroq`, `ChatDeepSeek`: API-compatible with OpenAI; handle provider-specific extensions (reasoning content, extended thinking, etc.).
- `ChatMistralAI`, `ChatPerplexity`, `ChatXAI`, `ChatUpstage`: Each with custom message conversion and tool support.
- Multi-cloud: `ChatBedrockConverse` (AWS), `ChatVertexAI` (Google), `AzureChatOpenAI` (Microsoft).

## Integration Boundaries and Extension

### Message Block Translators

`langchain_core.messages.block_translators.openai` provides utilities for provider-specific content block formats:
- `convert_to_openai_image_block()`: Converts LangChain image blocks to OpenAI format.
- `convert_to_openai_data_block()`: Handles audio/file data.

Other providers define similar translation modules.

### Tool and Function Calling Utilities

`langchain_core.utils.function_calling`:
- `convert_to_openai_tool()`: Converts Pydantic models, TypedDict, functions → OpenAI tool schema.
- `convert_to_json_schema()`: Converts Pydantic/TypedDict → JSON Schema.

Provider packages extend this with `convert_to_<provider>_tool()` as needed (e.g., `convert_to_anthropic_tool()`).

### Output Parsing

`langchain_core.output_parsers.openai_tools` provides parsers for tool-calling responses:
- `JsonOutputToolsParser`: Extracts tool calls as dicts.
- `PydanticToolsParser`: Extracts and validates against Pydantic models.
- `JsonOutputKeyToolsParser`: Extracts by tool name and returns specific fields.

## Lifecycle and State Management

### Model Initialization

1. Subclass `__init__` accepts provider-specific parameters (API key, base URL, model name, temperature, etc.).
2. Pydantic validators enforce constraints (e.g., temperature in `[0, 2]`, max_tokens >= 1).
3. `_set_model_profile()` (Pydantic `mode="after"` validator) auto-loads metadata from `_profiles.py` if not explicit.
4. Client and auth setup happens lazily (on first use) in implementations.

### Serialization and Caching

- `_serialized` (cached property): Dumps model to dict via `dumpd()` for LangSmith tracing.
- `to_json()`, `from_json()`: Serialize/deserialize model definition (not API keys).
- `_dict_for_compat()`: Returns identifying parameters for logging.

### Error Handling

- Model API errors are caught and converted to `langchain_core.exceptions` (e.g., `ModelAPIError`, `ModelAuthenticationError`, `ContextOverflowError`).
- Response parsing errors trigger `on_llm_error()` callbacks and propagate as exceptions (unless caught by fallbacks or retries).
- Stream errors during generation produce partial generations with error metadata.

## Tracing and Observability

### Callback Integration

Each invocation fires:
- `on_chat_model_start(serialized, messages_to_trace, invocation_params, options, ...)`: Model starting.
- `on_llm_new_token(content, chunk=...)` (during streaming): Each token.
- `on_stream_event(event)` (during v2 streaming): Protocol events.
- `on_llm_end(LLMResult(...))`: Model finished.
- `on_llm_error(exception, response=...)`: Model failed.

### LangSmith Metadata

`_get_ls_params()` and `_get_ls_params_with_defaults()` extract tracing metadata:
- `ls_provider`: Inferred from class name or overridden by subclass.
- `ls_model_name`: Extracted from `model`, `model_name`, or runtime kwargs.
- `ls_temperature`, `ls_max_tokens`, `ls_stop`: Call parameters.
- `ls_integration`: Always `"langchain_chat_model"`.

## Configuration and Patterns

### Retry and Fallback Chains

Models are composable with retry/fallback wrappers:

```python
model = ChatOpenAI(...).with_retry(max_attempts=3).with_fallbacks(
    [ChatAnthropic(...), ChatOllama(...)]
)
```

On failure, the chain tries the next fallback, enabling graceful degradation.

### Configurable Fields

Models expose configurable fields for runtime swapping:

```python
model = ChatOpenAI(temperature=0).configurable_fields(
    temperature=ConfigurableField(name="Temperature", description="...")
)

model.invoke(input, config={"configurable": {"temperature": 0.7}})
```

### Chain and Agent Patterns

Chat models are typically composed into chains:

```python
from langchain_core.runnables import RunnablePassthrough

chain = (
    RunnablePassthrough.assign(context=retriever)
    | prompt
    | model
    | output_parser
)
```

For agentic patterns, models are called in a loop with tool results fed back as context.

## Performance and Optimization

### Streaming for Latency

Streaming enables real-time token output, reducing time-to-first-token. Use `stream()` / `astream()` when user-facing responsiveness matters.

### Batch Processing

For throughput, use `batch()` / `abatch()` with concurrent execution (default thread pool) or use `agenerate()` for I/O-bound parallelism.

### Caching for Cost

Enable LLM caching to avoid redundant calls for identical inputs:

```python
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

set_llm_cache(InMemoryCache())
```

### Rate Limiting

Apply per-model rate limiting to avoid quota exhaustion:

```python
from langchain_core.rate_limiters import InMemoryRateLimiter

model = ChatOpenAI(rate_limiter=InMemoryRateLimiter(max_requests_per_second=10))
```

## Testing and Standards

Integration tests in `langchain_tests.integration_tests.chat_models` verify:
- Basic invocation, streaming, and async behavior.
- Tool calling (if supported).
- Structured output (if supported).
- Message conversion and role mapping.
- Error handling and recovery.

Each provider package includes tests that exercise its implementation against actual or mock APIs.
