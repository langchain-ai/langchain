---
type: "API Reference"
title: "Chat Models API Reference"
description: "Complete API reference for BaseChatModel, covering all methods, parameters, return types, and protocol implementation contracts for chat model integration in LangChain."
tags: [chat-models, api-reference, llm-integration, streaming, structured-output, tool-calling]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-29T08:20:01.046Z
sources:
  - id: openwiki-source-c52037e7b642f7ac5a7642a8
    resource: repo://libs/core/langchain_core/language_models/chat_models.py
generated: { by: "openwiki/0.5.0", at: "2026-10-29T08:20:01.046Z" }
---

# Chat Models API Reference

This reference documents the complete API for `BaseChatModel`, the protocol base class for all chat model integrations in LangChain. For architectural overview and lifecycle patterns, see [Chat Model Interface and Lifecycle](/openwiki/chat-models.md).

**Module**: `langchain_core.language_models.chat_models`

## BaseChatModel

`BaseChatModel` is an abstract base class that inherits from `BaseLanguageModel[AIMessage]` and implements the `Runnable` protocol. It defines the unified interface for all chat model implementations.

### Core Attributes

#### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rate_limiter` | `BaseRateLimiter \| None` | `None` | Optional rate limiter to constrain request throughput |
| `disable_streaming` | `bool \| Literal["tool_calling"]` | `False` | Streaming bypass mode: `True` disables all streaming; `"tool_calling"` disables streaming only when tools are passed; `False` uses streaming when available |
| `output_version` | `str \| None` | `None` (env: `LC_OUTPUT_VERSION`) | Content block format version: `"v0"` (provider-specific), `"v1"` (standardized), or `None` (auto) |
| `profile` | `ModelProfile \| None` | `None` | Model capability profile (beta feature, auto-loaded if available) |

### Input/Output Types

**Input** (`LanguageModelInput`): Union of:
- `str` → converted to simple user message
- `list[BaseMessage \| dict \| tuple]` → full conversation history
- `PromptValue` → structured prompt value (passed through)

**Output**: `AIMessage` or `AIMessageChunk` (streaming)

### Invocation Methods (Runnable Protocol)

#### Synchronous Methods

##### `invoke()`

```python
def invoke(
    self,
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    *,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> AIMessage
```

Single synchronous chat model call. Converts input to messages, calls the model, and returns the first generation.

**Parameters:**
- `input`: Message history or prompt text
- `config`: Runnable configuration (callbacks, tags, metadata, run_id)
- `stop`: Stop sequences for generation termination
- `**kwargs`: Model-specific parameters passed to `_generate`

**Returns**: `AIMessage` with model response

**Callback Chain**: `on_chat_model_start` → `on_llm_end` | `on_llm_error`

---

##### `stream()`

```python
def stream(
    self,
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    *,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> Iterator[AIMessageChunk]
```

Synchronous streaming of model output. Yields message chunks as they arrive.

**Parameters:**
- `input`: Message history or prompt text
- `config`: Runnable configuration
- `stop`: Stop sequences
- `**kwargs`: Model-specific parameters

**Returns**: Iterator yielding `AIMessageChunk` objects; final chunk has `chunk_position="last"`

**Behavior:**
- Routes to `_stream()` if streaming is implemented and enabled (checked via `_should_stream()`)
- Falls back to `invoke()` if streaming is disabled or not implemented
- Applies `disable_streaming` and `stream=<bool>` overrides
- Respects callback-attached streaming handlers (v1 `_StreamingCallbackHandler`)
- Applies rate limiting before stream starts
- Normalizes messages and handles output content version
- Wraps per-chunk callback with `on_llm_new_token`

**Callback Chain**: `on_chat_model_start` → `on_llm_new_token` (per chunk) → `on_llm_end` | `on_llm_error`

---

##### `batch()`

```python
def batch(
    self,
    inputs: list[LanguageModelInput],
    config: RunnableConfig | None = None,
    *,
    return_exceptions: bool = False,
    **kwargs: Any,
) -> list[AIMessage]
```

Inherited from `Runnable`. Synchronously processes multiple inputs in parallel (concurrent threads). Equivalent to calling `invoke()` for each input concurrently.

---

#### Asynchronous Methods

##### `ainvoke()`

```python
async def ainvoke(
    self,
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    *,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> AIMessage
```

Asynchronous single chat model call. Awaits `agenerate_prompt()`.

**Parameters**: Same as `invoke()`

**Returns**: `AIMessage`

**Callback Chain**: `on_chat_model_start` → `on_llm_end` | `on_llm_error`

---

##### `astream()`

```python
async def astream(
    self,
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    *,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> AsyncIterator[AIMessageChunk]
```

Asynchronous streaming of model output. Yields message chunks as they arrive.

**Parameters**: Same as `stream()`

**Returns**: Async iterator yielding `AIMessageChunk` objects

**Behavior**: Async variant of `stream()` with same routing logic and rate limiting

**Callback Chain**: `on_chat_model_start` → `on_llm_new_token` (per chunk) → `on_llm_end` | `on_llm_error`

---

##### `abatch()`

```python
async def abatch(
    self,
    inputs: list[LanguageModelInput],
    config: RunnableConfig | None = None,
    *,
    return_exceptions: bool = False,
    **kwargs: Any,
) -> list[AIMessage]
```

Inherited from `Runnable`. Asynchronously processes multiple inputs in parallel. Equivalent to concurrently calling `ainvoke()` for each input.

---

### Batch Generation Methods

These methods accept batches of message lists and return aggregated results with generation metadata.

#### `generate()`

```python
def generate(
    self,
    messages: list[list[BaseMessage]],
    stop: list[str] | None = None,
    callbacks: Callbacks = None,
    *,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    run_name: str | None = None,
    run_id: UUID | None = None,
    **kwargs: Any,
) -> LLMResult
```

Pass a batch of message sequences to the model and return full generation results.

**Parameters:**
- `messages`: List of message sequences (each is `list[BaseMessage]`)
- `stop`: Stop sequences
- `callbacks`: Callback handlers
- `tags`: Run tags for tracing
- `metadata`: Run metadata for tracing
- `run_name`: Run name for tracing
- `run_id`: Run ID for tracing
- `**kwargs`: Model-specific parameters

**Returns**: `LLMResult` containing:
- `generations`: List of `list[ChatGeneration]` (one list per input)
- `llm_output`: Combined provider-specific metadata (via `_combine_llm_outputs`)
- `run`: List of `RunInfo` with run IDs (populated if callbacks configured)

**Behavior:**
- Iterates over input messages, calling `_generate_with_cache()` for each
- Caching is transparent: checks cache before calling model, stores result after
- Handles callbacks for each batch item independently
- Catches per-item exceptions and reports via `on_llm_error`, re-raises after callback
- Combines multi-item LLM outputs into single `LLMResult`

---

#### `agenerate()`

```python
async def agenerate(
    self,
    messages: list[list[BaseMessage]],
    stop: list[str] | None = None,
    callbacks: Callbacks = None,
    *,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    run_name: str | None = None,
    run_id: UUID | None = None,
    **kwargs: Any,
) -> LLMResult
```

Asynchronous variant of `generate()`. Concurrently calls `_agenerate_with_cache()` for each input.

**Parameters**: Same as `generate()`

**Returns**: `LLMResult`

**Behavior**: Async variant with concurrent gathering; first exception re-raised after all handlers finish

---

#### `generate_prompt()`

```python
def generate_prompt(
    self,
    prompts: list[PromptValue],
    stop: list[str] | None = None,
    callbacks: Callbacks = None,
    **kwargs: Any,
) -> LLMResult
```

Convenience wrapper that converts `PromptValue` objects to message lists and calls `generate()`.

---

#### `agenerate_prompt()`

```python
async def agenerate_prompt(
    self,
    prompts: list[PromptValue],
    stop: list[str] | None = None,
    callbacks: Callbacks = None,
    **kwargs: Any,
) -> LLMResult
```

Asynchronous variant of `generate_prompt()`.

---

### Stream Events (Advanced)

#### `stream_events()`

```python
def stream_events(
    self,
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    *,
    version: Literal["v1", "v2", "v3"] = "v2",
    stop: list[str] | None = None,
    **kwargs: Any,
) -> Iterator[StreamEvent] | ChatModelStream
```

Stream protocol-level events during model invocation.

**Overloads:**
- `version="v1"` or `"v2"` → `Iterator[StreamEvent]` (standard runnable events)
- `version="v3"` → `ChatModelStream` (structured model output stream with message accumulation)

**Parameters:**
- `version`: Event format version
- `stop`: Stop sequences (used for v3)
- Other params same as `stream()`

**Returns:**
- v1/v2: Iterator of `StreamEvent` dicts with keys `event`, `data`
- v3: `ChatModelStream` object with accumulated message and metadata

**Behavior:**
- v1/v2: Routes to parent `Runnable.stream_events()` with event lifecycle
- v3: Streams through `_chat_model_stream_v3()` which lazily initializes run manager, applies rate limiter, and accumulates message chunks with protocol events

**Callback Integration:**
- v1/v2: `_StreamingCallbackHandler` and `_V2StreamingCallbackHandler` attached via config
- v3: Fires `on_stream_event` for each protocol message event

---

#### `astream_events()`

```python
async def astream_events(
    self,
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    *,
    version: Literal["v1", "v2", "v3"] = "v2",
    stop: list[str] | None = None,
    **kwargs: Any,
) -> AsyncIterator[StreamEvent] | Awaitable[AsyncChatModelStream]
```

Async variant of `stream_events()`.

**Overloads:** Same as `stream_events()`

**Returns:**
- v1/v2: Async iterator of `StreamEvent`
- v3: Awaitable that resolves to `AsyncChatModelStream`

---

### Tool Calling and Structured Output

#### `bind_tools()`

```python
def bind_tools(
    self,
    tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
    *,
    tool_choice: str | None = None,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, AIMessage]
```

Bind tools to the model, enabling tool calling in responses.

**Parameters:**
- `tools`: Tool definitions as OpenAI schema dicts, Pydantic classes, callables, or `BaseTool` instances
- `tool_choice`: Constraint on which tool to use; `"any"` allows the model to choose any tool
- `**kwargs`: Additional model-specific tool configuration

**Returns**: `Runnable` that produces `AIMessage` with tool calls in `additional_kwargs["tool_calls"]`

**Behavior:**
- Implementation is provider-specific; base class raises `NotImplementedError`
- Subclasses convert tool definitions to provider format (e.g., OpenAI function schema)
- Tool calls are embedded in response message, not as separate tool-call objects
- Messages can be parsed with `JsonOutputToolsParser` or `PydanticToolsParser`

**Contract:**
- Tool-calling responses must have structure parseable by output parsers
- `tool_choice="any"` binds schema for forced single-tool structured output
- Callable signatures extracted via `_signature_parameters()` for schema generation

---

#### `with_structured_output()`

```python
def with_structured_output(
    self,
    schema: dict[str, Any] | type,
    *,
    include_raw: bool = False,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]
```

Wrap the model to force output conforming to a schema.

**Parameters:**
- `schema`: Output schema as:
  - Pydantic model class → output is Pydantic instance
  - Typed dict class → output is dict
  - JSON schema dict → output is dict
  - OpenAI tool schema dict → output is dict
- `include_raw`: If `False` (default), return only parsed output; if `True`, return dict with `raw`, `parsed`, `parsing_error` keys
- `**kwargs`: Unsupported kwargs raise `ValueError`

**Returns:** `Runnable` producing:
- Without `include_raw`: Pydantic instance (if Pydantic schema) or dict
- With `include_raw`: Dict with keys `raw` (AIMessage), `parsed` (result or None), `parsing_error` (BaseException or None)

**Behavior:**
- Internally calls `bind_tools([schema], tool_choice="any", ...)`
- Parses tool-call response with `PydanticToolsParser` (Pydantic schema) or `JsonOutputKeyToolsParser` (dict schema)
- With `include_raw=True`, wraps parser with fallback: parsing errors caught and stored in result
- Raises `NotImplementedError` if model doesn't implement `bind_tools()`
- Raises `ValueError` if unsupported kwargs provided (e.g., `method`, `strict`)

**Example:**

```python
class Answer(BaseModel):
    answer: str
    confidence: float

model = ChatOpenAI(model="gpt-4o")
structured = model.with_structured_output(Answer)

result = structured.invoke("Is the sky blue?")
# → Answer(answer="Yes", confidence=0.99)
```

---

### Declarative Composition

#### `bind()`

```python
def bind(self, **kwargs: Any) -> _ChatModelBinding
```

Bind keyword arguments to model calls. Creates a `_ChatModelBinding` that preserves chat-model-specific type overloads (v3 `stream_events` returns).

**Parameters**: Arbitrary kwargs forwarded to every invocation

**Returns**: `_ChatModelBinding` (subclass of `RunnableBinding`)

**Example:**

```python
model = ChatOpenAI(model="gpt-4o")
bound = model.bind(temperature=0.7, top_p=0.9)
result = bound.invoke("Hello")
```

---

#### `with_retry()`

Inherited from `Runnable`. Wrap model with retry logic that catches exceptions and retries with exponential backoff.

```python
def with_retry(
    self,
    *,
    retry_if_exception_type: tuple[type[Exception], ...] = (Exception,),
    wait_strategy: BaseWaitStrategy = ...,
    stop_strategy: BaseStopStrategy = ...,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, AIMessage]
```

---

#### `with_fallbacks()`

Inherited from `Runnable`. Wrap model with fallback models invoked on exception.

```python
def with_fallbacks(
    self,
    fallbacks: list[Runnable[LanguageModelInput, AIMessage]],
    *,
    exceptions_to_handle: tuple[type[Exception], ...] = (Exception,),
    exception_key: str | None = None,
) -> Runnable[LanguageModelInput, AIMessage]
```

---

#### `configurable_fields()`

Inherited from `Runnable`. Mark init arguments as configurable at runtime via `RunnableConfig`.

```python
def configurable_fields(
    self,
    **kwargs: ConfigurableField,
) -> Runnable[LanguageModelInput, AIMessage]
```

---

#### `configurable_alternatives()`

Inherited from `Runnable`. Specify alternative models to swap in at runtime.

```python
def configurable_alternatives(
    self,
    which: ConfigurableFieldSpec,
    **kwargs: Runnable[LanguageModelInput, AIMessage],
) -> Runnable[LanguageModelInput, AIMessage]
```

---

### Extension Points (for Subclass Implementation)

#### Required: `_generate()`

```python
@abstractmethod
def _generate(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    run_manager: CallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> ChatResult
```

Abstract method that all subclasses must implement. Performs the actual model call.

**Parameters:**
- `messages`: Normalized message list
- `stop`: Stop sequences
- `run_manager`: Callback manager for this call; used to report `on_llm_new_token` during streaming
- `**kwargs`: Model-specific parameters

**Returns**: `ChatResult` with `generations` (list of `ChatGeneration`) and optional `llm_output` dict

**Contract:**
- Called by `_generate_with_cache()` after cache check and rate limiting
- Should not handle caching (done by caller)
- Should not fire lifecycle callbacks directly (done by caller)
- May report tokens via `run_manager.on_llm_new_token()` if streaming is synchronous
- Provider-specific output stored in `ChatGeneration.generation_info`

---

#### Required: `_llm_type` property

```python
@property
@abstractmethod
def _llm_type(self) -> str
```

Return a unique string identifier for the model type. Used for logging and model differentiation.

**Returns**: String like `"openai"`, `"anthropic"`, `"bedrock"`, etc.

---

#### Optional: `_stream()`

```python
def _stream(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    run_manager: CallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> Iterator[ChatGenerationChunk]
```

Implement streaming support. If not overridden, streaming falls back to `invoke()`.

**Parameters**: Same as `_generate()`

**Yields**: `ChatGenerationChunk` objects (message chunks)

**Contract:**
- Base implementation raises `NotImplementedError`
- Called by `stream()` and `astream()` after callbacks and rate limiting
- Each chunk should have `message` as `AIMessageChunk` with partial content
- Final chunk should include complete metadata in `generation_info`
- May report via `run_manager.on_llm_new_token()` if callback reports are desired

---

#### Optional: `_astream()`

```python
async def _astream(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    run_manager: AsyncCallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> AsyncIterator[ChatGenerationChunk]
```

Implement async streaming. If not overridden, falls back to `_stream()` in executor.

**Parameters**: Same as `_generate()`

**Yields**: `ChatGenerationChunk` objects

**Contract**: Same as `_stream()`, async variant

---

#### Optional: `_agenerate()`

```python
async def _agenerate(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    run_manager: AsyncCallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> ChatResult
```

Implement async generation. If not overridden, falls back to `_generate()` in executor.

**Parameters**: Same as `_generate()`

**Returns**: `ChatResult`

**Contract**: Async variant of `_generate()` with same semantics

---

#### Optional: `_identifying_params` property

```python
@property
def _identifying_params(self) -> dict[str, Any]
```

Return a dict of model parameters for serialization and tracing.

**Returns**: Dict like `{"model": "gpt-4", "temperature": 0.7}`

**Default**: Returns empty dict; subclasses should override

**Used for:**
- `asdict()` serialization
- LangSmith tracing metadata
- Model differentiation in logs

---

#### Optional: `_resolve_model_profile()`

```python
def _resolve_model_profile(self) -> ModelProfile | None
```

Override to provide model capability profile (beta feature). Called automatically by `_set_model_profile` validator.

**Parameters**: None (uses `self` attributes like `self.model`, `self.model_name`)

**Returns**: `ModelProfile` TypedDict or `None` if unavailable

**Example Profile Keys**: `context_window`, `supports_vision`, `supports_tool_calling`, `supports_parallel_function_calling`

**Default**: Returns `None` (no profile); partners typically override to load profile from package data

---

#### Optional: `_stream_chat_model_events()` and `_astream_chat_model_events()`

```python
def _stream_chat_model_events(
    self,
    messages: list[BaseMessage],
    *,
    stop: list[str] | None = None,
    run_manager: CallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> Iterator[MessagesData]
```

```python
async def _astream_chat_model_events(
    self,
    messages: list[BaseMessage],
    *,
    stop: list[str] | None = None,
    run_manager: AsyncCallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> AsyncIterator[MessagesData]
```

Implement protocol-level v2 event streaming (advanced). If provided, these are used directly; otherwise `_stream` / `_astream` output is bridged via `chunks_to_events` / `achunks_to_events`.

**Parameters**: Same as `_stream()` / `_astream()`

**Yields**: `MessagesData` protocol event objects

**Contract:**
- Called only when v2 handlers (`_V2StreamingCallbackHandler`) are attached
- Each event is dispatched to handler via `run_manager.on_stream_event()`
- Final message assembly still via normal `_stream()` / `_astream()` bridge
- Rarely overridden; only for models with native protocol event support

---

### Caching and Rate Limiting

#### Internal: `_generate_with_cache()`

```python
def _generate_with_cache(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    run_manager: CallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> ChatResult
```

Internal method that wraps `_generate()` with caching and rate limiting.

**Parameters**: Same as `_generate()`

**Returns**: `ChatResult`

**Behavior:**
- Checks `self.cache` and global `get_llm_cache()`
- Cache key: LLM string (model config) + serialized messages
- On cache hit: replays v2 events if handler attached, returns cached result
- On cache miss: applies rate limiter, calls `_generate()` or v2 streaming path
- On v2 streaming: accumulates chunks into `ChatModelStream`, fires `on_stream_event` per event
- Stores result in cache for future hits
- Zero-costs cached results (sets `usage_metadata.total_cost = 0`)

---

### Configuration Helpers

#### `_get_invocation_params()`

```python
def _get_invocation_params(
    self,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]
```

Extract model initialization and invocation parameters for tracing and serialization.

**Returns**: Dict merging `_dict_for_compat()` (model config) with call kwargs

**Used by:** LangSmith tracing, `_get_llm_string()` for cache keys

---

#### `_get_ls_params()`

```python
def _get_ls_params(
    self,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> LangSmithParams
```

Extract LangSmith-specific parameters from model config and call kwargs.

**Returns**: `LangSmithParams` dict with keys like `ls_provider`, `ls_model_name`, `ls_temperature`, `ls_max_tokens`, `ls_stop`

**Behavior:**
- Derives `ls_provider` from class name (e.g., `ChatOpenAI` → `"openai"`)
- Checks `kwargs["model"]`, then `self.model`, then `self.model_name` for `ls_model_name`
- Checks `kwargs["temperature"]` and `self.temperature` for `ls_temperature`
- Checks `kwargs["max_tokens"]` and `self.max_tokens` for `ls_max_tokens`
- Subclasses should override to handle provider-specific model identifier names

---

### Serialization and Utility

#### `asdict()`

```python
def asdict(self) -> dict[str, Any]
```

Return a dictionary representation of the model configuration.

**Returns**: Dict with `_type` key (from `_llm_type`) and model params (from `_identifying_params`)

---

#### `OutputType` property

```python
@property
def OutputType(self) -> Any
```

Return the output type for this Runnable (used for type hints and validation).

**Returns**: `AnyMessage` (union of all message types)

---

## SimpleChatModel

`SimpleChatModel` is a convenience base class for implementations that need only a simple string-output interface.

```python
class SimpleChatModel(BaseChatModel):
    @abstractmethod
    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str
```

**Subclass Contract:**
- Override `_call()` to return response string
- `_generate()` wrapper automatically creates `AIMessage(content=...)` and wraps in `ChatResult`
- Async calls fall back to `_generate()` in executor
- Simpler than full `BaseChatModel` but less flexible

---

## _ChatModelBinding

`_ChatModelBinding` is a `RunnableBinding` subclass returned by `BaseChatModel.bind()`.

```python
class _ChatModelBinding(RunnableBinding[LanguageModelInput, AIMessage])
```

**Purpose:** Preserves chat-model-specific type overloads for `stream_events(version="v3")` and `astream_events(version="v3")` so the type checker knows the return type is `ChatModelStream` or `AsyncChatModelStream`.

**Behavior:**
- Serializes as plain `RunnableBinding` for backward compatibility
- Preserves narrowed typing across further `bind()` and `with_config()` calls
- At runtime, behaviorally identical to `RunnableBinding`

---

## Output Types

### ChatGeneration

Result wrapper for a single generation from a model call.

```python
class ChatGeneration(Generation):
    message: AIMessage
    generation_info: dict[str, Any] | None = None
```

### ChatGenerationChunk

Chunk of a streamed generation.

```python
class ChatGenerationChunk(Generation):
    message: AIMessageChunk
    generation_info: dict[str, Any] | None = None
```

### ChatResult

Result wrapper for a batch generation.

```python
class ChatResult(NamedTuple):
    generations: list[ChatGeneration]
    llm_output: dict[str, Any] | None = None
    run: list[RunInfo] | None = None
```

### LLMResult

Result wrapper aggregating generations from multiple prompts.

```python
class LLMResult(NamedTuple):
    generations: list[list[ChatGeneration]]  # One list per prompt
    llm_output: dict[str, Any] | None = None
    run: list[RunInfo] | None = None
```

---

## Callback Integration

Chat models fire the following callback lifecycle:

**Synchronous invoke/stream:**
- `on_chat_model_start(serialized, messages_list, ...)`
- `on_llm_new_token(token_str, chunk=ChatGenerationChunk)` (per chunk during streaming)
- `on_llm_end(LLMResult(generations=[[generation]]))` (on success)
- `on_llm_error(exception, response=LLMResult(...))` (on exception)

**Asynchronous ainvoke/astream:**
- `await on_chat_model_start(...)`
- `await on_llm_new_token(...)` (per chunk)
- `await on_llm_end(...)`
- `await on_llm_error(...)`

**Batch generate/agenerate:**
- `on_chat_model_start(serialized, messages_list_for_all_prompts, ...)`
- Per-prompt error handling via `on_llm_error`
- Per-prompt completion via `on_llm_end`

**v3 stream_events:**
- Lazy initialization: callbacks attached on first stream consumption
- `on_stream_event(MessagesData)` for each protocol event
- `on_llm_end` / `on_llm_error` on stream completion or error

---

## Streaming Decision Logic

The model decides whether to stream based on `_should_stream()` and `_should_use_protocol_streaming()`:

**`_should_stream(async_api, run_manager, **kwargs)`:**
- Returns `False` if disabled (via `disable_streaming`, `stream=False`, or `self.streaming=False`)
- Returns `False` if streaming not implemented for requested variant
- Returns `True` if explicit `stream=True`, `self.streaming=True`, or v1 handler attached

**`_should_use_protocol_streaming(async_api, run_manager, **kwargs)`:**
- Returns `False` if no v2 handler (`_V2StreamingCallbackHandler`) attached
- Returns `False` if no streaming source available (native `_stream_chat_model_events` or `_stream` override)
- Returns `False` if disabled (same as `_should_stream`)
- Returns `True` otherwise

---

## Rate Limiting

If `rate_limiter` is configured:
- `stream()` acquires rate limit before entering loop via `rate_limiter.acquire(blocking=True)`
- `astream()` acquires via `await rate_limiter.aacquire(blocking=True)`
- `generate()` and `agenerate()` acquire per-item via `_generate_with_cache()` and `_agenerate_with_cache()`

Acquisition is blocking, so throughput is constrained by limiter policy.

---

## Provider Implementation Guide

To implement a new chat model:

1. **Subclass `BaseChatModel`** and implement required abstract methods

2. **Implement `_generate()`** (required):
   ```python
   def _generate(self, messages, stop=None, run_manager=None, **kwargs):
       # Call provider API
       response = self.client.chat.completions.create(...)
       # Return ChatResult
       return ChatResult(
           generations=[ChatGeneration(
               message=AIMessage(content=response.choices[0].message.content),
               generation_info={...}
           )]
       )
   ```

3. **Set `_llm_type`** property:
   ```python
   @property
   def _llm_type(self) -> str:
       return "provider_name"
   ```

4. **Set `_identifying_params`** property (optional but recommended):
   ```python
   @property
   def _identifying_params(self) -> dict[str, Any]:
       return {"model": self.model, "temperature": self.temperature}
   ```

5. **Implement `_stream()`** (optional):
   ```python
   def _stream(self, messages, stop=None, run_manager=None, **kwargs):
       # Stream from provider API
       stream = self.client.chat.completions.create(..., stream=True)
       for chunk in stream:
           delta = chunk.choices[0].delta
           yield ChatGenerationChunk(
               message=AIMessageChunk(content=delta.content)
           )
   ```

6. **Implement `_astream()`** (optional, async variant)

7. **Implement `bind_tools()`** if tool calling is supported:
   ```python
   def bind_tools(self, tools, *, tool_choice=None, **kwargs):
       # Convert tools to provider format
       converted = [convert_to_provider_tool(t) for t in tools]
       # Return runnable that includes tools in request
       return self.bind(tools=converted, tool_choice=tool_choice, **kwargs)
   ```

8. **Override `_get_ls_params()`** if model identifier name differs:
   ```python
   def _get_ls_params(self, stop=None, **kwargs):
       params = super()._get_ls_params(stop=stop, **kwargs)
       if hasattr(self, "deployment_name"):
           params["ls_model_name"] = self.deployment_name
       return params
   ```

9. **Resolve model profile** (optional, requires provider data):
   ```python
   def _resolve_model_profile(self):
       from .profiles import get_model_profile
       return get_model_profile(self.model)
   ```

---

## See Also

- [Chat Model Interface and Lifecycle](/openwiki/chat-models.md) — Architectural overview
- [Messages](/openwiki/messages.md) — Message types and structures
- [Runnable Protocol](/openwiki/runnables.md) — Runnable base class
- [Callbacks](/openwiki/callbacks.md) — Callback handlers and lifecycle
