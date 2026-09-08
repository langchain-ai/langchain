---
type: "Architecture"
title: "Chat Model Interface and Lifecycle"
description: "Document BaseChatModel protocol, input/output handling, streaming, and integration points with callbacks and model profiling."
tags: [chat-models, llm-integration, streaming, structured-output, model-capabilities]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:18:34.589Z
sources:
  - id: openwiki-source-132f3183693cd9cf79d029a5
    resource: repo://libs/core/langchain_core/language_models/base.py
  - id: openwiki-source-5f8bc32563177d89fbab9b2f
    resource: repo://libs/core/langchain_core/language_models/chat_model_stream.py
  - id: openwiki-source-c52037e7b642f7ac5a7642a8
    resource: repo://libs/core/langchain_core/language_models/chat_models.py
  - id: openwiki-source-a0aef6917b7e1f4a06e6db95
    resource: repo://libs/core/langchain_core/language_models/model_profile.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:18:34.589Z" }
---

## Overview

The **chat model system** is the core interface for integrating large language models into LangChain applications. `BaseChatModel` is the abstract protocol that all chat model implementations inherit from. It defines the contract for synchronous and asynchronous invoke/streaming behavior, callback integration, rate limiting, structured output binding, and capability discovery via model profiles.

Chat models convert conversational message history into AI responses, supporting both simple generation (`invoke`) and streaming output (`stream`). The framework unifies sync/async patterns, handles caching transparently, routes to streaming or non-streaming backends based on configuration and attached callbacks, and provides extension points for custom behavior via method overrides.

## Core Interface: BaseChatModel

**Location**: `repo://libs/core/langchain_core/language_models/chat_models.py#L284-L2400`

`BaseChatModel` inherits from `BaseLanguageModel[AIMessage]` and is a `Runnable` that accepts `LanguageModelInput` and produces `AIMessage` outputs. It is designed for subclassing; implementations must override `_generate` (required) and optionally `_llm_type`, `_stream`, and `_agenerate`.

### Input and Output Types

**LanguageModelInput** (`repo://libs/core/langchain_core/language_models/base.py#L140`) is a union type:

```python
LanguageModelInput = PromptValue | str | Sequence[MessageLikeRepresentation]
```

- **string**: Converted to a `StringPromptValue` (simple user message)
- **list of messages**: Converted to a `ChatPromptValue` (full conversation history)
- **PromptValue**: Already a structured prompt (passed through)

The `_convert_input` method normalizes all input forms to a `PromptValue` for downstream processing.

**Output**: All invoke/stream methods return `AIMessage` or `AIMessageChunk` (for streaming). Chat results are wrapped in `ChatGeneration` objects (holding message + generation metadata) aggregated into `ChatResult`.

### Synchronous Methods

**`invoke`** (`repo://libs/core/langchain_core/language_models/chat_models.py#L474-L499`) is the primary synchronous entrypoint:

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

- Converts input to `PromptValue`, then to messages
- Calls `generate_prompt` (which internally calls `_generate_with_cache`)
- Extracts and returns the first generation's message
- Propagates `run_id`, callbacks, tags, and metadata from config

**`stream`** (`repo://libs/core/langchain_core/language_models/chat_models.py#L726-L856`) yields `AIMessageChunk` objects as they arrive:

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

- Checks if streaming is enabled and implemented via `_should_stream()`
- Falls back to `invoke` if streaming is disabled or not implemented
- For streaming-enabled models, calls `_stream()` directly and yields chunks
- Wraps output in callback lifecycle: `on_chat_model_start`, `on_llm_new_token` (per chunk), `on_llm_end` or `on_llm_error`
- Applies rate limiting if configured
- Normalizes messages and handles streaming-specific output formatting (e.g., `output_version="v1"`)
- Yields a final empty chunk with `chunk_position="last"` when streaming completes

### Asynchronous Methods

**`ainvoke`** (`repo://libs/core/langchain_core/language_models/chat_models.py#L501-L523`) is the async variant:

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

- Awaits `agenerate_prompt`
- Otherwise mirrors `invoke` behavior

**`astream`** (`repo://libs/core/langchain_core/language_models/chat_models.py#L857-L990`) is the async streaming variant:

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

- Checks `_should_stream(async_api=True)` to route to `_astream` or fallback
- Otherwise mirrors `stream` behavior with async callback dispatch

## Streaming Architecture

### Stream Decision Logic

**`_should_stream()`** (`repo://libs/core/langchain_core/language_models/chat_models.py#L549-L585`) determines whether to use the streaming code path:

```python
def _should_stream(
    self,
    *,
    async_api: bool,
    run_manager: CallbackManagerForLLMRun | AsyncCallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> bool
```

Returns `True` if:
1. Streaming is not disabled (`_streaming_disabled()` returns `False`)
2. Streaming method is implemented for the requested variant (sync/async)
3. Any of these are true:
   - Explicit `stream=True` kwarg
   - Instance-level `streaming=True` attribute
   - A v1-style `_StreamingCallbackHandler` is attached

Returns `False` (fallback to non-streaming) if:
- `disable_streaming=True` (hard disable)
- `disable_streaming="tool_calling"` and tools are provided
- `stream=False` explicitly
- Streaming is not implemented and async falls back to sync

### Stream Implementation Methods

**`_stream()`** (`repo://libs/core/langchain_core/language_models/chat_models.py#L2255-L2273`) is the sync streaming hook (optional override):

```python
def _stream(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    run_manager: CallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> Iterator[ChatGenerationChunk]
```

- Subclasses override to implement native streaming
- Default raises `NotImplementedError` (fallback to `_generate`)
- Receives run_manager for per-token callbacks

**`_astream()`** (`repo://libs/core/langchain_core/language_models/chat_models.py#L2275-L2311`) is the async streaming hook (optional override):

```python
async def _astream(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    run_manager: AsyncCallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> AsyncIterator[ChatGenerationChunk]
```

- Default implementation runs `_stream()` in an executor and yields results
- Subclasses can override for native async streaming

### ChatModelStream and AsyncChatModelStream

**Location**: `repo://libs/core/langchain_core/language_models/chat_model_stream.py`

For the v3 event protocol (`stream_events(version="v3")`), models return a `ChatModelStream` (sync) or `AsyncChatModelStream` (async) that expose **typed projections** for incremental content:

- **`.text`**: Accumulates text content blocks
- **`.reasoning`**: Accumulates reasoning/chain-of-thought content
- **`.tool_calls`**: Accumulates parsed tool call blocks
- **`.usage`**: Accumulates token usage info
- **`.output`**: Final assembled `AIMessage`

Each projection can be iterated for deltas or awaited for the final value. Internally, these accumulators track incoming protocol events and merge them into structured output.

## Generation and Caching

### Core Generation Methods

**`_generate()`** (`repo://libs/core/langchain_core/language_models/chat_models.py#L2208-L2226`) is the **required abstract method** all subclasses must implement:

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

- Calls the underlying model's API
- Returns a `ChatResult` with a list of `ChatGeneration` objects
- Must handle errors internally or propagate them
- Receives normalized messages and a run manager for callbacks

**`_agenerate()`** (`repo://libs/core/langchain_core/language_models/chat_models.py#L2228-L2253`) is the optional async override:

```python
async def _agenerate(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    run_manager: AsyncCallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> ChatResult
```

- Default implementation runs `_generate` in an executor
- Subclasses override for native async API support

### Cached Generation

**`_generate_with_cache()`** and **`_agenerate_with_cache()`** wrap the core methods with:

1. **Prompt caching**: Checks if `self.cache` or global `get_llm_cache()` has cached results for the input
2. **Cache hits**: Returns cached generations and replays them as v2 events if a v2 handler is attached
3. **Cache misses**: Routes through streaming or non-streaming path
4. **Protocol routing**: Dispatches to v2 events (`_should_use_protocol_streaming`) or v1 callback path (`_should_stream`)

### Batch Methods

**`generate()`** and **`agenerate()`** accept a list of message lists and use internal caching/streaming to batch-process prompts:

```python
def generate(
    self,
    messages: list[list[BaseMessage]],
    stop: list[str] | None = None,
    callbacks: Callbacks = None,
    **kwargs: Any,
) -> LLMResult
```

Returns an `LLMResult` with generations grouped by input prompt and combined llm_output.

## Callback Lifecycle

Chat models integrate with the callback system to emit structured events throughout execution:

### LLM Run Lifecycle

1. **`on_chat_model_start`** (or fallback `on_llm_start`):
   - Fires when `invoke`, `stream`, or `generate` begins
   - Receives serialized model config, formatted input messages, invocation params, and batch size
   - Returns run manager(s) bound to the operation
   
2. **`on_llm_new_token`** (streaming only):
   - Fires once per streamed token/chunk
   - Receives token string and `ChatGenerationChunk` metadata
   - Allows real-time output capture

3. **`on_llm_end`**:
   - Fires when generation completes successfully
   - Receives final `LLMResult` with all generations and metadata

4. **`on_llm_error`**:
   - Fires if generation raises an exception
   - Receives the exception and partial `LLMResult` (if available)
   - `_generate_response_from_error()` extracts response metadata from HTTP errors

5. **`on_stream_event`** (v2/v3 protocol):
   - Fires for each content-block protocol event during streaming
   - Allows fine-grained event observation for advanced tracing

### Callback Configuration

Callbacks are configured via `RunnableConfig`:

```python
config = {
    "callbacks": [my_handler],  # Callbacks for this run
    "tags": ["agent", "tools"],  # Labels for filtering
    "metadata": {"user_id": "123"},  # Context data
    "run_name": "my_run",  # Human-readable run name
    "run_id": uuid.uuid4(),  # Explicit run ID (optional)
}
result = model.invoke(input, config=config)
```

Inheritable metadata and LangSmith params are extracted via `_get_invocation_params()` and `_get_ls_params()`.

## Structured Output and Tool Binding

### with_structured_output()

**Location**: `repo://libs/core/langchain_core/language_models/chat_models.py#L2385-L2565`

`with_structured_output()` wraps a chat model to constrain output to a specified schema:

```python
def with_structured_output(
    self,
    schema: dict[str, Any] | type,
    *,
    include_raw: bool = False,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]
```

**How it works**:

1. Delegates to `bind_tools([schema], tool_choice="any", ...)`
2. Chains the result through an output parser:
   - If schema is a Pydantic class: `PydanticToolsParser` → Pydantic instance
   - If schema is a dict: `JsonOutputKeyToolsParser` → dict
3. If `include_raw=True`: Wraps output in `{"raw": AIMessage, "parsed": ..., "parsing_error": ...}`
4. If parsing fails and `include_raw=False`: Raises exception

**Prerequisites**: Requires the model to implement `bind_tools()` (not all models support this).

### bind_tools()

**Location**: `repo://libs/core/langchain_core/language_models/chat_models.py#L2366-L2383`

```python
def bind_tools(
    self,
    tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
    *,
    tool_choice: str | None = None,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, AIMessage]
```

- Abstract method; must be implemented by subclasses that support tool calling
- Binds a list of tools to the model
- Returns a bound runnable that includes tool definitions in the API request
- `tool_choice="any"` forces the model to call at least one tool

## Model Profiles and Capabilities

**Location**: `repo://libs/core/langchain_core/language_models/model_profile.py`

The `profile` field on `BaseChatModel` holds metadata about model capabilities:

```python
class ModelProfile(TypedDict, total=False):
    # Metadata
    name: str  # Human-readable model name
    status: str  # 'active', 'deprecated', etc.
    release_date: str  # ISO 8601
    last_updated: str  # ISO 8601
    open_weights: bool  # Weights publicly available?
    
    # Input constraints
    max_input_tokens: int  # Context window size
    text_inputs: bool
    image_inputs: bool
    image_url_inputs: bool
    pdf_inputs: bool
    audio_inputs: bool
    video_inputs: bool
    image_tool_message: bool  # Images in ToolMessage?
    pdf_tool_message: bool  # PDFs in ToolMessage?
    
    # Output constraints
    max_output_tokens: int
    text_outputs: bool
    image_outputs: bool
    audio_outputs: bool
    video_outputs: bool
    
    # Capabilities
    tool_calling: bool  # Supports function calling?
    tool_choice: bool  # Supports tool_choice parameter?
    tool_call_streaming: bool  # Returns structured tool_call_chunks when streaming?
    structured_output: bool  # Native structured output support?
    reasoning_output: bool  # Reasoning/chain-of-thought?
    reasoning_effort_levels: list[str]  # ['low', 'medium', 'high']
    reasoning_effort_default: str
    temperature: bool  # Supports temperature parameter?
    attachment: bool  # Supports file attachments?
```

**Auto-loading**: Profiles are resolved via `_resolve_model_profile()` (subclass override) and cached in the `profile` field. Unrecognized keys trigger a warning via `_warn_unknown_profile_keys()`.

### Partner Pattern Integration

Partner packages (e.g., `langchain-openai`) override `_resolve_model_profile()` to load model-specific metadata from their own profile data. The base validator `_set_model_profile` (Pydantic mode="after") automatically populates the field if not explicitly set.

## Configuration and State

### Core Fields

```python
class BaseChatModel(BaseLanguageModel[AIMessage], ABC):
    rate_limiter: BaseRateLimiter | None = Field(default=None, exclude=True)
    
    disable_streaming: bool | Literal["tool_calling"] = False
    # False: use streaming if available
    # True: always use non-streaming (invoke)
    # "tool_calling": use non-streaming only when tools are passed
    
    output_version: str | None = None
    # 'v0': provider-specific format (lazy-parse via content_blocks)
    # 'v1': standardized format (merged into content)
    
    profile: ModelProfile | None = Field(default=None, exclude=True)
    # Capability metadata (auto-loaded if not provided)
    
    cache: BaseCache | None = None  # Inherited from BaseLanguageModel
    callbacks: list[BaseCallbackHandler] | None = None
    verbose: bool = False
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None
```

### Required Properties

- **`_llm_type`** (property, abstract): Unique model type identifier (e.g., `"openai"`, `"anthropic"`)
- **`_identifying_params`** (property, optional): Dict of model configuration for tracing (e.g., `{"model": "gpt-4", "temperature": 0.7}`)

## Implementation Requirements

Subclasses must implement:

| Method/Property | Description | Required | Notes |
|---|---|---|---|
| `_generate()` | Core generation logic | ✓ | Calls provider API, returns `ChatResult` |
| `_llm_type` | Model type identifier | ✓ | String like `"openai"`, `"anthropic"` |
| `_identifying_params` | Config dict for tracing | ✗ | Used by `_get_llm_string()` and serialization |
| `_stream()` | Sync streaming | ✗ | Optional; if not implemented, stream falls back to invoke |
| `_agenerate()` | Native async generation | ✗ | Optional; defaults to running `_generate` in executor |
| `_astream()` | Native async streaming | ✗ | Optional; defaults to running `_stream` in executor |
| `bind_tools()` | Tool binding for structured output | ✗ | Required only if `with_structured_output()` is needed |

## Model Initialization

**Location**: `repo://libs/langchain_v1/langchain/chat_models/base.py` (v1 compat) and `langchain_core` partner packages

Models are instantiated via:

1. **Direct instantiation**: `ChatOpenAI(model="gpt-4", temperature=0)`
2. **Factory function `init_chat_model()`**: Auto-detects provider and imports the class dynamically
3. **Partner package exports**: Each provider (e.g., `langchain-openai`) exports a concrete model class

The `init_chat_model()` function accepts a model name string (e.g., `"gpt-4"`, `"claude-3-sonnet"`) and optional `model_provider` to instantiate the correct class without explicit imports.

## Example: Custom Chat Model

```python
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun

class MyCustomChatModel(BaseChatModel):
    """Custom chat model for demonstration."""
    
    model_name: str = "my-model"
    temperature: float = 0.7
    
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from the messages."""
        # Call your model API here
        response_text = f"Echo: {messages[-1].content}"
        
        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream tokens from the model."""
        text = f"Echo: {messages[-1].content}"
        for char in text:
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=char)
            )
            yield chunk
    
    @property
    def _llm_type(self) -> str:
        """Return the model type identifier."""
        return "my-custom-model"
    
    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Return identifying parameters for tracing."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
        }
```

## Advanced Patterns

### Streaming with Callbacks

```python
from langchain_core.callbacks import StreamingStdOutCallbackHandler

handler = StreamingStdOutCallbackHandler()
config = {"callbacks": [handler]}

# Streams token-by-token to stdout
for chunk in model.stream("Tell me a joke", config=config):
    pass  # Handler prints as chunks arrive
```

### Structured Output with Validation

```python
from pydantic import BaseModel

class Answer(BaseModel):
    text: str
    confidence: float

structured_model = model.with_structured_output(Answer)
result = structured_model.invoke("What is 2+2?")  # -> Answer(text="4", confidence=0.99)
```

### Caching and Rate Limiting

```python
from langchain_core.caches import InMemoryCache
from langchain_core.rate_limiters import InMemoryRateLimiter

model = ChatOpenAI(
    model="gpt-4",
    cache=InMemoryCache(),  # Cache results
    rate_limiter=InMemoryRateLimiter(requests_per_second=10)  # Limit requests
)

# Subsequent identical calls hit the cache
result1 = model.invoke("Hello")
result2 = model.invoke("Hello")  # Cached, no API call
```

### Conditional Streaming

```python
model_with_fallback = ChatOpenAI().with_fallbacks([ChatAnthropic()])

# Use streaming only when a handler requests it
config = {"callbacks": [MyStreamingHandler()]}
model_with_fallback.invoke("Prompt", config=config)
```

## Key Invariants and Guarantees

1. **Input normalization**: All input forms (string, message list, PromptValue) are normalized to messages before `_generate`/`_stream` are called.

2. **Message IDs**: Each streamed message chunk and final message gets a unique ID (derived from run_id) for tracing.

3. **Callback ordering**: Callbacks fire in order: `on_chat_model_start` → `on_llm_new_token` (per chunk) → `on_llm_end` or `on_llm_error`.

4. **Streaming fallback**: If streaming is not implemented or disabled, `stream` seamlessly falls back to `invoke` and yields the result as a single chunk.

5. **Cache transparency**: Cache hits are completely transparent—same lifecycle callbacks fire as for cache misses.

6. **Async/sync equivalence**: Async methods mirror sync behavior; default async implementations run sync methods in an executor.

7. **Response metadata**: Each generation accumulates metadata (tokens, finish_reason, etc.) in `message.response_metadata`.

8. **Error handling**: Exceptions during generation trigger `on_llm_error` and propagate to the caller; error metadata is extracted from HTTP responses if available.
