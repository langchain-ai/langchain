---
type: "Concept"
title: "Streaming: Token-by-Token Output"
description: "How streaming works across LLM components and chains, token-by-token delivery via AIMessageChunk, callback integration, and memory/latency tradeoffs."
tags: [streaming, token-streaming, llm-output, chat-models, callbacks, astream, real-time-feedback]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:18:34.589Z
sources:
  - id: openwiki-source-c9313cf42f0120d86b20245f
    resource: repo://libs/core/langchain_core/callbacks/base.py
  - id: openwiki-source-c7a2c3ef4ec61c3e28011205
    resource: repo://libs/core/langchain_core/callbacks/streaming_stdout.py
  - id: openwiki-source-5f8bc32563177d89fbab9b2f
    resource: repo://libs/core/langchain_core/language_models/chat_model_stream.py
  - id: openwiki-source-c52037e7b642f7ac5a7642a8
    resource: repo://libs/core/langchain_core/language_models/chat_models.py
  - id: openwiki-source-77dc1fb726463969f9d53658
    resource: repo://libs/core/langchain_core/messages/ai.py
  - id: openwiki-source-a1981e868973f6fd7f71e12e
    resource: repo://libs/core/langchain_core/runnables/base.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:18:34.589Z" }
---

## Overview

**Streaming** is the mechanism by which LangChain delivers model output incrementally, token by token, rather than waiting for the entire response. This enables real-time feedback in web UIs, console displays, and other user-facing contexts, and forms the foundation for building responsive applications that do not block on model latency.

Instead of blocking with `invoke()` until a full response is ready, applications call `stream()` or `astream()` and receive a sequence of partial outputs as they arrive from the model. Each chunk is an `AIMessageChunk` carrying delta content. Callbacks intercept these chunks via the `on_llm_new_token` event, making it possible to observe, log, or react to each token without collecting the entire response first.

Streaming flows through chains—prompts, models, output parsers, and other runnables—preserving incremental output delivery at each stage. By composition, a chain automatically supports streaming if all its components do. This page documents the mechanics of streaming across components, the trade-offs versus non-streaming invoke, and how to integrate streaming into applications.

## Synchronous Streaming: stream()

**Location**: `repo://libs/core/langchain_core/language_models/chat_models.py#L727-L856`

`BaseChatModel.stream()` is the primary synchronous streaming entry point. It yields `AIMessageChunk` objects as they are produced by the underlying model, with incremental content—a single token, a fragment of JSON, or a structured block update.

### Control Flow

1. **Check if streaming is implemented**: `_should_stream()` determines whether the model supports streaming. If not, `stream()` falls back to `invoke()` and yields one complete result.

2. **Initialize callbacks**: A `CallbackManager` is configured from the provided `RunnableConfig`, binding callbacks, tags, and metadata.

3. **Fire on_chat_model_start**: The callback lifecycle begins with `on_chat_model_start`, signaling that LLM invocation is beginning.

4. **Iterate model chunks**: For each `ChatGenerationChunk` from the underlying `_stream()` implementation:
   - The chunk's message ID is set to a unique run ID if not already present.
   - Response metadata (model provider, latency, etc.) is computed and attached.
   - **on_llm_new_token is fired** with the chunk's content and the full chunk object, allowing callbacks to observe or buffer each token.
   - The chunk message is cast to `AIMessageChunk` and yielded immediately.
   - Chunks are accumulated for later aggregation.

5. **Yield final "last" chunk**: After the model finishes, if output_version is v1 (content-block format), an empty chunk with `chunk_position="last"` is yielded. This signals to parsers and consumers that the stream is complete and that tool_call_chunks should be finalized.

6. **Callback lifecycle closes**: If successful, `on_llm_end` fires with a merged `ChatGeneration` containing all chunks. If an exception occurs, `on_llm_error` fires with partial accumulation.

### Fallback Behavior

If the model does not implement streaming (checked via `_should_stream(async_api=False)`), `stream()` delegates to `invoke()` and yields a single result cast to `AIMessageChunk`. This ensures all models provide a consistent streaming interface, even if only non-streaming invoke is available.

### Rate Limiting

If a rate limiter is attached to the model, `stream()` acquires a permit before beginning, blocking until the rate limit allows.

## Asynchronous Streaming: astream()

**Location**: `repo://libs/core/langchain_core/language_models/chat_models.py#L858-L991`

`BaseChatModel.astream()` is the async variant of `stream()`, mirroring the synchronous logic but using async/await and `AsyncCallbackManager`.

**Key differences**:
- Uses `await` for callback events (`await run_manager.on_llm_new_token(...)`, `await run_manager.on_llm_end(...)`)
- Iterates via `async for chunk in self._astream(...)`
- Acquires rate limit via `await self.rate_limiter.aacquire(blocking=True)`

The async streaming protocol is identical to sync: yield chunks immediately, fire callbacks per token, finalize tool call chunks on the "last" signal.

## AIMessageChunk: Incremental Content

**Location**: `repo://libs/core/langchain_core/messages/ai.py#L418-L536`

`AIMessageChunk` is the message type yielded during streaming. Unlike `AIMessage`, it represents a **partial, incremental update** to a conversation message and supports merging via the `+` operator.

### Structure

- **content**: String or list of content blocks. During streaming, each chunk contains only the new token(s) or delta for that step.
- **tool_call_chunks**: List of `ToolCallChunk` objects (incomplete tool calls being streamed). These are progressively updated as arguments arrive.
- **chunk_position**: Optional sentinel; when set to `"last"`, indicates the final chunk in the stream, triggering finalization of tool calls and reasoning blocks.
- **response_metadata**: Model-specific metadata (latency, model_provider, usage counters, etc.) attached by the streaming handler.

### Merging and Aggregation

Streaming chunks accumulate via the `+` operator, which merges content, concatenates tool_call arguments, and combines metadata. A complete `AIMessage` with finalized `tool_calls` (not chunks) is reconstructed when chunks are merged or when the "last" signal is received.

## Callback Integration: on_llm_new_token

**Location**: `repo://libs/core/langchain_core/callbacks/base.py#L65-L88`

The `on_llm_new_token` callback fires for each token or chunk during streaming, enabling real-time observation and logging.

### Signature

```python
def on_llm_new_token(
    self,
    token: str | list[str | dict[str, Any]],
    *,
    chunk: GenerationChunk | ChatGenerationChunk | None = None,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> Any:
```

- **token**: The string token or list of content blocks (when output_version="v1").
- **chunk**: The full `ChatGenerationChunk` carrying metadata, message ID, response metadata, and tool_call_chunks.
- **run_id**: Unique identifier for this streaming run, used for tracing and correlation.

### Example: Stream to stdout

```python
from langchain_core.callbacks import StreamingStdOutCallbackHandler

callback = StreamingStdOutCallbackHandler()

# Callbacks are passed via RunnableConfig
for chunk in model.stream(
    messages,
    config=RunnableConfig(callbacks=[callback])
):
    pass  # callback prints each token to stdout
```

The `StreamingStdOutCallbackHandler` implements `on_llm_new_token` to write tokens to `sys.stdout`, making streaming output visible in real-time.

## Streaming Through Chains

Streaming flows through chains composed of runnables (prompts, models, parsers). The streaming protocol is implemented at each stage via the `stream()` and `transform()` methods on `Runnable`.

### Default Behavior

**Location**: `repo://libs/core/langchain_core/runnables/base.py#L1194-L1235`

By default, `Runnable.stream()` yields one full output from `invoke()`. Subclasses that support streaming override `stream()` or `transform()` to yield chunks.

### Streaming through RunnableSequence

**Location**: `repo://libs/core/langchain_core/runnables/base.py#L3075-L3320`

`RunnableSequence` (a chain created with the `|` operator) automatically supports streaming if:
1. **All upstream components implement transform**: The `transform()` method maps streaming input to streaming output.
2. **The last component produces chunks**: Output parsers and models implement `transform()` to yield partial results.

If any component does not implement `transform()`, streaming begins only after that component completes (blocking point). Multiple blocking components create multiple buffering points, but the final output still streams from the last component if it supports streaming.

### Streaming Example: Model → Parser

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI()
parser = StrOutputParser()
chain = model | parser

# stream yields parser outputs incrementally as tokens arrive
for chunk in chain.stream("What is 2+2?"):
    print(chunk, end="", flush=True)
```

When `model.stream()` yields chunks, the parser's `transform()` (or default `stream()`) consumes each chunk and yields its transformation. Text parsers may yield tokens directly; JSON parsers yield partial JSON objects as they become parseable.

## Streaming via stream_events: ChatModelStream

**Location**: `repo://libs/core/langchain_core/language_models/chat_model_stream.py`

For advanced use cases requiring detailed event granularity, `BaseChatModel.stream_events(version="v3")` returns a `ChatModelStream` object that exposes **typed projection properties** (`.text`, `.tool_calls`, `.usage`, `.reasoning`, `.output`) which accumulate events as they arrive.

This is distinct from simple token streaming and is useful for applications needing structured, event-by-event visibility into reasoning, tool calls, and other protocol events. The `ChatModelStream` also fires `on_stream_event` callbacks for each protocol event, not just tokens.

## Memory and Latency Trade-offs: stream() vs invoke()

### invoke()

- **Latency**: Waits for the entire model response before returning.
- **Memory**: No intermediate storage required; only the final message is held.
- **Responsiveness**: Blocks the calling thread/coroutine until complete.
- **Use case**: Batch processing, when a complete response is needed upfront.

### stream()

- **Latency**: Yields the first token as soon as available; responsive to user.
- **Memory**: Requires buffering of accumulated chunks if the caller collects them.
- **Responsiveness**: Non-blocking; enables progressive display.
- **Use case**: Web UIs, console applications, user-facing interactions where real-time feedback improves UX.

In practice, streaming does not add significant latency compared to invoke; the model produces tokens at the same rate. The difference is **when tokens are delivered to the caller**. Stream delivery is preferable for interactive applications because users see output appearing in real-time rather than a blank screen until the full response is ready.

## Integration Patterns

### Real-time Console Output

```python
from langchain_core.callbacks import StreamingStdOutCallbackHandler

callback = StreamingStdOutCallbackHandler()
for _ in model.stream(
    messages,
    config=RunnableConfig(callbacks=[callback])
):
    pass  # Tokens are printed as they arrive
```

### Accumulate Streamed Output

```python
result = ""
for chunk in model.stream(messages):
    result += chunk.content or ""
print(result)  # Final complete response
```

### Custom Callback for Application Logic

```python
from langchain_core.callbacks import BaseCallbackHandler

class MyCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        # React to each token (e.g., update UI, log, rate-limit)
        self.buffer.append(token)

for _ in model.stream(
    messages,
    config=RunnableConfig(callbacks=[MyCallback()])
):
    pass
```

### Async Streaming in Web Framework

```python
async def chat_endpoint(messages):
    async for chunk in model.astream(messages):
        # Yield to HTTP client as server-sent event
        yield f"data: {chunk.content}\n\n"
```

## Lifecycle and Error Handling

### Successful Stream

1. `on_chat_model_start` fires
2. For each chunk: `on_llm_new_token` fires
3. `on_llm_end` fires with merged `ChatGeneration`

### Stream with Error

1. `on_chat_model_start` fires
2. For each chunk before error: `on_llm_new_token` fires
3. Error occurs in `_stream()` or callback
4. `on_llm_error` fires with partial chunks aggregated
5. Exception is re-raised to caller

### Cleanup

When a stream exits (via break, exception, or normal completion), any buffered chunks are merged and callbacks finalize the run. Async streaming also closes async generators via `aclose()` if present.

## Extension Points

### Custom Streaming Implementation

Subclasses of `BaseChatModel` override `_stream()` and/or `_astream()` to implement model-specific streaming:

```python
class MyModel(BaseChatModel):
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # Yield ChatGenerationChunk for each token
        for token in model_api.stream(messages, stop=stop, **kwargs):
            yield ChatGenerationChunk(message=AIMessageChunk(content=token))
```

The `stream()` method handles callbacks, merging, and lifecycle; subclasses only implement the core streaming loop.

### Custom Output Parser Transform

Output parsers can override `transform()` to stream partial results:

```python
class MyParser(BaseGenerationOutputParser[T]):
    def transform(
        self,
        input: Iterator[str | BaseMessage],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[T]:
        buffer = ""
        for chunk in input:
            buffer += chunk.content or ""
            # Attempt partial parsing
            if partial := self.parse_result([Generation(text=buffer)], partial=True):
                yield partial
```

This allows parsers to yield progressively more complete results as tokens arrive.

## Configuration and Operations

### Disabling Streaming

Models respect the `stream=False` parameter or a falsy check in `_should_stream()`. Calling `invoke()` directly bypasses streaming even if the model supports it.

### Configuring Callbacks

```python
config = RunnableConfig(
    callbacks=[StreamingStdOutCallbackHandler()],
    tags=["user-interaction"],
    metadata={"session_id": "..."},
)
for chunk in model.stream(messages, config=config):
    pass
```

Callbacks, tags, and metadata propagate through the callback lifecycle.

### Async Streaming

Use `astream()` in async contexts and `await` on async callbacks:

```python
async for chunk in model.astream(messages):
    # Process chunks asynchronously
    await handle_chunk(chunk)
```

## Conclusion

Streaming is central to building responsive LangChain applications. By yielding output token-by-token and firing callbacks per token, streaming enables real-time user feedback without sacrificing performance. The protocol is consistent across models, chains, and parsers, making it easy to compose streaming operations and observe output at any level of the application stack.
