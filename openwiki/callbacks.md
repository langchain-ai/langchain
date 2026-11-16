---
type: "Reference"
title: "Callback System and Handler Integration"
description: "Document the callback handler architecture, integration with runnables and chat models, and patterns for tracking execution events, streaming, and instrumentation."
tags: ["callbacks", "observability", "handlers", "tracing", "streaming", "langsmith"]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-16T08:30:18.041Z
sources:
  - id: openwiki-source-c9313cf42f0120d86b20245f
    resource: repo://libs/core/langchain_core/callbacks/base.py
  - id: openwiki-source-b15ceb5ed590ce6a2b569ed6
    resource: repo://libs/core/langchain_core/callbacks/file.py
  - id: openwiki-source-1c233fccf5a66b84d0045366
    resource: repo://libs/core/langchain_core/callbacks/manager.py
  - id: openwiki-source-55497dabc655f803e40dc13a
    resource: repo://libs/core/langchain_core/callbacks/stdout.py
  - id: openwiki-source-c7a2c3ef4ec61c3e28011205
    resource: repo://libs/core/langchain_core/callbacks/streaming_stdout.py
  - id: openwiki-source-2685400b7962e4c90cefe9df
    resource: repo://libs/core/langchain_core/callbacks/usage.py
  - id: openwiki-source-079792f059657900794e2955
    resource: repo://libs/core/langchain_core/runnables/config.py
  - id: openwiki-source-bfd8b1aa6ad00852a2e99762
    resource: repo://libs/core/langchain_core/tracers/context.py
generated: { by: "openwiki/0.5.0", at: "2026-11-16T08:30:18.041Z" }
---

## Overview

LangChain's callback system provides a unified mechanism for observability, logging, and tracing of all operations—LLM calls, chain execution, tool usage, and retrieval. Callbacks allow developers to monitor execution flow, collect metrics, stream output, and integrate with external observability platforms like LangSmith without modifying core application code.

The system is built on a hierarchical run structure where parent-child relationships are tracked through run IDs, enabling comprehensive trace trees. Each operation generates events that are dispatched to one or more registered handlers, which can act on them synchronously or asynchronously.

## Architecture

### Core Components

**BaseCallbackHandler** (`repo://libs/core/langchain_core/callbacks/base.py#L496-L546`) is the base class for all callback implementations. It inherits from multiple mixins that define event methods for different operation types:

- **LLMManagerMixin**: `on_llm_new_token`, `on_llm_end`, `on_llm_error`, `on_stream_event`
- **ChainManagerMixin**: `on_chain_end`, `on_chain_error`, `on_agent_action`, `on_agent_finish`
- **ToolManagerMixin**: `on_tool_end`, `on_tool_error`
- **RetrieverManagerMixin**: `on_retriever_end`, `on_retriever_error`
- **RunManagerMixin**: `on_text`, `on_retry`, `on_custom_event`
- **CallbackManagerMixin**: `on_llm_start`, `on_chat_model_start`, `on_chain_start`, `on_tool_start`, `on_retriever_start`

Every handler also supports `raise_error` and `run_inline` attributes to control error propagation and execution context.

**BaseCallbackManager** (`repo://libs/core/langchain_core/callbacks/base.py#L1004-L1227`) manages a collection of handlers and their lifecycle. It maintains:

- **handlers**: non-inheritable callbacks for the current operation
- **inheritable_handlers**: callbacks passed down to child operations
- **tags**: labels for filtering and organizing runs (inheritable)
- **metadata**: JSON-serializable context (inheritable)
- **parent_run_id**: reference to parent operation for hierarchy

**CallbackManager** (sync, `repo://libs/core/langchain_core/callbacks/manager.py#L1377-L1726`) and **AsyncCallbackManager** (async, `repo://libs/core/langchain_core/callbacks/manager.py#L1859`) are the primary implementations that dispatch events to handlers. They provide `on_llm_start`, `on_chat_model_start`, `on_chain_start`, `on_tool_start`, and `on_retriever_start` methods that return specialized run managers bound to a specific operation.

**Run Managers** are returned from start events and provide context-bound methods for end and error events:

- **CallbackManagerForLLMRun**: `on_llm_new_token`, `on_llm_end`, `on_llm_error`, `on_stream_event`
- **CallbackManagerForChainRun**: `on_chain_end`, `on_chain_error`, `on_agent_action`, `on_agent_finish`
- **CallbackManagerForToolRun**: `on_tool_end`, `on_tool_error`
- **CallbackManagerForRetrieverRun**: `on_retriever_end`, `on_retriever_error`

Each run manager holds the **run_id** and **parent_run_id** for the operation, ensuring trace hierarchy is preserved.

### Event Dispatch

The callback system uses two dispatch functions:

**handle_event** (`repo://libs/core/langchain_core/callbacks/manager.py#L287-L369`, sync) iterates over handlers and calls the corresponding event method. It handles:

- Skipping handlers based on ignore conditions (e.g., `ignore_llm`, `ignore_chain`)
- Catching and logging exceptions (respecting `raise_error`)
- Collecting async coroutines and running them via executor pool or event loop
- Converting `on_chat_model_start` to `on_llm_start` fallback when not implemented

**ahandle_event** (`repo://libs/core/langchain_core/callbacks/manager.py#L448-L474`, async) separates inline (sequential) and non-inline (concurrent) handlers, using `asyncio.gather()` for parallelism.

The **shielded** decorator (`repo://libs/core/langchain_core/callbacks/manager.py#L232-L255`) preserves context variables in async handlers when cancellation occurs, avoiding task cancellation deadlocks.

### Context Variables and Propagation

Context variables ensure callback configuration propagates across sync/async boundaries and thread pool execution:

- **tracing_v2_callback_var** (`repo://libs/core/langchain_core/tracers/context.py#L32-L34`): Holds the active LangChainTracer for automatic tracing
- **run_collector_var** (`repo://libs/core/langchain_core/tracers/context.py#L35-L37`): Tracks collected runs during context manager usage
- **copy_context().run()** is used in thread pool execution to preserve these variables across threads

### Inheritance and Child Managers

When a child operation is created via `ParentRunManager.get_child()` (`repo://libs/core/langchain_core/callbacks/manager.py#L591-L605`), the new manager:

1. Inherits all `inheritable_handlers` from the parent
2. Copies `inheritable_tags` and `inheritable_metadata` to the child's active state
3. Sets its `parent_run_id` to the parent's `run_id`, establishing the trace hierarchy
4. Generates a new unique `run_id` for the child operation

This ensures that metadata and handlers flow down the call tree, while maintaining separate run identities for fine-grained tracing.

## Built-in Handlers

### StdOutCallbackHandler

**Location**: `repo://libs/core/langchain_core/callbacks/stdout.py`

Prints human-readable messages to standard output for chain entry/exit, agent actions, tool observations, and freeform text. Useful for interactive CLI applications.

```python
from langchain_core.callbacks.stdout import StdOutCallbackHandler

handler = StdOutCallbackHandler(color=None)
chain.invoke(input, config={"callbacks": [handler]})
# > Entering new SequentialChain chain...
# > Finished chain.
```

Methods override:
- `on_chain_start`: Prints "Entering new {name} chain"
- `on_chain_end`: Prints "Finished chain"
- `on_agent_action`: Prints action log
- `on_tool_end`: Prints tool observation with optional prefixes
- `on_text`: Prints arbitrary text

### FileCallbackHandler

**Location**: `repo://libs/core/langchain_core/callbacks/file.py`

Writes callback events to a file with optional coloring. Supports both context manager (recommended) and direct instantiation patterns.

```python
from langchain_core.callbacks.file import FileCallbackHandler

# Context manager (recommended)
with FileCallbackHandler("output.txt") as handler:
    chain.invoke(input, config={"callbacks": [handler]})

# Direct instantiation (deprecated)
handler = FileCallbackHandler("output.txt")
try:
    chain.invoke(input, config={"callbacks": [handler]})
finally:
    handler.close()
```

The handler opens a file in append mode by default and writes formatted output for chains, agents, and tools.

### StreamingStdOutCallbackHandler

**Location**: `repo://libs/core/langchain_core/callbacks/streaming_stdout.py`

Streams individual LLM tokens to stdout as they are generated during streaming, enabling real-time output visualization.

```python
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

handler = StreamingStdOutCallbackHandler()
llm.stream("Hello", config={"callbacks": [handler]})  # Prints tokens as they arrive
```

Only works with LLMs that support streaming via `on_llm_new_token`.

### UsageMetadataCallbackHandler

**Location**: `repo://libs/core/langchain_core/callbacks/usage.py`

Aggregates token usage metadata across LLM calls, collecting input/output tokens and cost information keyed by model name. Thread-safe using internal locks.

```python
from langchain_core.callbacks.usage import UsageMetadataCallbackHandler

callback = UsageMetadataCallbackHandler()
llm_1.invoke("Hello", config={"callbacks": [callback]})
llm_2.invoke("Hello", config={"callbacks": [callback]})
print(callback.usage_metadata)
# {"openai:gpt-4": UsageMetadata(...), "anthropic:claude-3": UsageMetadata(...)}
```

## LangSmith Integration

LangSmith is the observability platform for LangChain. When enabled via environment variables or context managers, all callback events are automatically sent to LangSmith for visualization and analysis.

### Automatic Tracing

**Enable with environment variables**:
```bash
export LANGCHAIN_TRACING_V2=true
export LANGSMITH_API_KEY=<your-key>
export LANGSMITH_PROJECT=<project-name>
```

When `LANGCHAIN_TRACING_V2` is enabled, the callback manager automatically creates a **LangChainTracer** (`repo://libs/core/langchain_core/tracers/langchain.py`) and registers it as a handler. This tracer:

- Captures all run lifecycle events (start, end, error)
- Builds a hierarchical trace tree using parent_run_id
- Sends traces to the LangSmith backend
- Provides run URLs and unique run IDs

**Context manager approach**:

```python
from langchain_core.tracers.context import tracing_v2_enabled

with tracing_v2_enabled(project_name="my_project") as tracer:
    chain.invoke("hello")
    run_url = tracer.get_run_url()
    print(f"View trace at: {run_url}")
```

**Automatic trace callback injection** (`repo://libs/core/langchain_core/tracers/context.py#L113-L138`) happens via `_get_trace_callbacks`, which:

1. Checks if tracing is enabled via `_tracing_v2_is_enabled()`
2. Creates or reuses an existing LangChainTracer
3. Adds it to the callback manager without duplication
4. Returns the configured callback manager for use in operations

## Configuration and Usage Patterns

### Via RunnableConfig

Callbacks are configured through **RunnableConfig** (`repo://libs/core/langchain_core/runnables/config.py#L57-L129`), which has the following callback-related fields:

```python
class RunnableConfig(TypedDict, total=False):
    callbacks: Callbacks          # Callback handlers for this and child operations
    tags: list[str]              # Inheritable tags for filtering runs
    metadata: dict[str, Any]     # Inheritable context metadata
    run_name: str                # Named run for tracing
    run_id: uuid.UUID | None     # Explicit run ID
```

Tags and metadata are automatically propagated to nested operations, enabling hierarchical filtering and context enrichment.

### Via CallbackManager.configure()

The **CallbackManager.configure()** class method (`repo://libs/core/langchain_core/callbacks/manager.py#L1669-L1712`) provides explicit control over callback registration:

```python
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.stdout import StdOutCallbackHandler

manager = CallbackManager.configure(
    inheritable_callbacks=[StdOutCallbackHandler()],
    inheritable_tags=["my_tag"],
    inheritable_metadata={"context": "value"}
)

chain.invoke(input, config={"callbacks": manager})
```

Parameters:
- **inheritable_callbacks**: Callbacks passed to child operations
- **local_callbacks**: Callbacks for current operation only
- **inheritable_tags**: Tags inherited by child operations
- **local_tags**: Tags for current operation only
- **inheritable_metadata**: Metadata inherited by child operations
- **local_metadata**: Metadata for current operation only

### Chain Group Tracing

**trace_as_chain_group** and **atrace_as_chain_group** (`repo://libs/core/langchain_core/callbacks/manager.py#L57-L226`) group multiple separate calls as a single logical operation:

```python
from langchain_core.callbacks.manager import trace_as_chain_group

with trace_as_chain_group("my_group", inputs={"key": "value"}) as manager:
    result = llm.invoke("prompt", config={"callbacks": manager})
    manager.on_chain_end({"output": result})
```

This is useful for combining unrelated operations (e.g., multiple API calls) into a unified trace for better observability.

## Handler Execution Control

### run_inline Attribute

The `run_inline` attribute on handlers controls execution context (`repo://libs/core/langchain_core/callbacks/base.py#L509-L510`):

- **run_inline=True**: Handler executes sequentially in the caller's context, blocking until complete. Useful for synchronous operations or when handler state must be immediately visible.
- **run_inline=False** (default): Handler executes concurrently via thread pool (sync) or `asyncio.gather()` (async). Better for I/O-bound operations like sending traces to remote servers.

```python
class MyHandler(BaseCallbackHandler):
    run_inline = True  # Execute sequentially in caller context
```

### raise_error Attribute

The `raise_error` attribute controls exception propagation (`repo://libs/core/langchain_core/callbacks/base.py#L506-L507`):

- **raise_error=False** (default): Handler exceptions are logged and swallowed, preventing callback failures from breaking the application.
- **raise_error=True**: Handler exceptions propagate to the caller.

```python
class StrictHandler(BaseCallbackHandler):
    raise_error = True  # Propagate handler errors
```

### Ignore Conditions

Handlers can opt out of specific event types via ignore properties (`repo://libs/core/langchain_core/callbacks/base.py#L512-L545`):

```python
class SelectiveHandler(BaseCallbackHandler):
    @property
    def ignore_llm(self) -> bool:
        return True  # Skip all LLM events
    
    @property
    def ignore_chain(self) -> bool:
        return False  # Process chain events
```

Available ignore properties:
- `ignore_llm`: Skip LLM events
- `ignore_chat_model`: Skip chat model events
- `ignore_chain`: Skip chain events
- `ignore_agent`: Skip agent events
- `ignore_retriever`: Skip retriever events
- `ignore_retry`: Skip retry events
- `ignore_custom_event`: Skip custom events

## Custom Events

Custom events allow applications to dispatch user-defined observability signals:

```python
from langchain_core.callbacks.manager import CallbackManager

manager = CallbackManager(handlers=[my_handler])
manager.on_custom_event("my_event", {"data": "value"}, run_id=run_id)
```

Handlers receive custom events via `on_custom_event(name, data, **kwargs)` (`repo://libs/core/langchain_core/callbacks/manager.py#L1625-L1667`).

## Implementing Custom Handlers

Custom handlers are implemented by subclassing **BaseCallbackHandler** and overriding relevant event methods:

```python
from langchain_core.callbacks.base import BaseCallbackHandler
from typing import Any
from uuid import UUID

class MyCustomHandler(BaseCallbackHandler):
    """Custom callback handler for application-specific events."""
    
    # Control execution context
    run_inline = False  # Execute asynchronously by default
    raise_error = False  # Don't break on errors
    
    # Opt out of irrelevant events
    @property
    def ignore_retriever(self) -> bool:
        return True
    
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM starts execution."""
        print(f"LLM started with {len(prompts)} prompts")
    
    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM finishes."""
        print(f"LLM completed")
    
    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any] | Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a chain starts."""
        if isinstance(inputs, dict):
            print(f"Chain started with keys: {list(inputs.keys())}")
```

**Key implementation patterns**:

1. **Use **kwargs for forward compatibility**: Event signatures may add fields in future versions; always accept `**kwargs`.
2. **Set run_inline=False for I/O operations**: Avoid blocking the main execution path.
3. **Use ignore_* properties for selective filtering**: Reduce callback overhead for events you don't need.
4. **Thread-safe state for async contexts**: Use locks when maintaining shared state (as in **UsageMetadataCallbackHandler**).

## Event Streaming and Real-Time Monitoring

The callback system supports real-time event streaming for monitoring:

- **on_llm_new_token**: Called for each token during LLM streaming
- **on_stream_event**: Called for structured events from the LangChain protocol (version "v3")
- **on_text**: Called for arbitrary text output

```python
class TokenCollectorHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []
    
    def on_llm_new_token(
        self,
        token: str | list[str | dict[str, Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if isinstance(token, str):
            self.tokens.append(token)

handler = TokenCollectorHandler()
result = llm.stream("Hello", config={"callbacks": [handler]})
print("".join(handler.tokens))
```

## Monitoring and Instrumentation

The callback system integrates with environment variables for global configuration:

- **LANGCHAIN_TRACING_V2**: Enable automatic LangSmith tracing (set to "true")
- **LANGSMITH_API_KEY**: API credentials for LangSmith
- **LANGSMITH_PROJECT**: Default project for traces
- **LANGCHAIN_DEBUG**: Enable debug mode (affects logging verbosity)

Debug mode can be queried at runtime:

```python
from langchain_core.globals import get_debug

if get_debug():
    print("Debug mode is enabled")
```

Run managers support error tracking via exception callbacks:

```python
try:
    result = chain.invoke(input)
except Exception as e:
    run_manager.on_error(e)  # Reports to tracing system
    raise
```

## Async/Sync Integration

The callback system transparently handles both sync and async contexts:

- **CallbackManager**: Used in synchronous code
- **AsyncCallbackManager**: Used in async code

Both are instantiated the same way, and their event dispatch is context-aware:

```python
# Synchronous usage
chain.invoke(input, config={"callbacks": [handler]})

# Asynchronous usage
await async_chain.ainvoke(input, config={"callbacks": [handler]})
```

The dispatcher automatically:
- Runs coroutines via executor or event loop (sync path)
- Separates inline and non-inline handlers (async path)
- Preserves context variables across thread/task boundaries
- Falls back from `on_chat_model_start` to `on_llm_start` when not implemented
