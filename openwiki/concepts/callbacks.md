---
type: Concept
title: "Callbacks: Instrumentation & Observability"
description: "Callbacks are an event-driven instrumentation system that enables monitoring, tracing, and custom logic in LangChain applications. Handler classes respond to component lifecycle events (start/end/error), managers coordinate dispatch and context propagation (tags, metadata, run IDs), and built-in handlers integrate with LangSmith for production tracing."
tags: [callbacks, observability, tracing, langsmith, run_id, instrumentation]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-18T08:17:15.497Z
sources:
  - id: openwiki-source-96c73d2c05223b0b46abdbe9
    resource: repo://libs/core/langchain_core/callbacks/__init__.py
  - id: openwiki-source-c9313cf42f0120d86b20245f
    resource: repo://libs/core/langchain_core/callbacks/base.py
  - id: openwiki-source-b15ceb5ed590ce6a2b569ed6
    resource: repo://libs/core/langchain_core/callbacks/file.py
  - id: openwiki-source-1c233fccf5a66b84d0045366
    resource: repo://libs/core/langchain_core/callbacks/manager.py
  - id: openwiki-source-55497dabc655f803e40dc13a
    resource: repo://libs/core/langchain_core/callbacks/stdout.py
  - id: openwiki-source-2685400b7962e4c90cefe9df
    resource: repo://libs/core/langchain_core/callbacks/usage.py
  - id: openwiki-source-bfd8b1aa6ad00852a2e99762
    resource: repo://libs/core/langchain_core/tracers/context.py
generated: { by: "openwiki/0.5.0", at: "2026-11-18T08:17:15.497Z" }
---

## Overview

LangChain's callback system is an event-driven instrumentation layer that enables monitoring, tracing, and debugging of LLM applications. Callbacks fire at key points in the execution lifecycle of LLMs, chains, tools, and retrievers, providing access to inputs, outputs, and errors for observability and custom logic.

The callback system consists of:

- **Event handlers** that respond to execution events
- **Manager patterns** that coordinate handlers and propagate context (tags, metadata, run IDs)
- **Built-in integrations** with LangSmith for production tracing
- **Custom event support** for application-specific instrumentation

## Core Concepts

### Event Lifecycle and Callbacks

Every callable component in LangChain (LLM, chain, tool, retriever) generates a sequence of callback events:

| Component | Start Event | End Event | Error Event | Streaming Event |
|-----------|------------|----------|------------|-----------------|
| **LLM** (text) | `on_llm_start` | `on_llm_end` | `on_llm_error` | `on_llm_new_token` |
| **Chat Model** | `on_chat_model_start` | `on_llm_end` | `on_llm_error` | `on_llm_new_token` |
| **Chain** | `on_chain_start` | `on_chain_end` | `on_chain_error` | — |
| **Tool** | `on_tool_start` | `on_tool_end` | `on_tool_error` | — |
| **Retriever** | `on_retriever_start` | `on_retriever_end` | `on_retriever_error` | — |
| **Agent** | `on_chain_start` | `on_chain_end` | `on_chain_error` | `on_agent_action`, `on_agent_finish` |

Chat models also emit `on_stream_event` for v3 protocol events when using `stream_events(version="v3")` or `astream_events(version="v3")`.

### BaseCallbackHandler

The `BaseCallbackHandler` is the core interface for implementing custom callbacks. It defines optional methods for each event type, organized into mixins by component category.

```python
from langchain_core.callbacks import BaseCallbackHandler
from uuid import UUID
from typing import Any

class MyCallbackHandler(BaseCallbackHandler):
    """Custom callback handler."""
    
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
        """Called when an LLM starts."""
        print(f"LLM starting with prompt: {prompts[0][:50]}...")
    
    def on_llm_end(self, response, **kwargs: Any) -> None:
        """Called when an LLM finishes."""
        print(f"LLM finished with response tokens: {len(response.generations[0])}")
    
    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Called when an LLM errors."""
        print(f"LLM error: {type(error).__name__}")
```

**Key handler properties:**

- `raise_error` (bool, default=False): Whether to raise exceptions from the handler instead of swallowing them
- `run_inline` (bool, default=False): Whether to run the handler synchronously in the calling thread (vs. thread pool for async handlers)
- `ignore_*` properties: Filter callbacks for specific component types (e.g., `ignore_llm`, `ignore_chain`, `ignore_agent`, `ignore_retriever`, `ignore_chat_model`, `ignore_retry`, `ignore_custom_event`)

### Callback Event Parameters

All callback events receive common parameters:

- **`run_id`** (UUID): Unique identifier for the current execution
- **`parent_run_id`** (UUID | None): ID of the parent run in a nested execution tree
- **`tags`** (list[str] | None): Tags propagated from parent runs for filtering and metadata
- **`metadata`** (dict[str, Any] | None): Custom metadata propagated through the execution tree

Example with metadata:

```python
from langchain_core.callbacks import BaseCallbackHandler

class DebugHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        metadata = kwargs.get("metadata", {})
        print(f"Chain: {serialized.get('name')}")
        print(f"Metadata: {metadata}")
        print(f"Tags: {kwargs.get('tags', [])}")
```

## Callback Manager Patterns

Callback managers coordinate handlers and manage context propagation. They enforce the start/end/error lifecycle and return specialized run managers for nested callbacks.

### CallbackManager (Synchronous)

The `CallbackManager` is the synchronous coordinator. It dispatches events to all registered handlers and returns specialized run managers for different component types.

```python
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager
from langchain_core.llms import LLM

handler = BaseCallbackHandler()
manager = CallbackManager(handlers=[handler])

# Create an LLM run
llm_managers = manager.on_llm_start(
    serialized={"name": "my-llm"},
    prompts=["What is AI?"],
    tags=["user-query"]
)

# Each returned manager tracks that specific LLM run
for llm_manager in llm_managers:
    # ... LLM execution ...
    llm_manager.on_llm_end(response)
```

**Return types from `CallbackManager` start methods:**

- `on_llm_start()` → `list[CallbackManagerForLLMRun]` (one per prompt)
- `on_chat_model_start()` → `list[CallbackManagerForLLMRun]` (one per message list)
- `on_chain_start()` → `CallbackManagerForChainRun` (single manager for chain)
- `on_tool_start()` → `CallbackManagerForToolRun` (single manager for tool)
- `on_retriever_start()` → `CallbackManagerForRetrieverRun` (single manager for retriever)

### AsyncCallbackManager

The `AsyncCallbackManager` is the async variant. It dispatches events concurrently to multiple handlers and returns async run managers.

```python
from langchain_core.callbacks import AsyncCallbackManager
import asyncio

async def main():
    manager = AsyncCallbackManager(handlers=[handler])
    
    # Async dispatch with concurrency
    llm_managers = await manager.on_chat_model_start(
        serialized={"name": "gpt-4"},
        messages=[[message1, message2]],
        tags=["chat"]
    )
    
    for mgr in llm_managers:
        # ... async LLM execution ...
        await mgr.on_llm_end(response)

asyncio.run(main())
```

**Async handler execution:**
- Async methods (coroutine functions) are awaited naturally
- Sync methods with `run_inline=True` are called directly in the event loop
- Sync methods with `run_inline=False` (default) are run in a thread pool executor to avoid blocking the event loop

### Run Managers

Run managers are bound to a specific execution run (LLM call, chain invocation, etc.). They extend their corresponding handlers and propagate context to child runs.

**ParentRunManager** (base for chains, tools, retrievers):

```python
class CallbackManagerForChainRun(ParentRunManager, ChainManagerMixin):
    def get_child(self, tag: str | None = None) -> CallbackManager:
        """Create a child callback manager inheriting tags and handlers."""
        # Returns a CallbackManager with inheritable_handlers and inheritable_tags
```

**Run manager hierarchy:**

```
BaseRunManager
├── RunManager (sync)
│   └── ParentRunManager
│       ├── CallbackManagerForChainRun
│       ├── CallbackManagerForToolRun
│       └── CallbackManagerForRetrieverRun
└── AsyncRunManager (async)
    └── AsyncParentRunManager
        ├── AsyncCallbackManagerForChainRun
        ├── AsyncCallbackManagerForToolRun
        └── AsyncCallbackManagerForRetrieverRun
```

## Handler and Context Composition

### Inheritable Handlers and Tags

Context is propagated through nested execution via "inheritable" handlers and tags:

```python
manager = CallbackManager(
    handlers=[debug_handler],           # Non-inheritable (this run only)
    inheritable_handlers=[tracer],      # Inherited by child runs
    tags=["debug"],                     # Non-inheritable tags
    inheritable_tags=["user:alice"],    # Inherited tags
    metadata={"version": "1.0"},        # Non-inheritable metadata
    inheritable_metadata={"user": "alice"}  # Inherited metadata
)

# When on_chain_start is called, it returns a CallbackManagerForChainRun
chain_manager = manager.on_chain_start(...)

# The child manager created by chain_manager.get_child() inherits:
child = chain_manager.get_child(tag="step1")
# child has empty handlers (inheritable_handlers not added to handlers)
# child.inheritable_handlers = [tracer]
# child.inheritable_tags = ["user:alice", "step1"]
# child.inheritable_metadata = {"user": "alice"}
```

### Sync Handlers in Async Contexts

When async handlers are invoked from a synchronous `handle_event` call (e.g., in a sync LLM), LangChain collects coroutines and executes them in a thread pool to avoid blocking:

```python
# In sync handle_event, if a handler has async methods:
coros = []
for handler in handlers:
    event = getattr(handler, event_name)(*args, **kwargs)
    if asyncio.iscoroutine(event):  # Async handler
        coros.append(event)

# Later, run collected coroutines in a thread pool
if coros:
    _executor().submit(_run_coros, coros).result()  # Blocks until async work completes
```

This allows sync components to work with async handlers without deadlock.

### on_chat_model_start Fallback

When `on_chat_model_start` raises `NotImplementedError`, the callback system automatically falls back to `on_llm_start` with message lists converted to prompt strings:

```python
class MyHandler(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        raise NotImplementedError  # Triggers fallback
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        # Called with prompts: [get_buffer_string(m) for m in messages]
        pass
```

This fallback applies to both sync `handle_event` and async `ahandle_event` contexts. **Important**: When implementing `on_chat_model_start`, always use explicit `serialized` and `messages` parameters; avoid `*args` to prevent `IndexError` during the fallback conversion.

### Handler Filtering

Handlers can ignore specific event types via boolean properties:

```python
class SelectiveHandler(BaseCallbackHandler):
    @property
    def ignore_llm(self) -> bool:
        return True  # Skip all LLM events
    
    @property
    def ignore_chain(self) -> bool:
        return False  # Process chain events
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        print("Chain event received")
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        # Never called due to ignore_llm=True
        pass
```

Available ignore properties: `ignore_llm`, `ignore_chat_model`, `ignore_chain`, `ignore_agent`, `ignore_retriever`, `ignore_retry`, `ignore_custom_event`.

### Event Dispatch and Error Handling

The `handle_event` function dispatches events to handlers with careful error handling:

```python
def handle_event(
    handlers: list[BaseCallbackHandler],
    event_name: str,
    ignore_condition_name: str | None,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Dispatch an event to all handlers."""
    for handler in handlers:
        try:
            if ignore_condition_name and getattr(handler, ignore_condition_name):
                continue  # Skip this handler
            
            event = getattr(handler, event_name)(*args, **kwargs)
            if asyncio.iscoroutine(event):
                coros.append(event)  # Collect async work
        except NotImplementedError:
            # Handle fallback (e.g., on_chat_model_start -> on_llm_start)
            pass
        except Exception as e:
            logger.warning(f"Error in {handler}.{event_name}: {e}")
            if handler.raise_error:
                raise
```

**Key behaviors:**
- Exceptions are logged and swallowed unless `handler.raise_error=True`
- Exceptions in async handlers run through sync `handle_event` are logged but never propagated (pre-existing asymmetry in callback error handling)
- Exceptions in sync handlers run through async `ahandle_event` honor `raise_error` and propagate when True

### Merging Callback Managers

Managers can be merged to combine handlers and tags:

```python
from langchain_core.callbacks.manager import CallbackManager

manager1 = CallbackManager(handlers=[handler1], tags=["tag1"])
manager2 = CallbackManager(handlers=[handler2], tags=["tag2"])

merged = manager1.merge(manager2)
# merged.handlers = [handler1, handler2]  (deduplicated via set union)
# merged.tags = ["tag1", "tag2"]  (deduplicated via set union)
```

## Built-in Handlers

### StdOutCallbackHandler

Prints chain and agent execution to stdout:

```python
from langchain_core.callbacks import StdOutCallbackHandler

handler = StdOutCallbackHandler(color="green")
# Prints formatted output for chain start/end and agent actions
```

### StreamingStdOutCallbackHandler

Prints LLM tokens to stdout as they arrive (streaming):

```python
from langchain_core.callbacks import StreamingStdOutCallbackHandler

handler = StreamingStdOutCallbackHandler()
# Each token from on_llm_new_token is printed immediately
```

### FileCallbackHandler

Writes chain execution output to a file:

```python
from langchain_core.callbacks import FileCallbackHandler

# Using context manager (recommended)
with FileCallbackHandler("trace.log") as handler:
    chain.invoke(inputs, config={"callbacks": [handler]})

# Direct instantiation (deprecated)
handler = FileCallbackHandler("trace.log", mode="a")
try:
    chain.invoke(inputs, config={"callbacks": [handler]})
finally:
    handler.close()
```

The handler writes formatted output for chain and tool execution, similar to `StdOutCallbackHandler` but to a file.

### UsageMetadataCallbackHandler

Tracks token usage across LLM calls:

```python
from langchain_core.callbacks import UsageMetadataCallbackHandler, get_usage_metadata_callback

with get_usage_metadata_callback() as callback:
    result = llm.invoke("Hello", config={"callbacks": [callback]})
    print(callback.usage_metadata)
    # {"gpt-4": UsageMetadata(input_tokens=2, output_tokens=10)}
```

## LangSmith Integration

LangChain's callback system integrates with LangSmith for production tracing, debugging, and evaluation.

### Automatic Tracing

Enable automatic tracing to LangSmith via environment variables:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your_langsmith_api_key>
export LANGCHAIN_PROJECT=my-project
```

When `LANGCHAIN_TRACING_V2` is set to `true`, a `LangChainTracer` callback handler is automatically added to all executions, sending traces to the configured LangSmith project.

### Manual LangChainTracer

For explicit control, use `tracing_v2_enabled()`:

```python
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.runnables import RunnablePassthrough

with tracing_v2_enabled(project_name="my-project") as tracer:
    result = chain.invoke("input")
    run_url = tracer.get_run_url()
    print(f"View trace: {run_url}")
```

The `tracing_v2_enabled()` context manager returns a `LangChainTracer` instance that can be used to fetch the run URL for accessing the trace in LangSmith.

### Collecting Runs Locally

Use `collect_runs()` to gather traces in memory for testing:

```python
from langchain_core.tracers.context import collect_runs

with collect_runs() as collector:
    result = chain.invoke("input")
    
    for run in collector.traced_runs:
        print(f"Run: {run.name}, Inputs: {run.inputs}")
```

### LangChainTracer Configuration

The `LangChainTracer` accepts:

- **`project_name`**: LangSmith project to send traces to
- **`example_id`**: Associate traces with a dataset example
- **`tags`**: Tags to apply to all traced runs
- **`metadata`**: Metadata for run filtering and querying
- **`address`**: Agent address for agent addressing (beta feature)

## Callback Lifecycle and Execution Order

### Run Lifecycle

Every execution follows a start→end or start→error pattern:

1. **on_*_start**: Called when a component begins execution with initialization parameters
2. **on_*_new_token** (LLM/chat only): Called for each token during streaming
3. **on_stream_event** (v3 events only): Called for each protocol event in v3 streaming
4. **on_*_end** or **on_*_error**: Called when execution completes or fails

For agents, additional events fire:

```
on_chain_start (agent)
├── on_tool_start
├── on_tool_end
├── on_agent_action
│
└── on_agent_finish
on_chain_end or on_chain_error
```

### Nested Execution Context

When a parent component calls children, context flows through the callback hierarchy:

```
Parent run (run_id=A)
├── handlers: [debug_handler]
├── inheritable_handlers: [tracer]
├── on_chain_start → fires at parent
│
└─ Child run (parent_run_id=A, run_id=B)
   ├── handlers: []  (empty, inheritable handlers not copied)
   ├── inheritable_handlers: [tracer]  (inherited)
   ├── on_chain_start → fires at child with parent_run_id=A
   │
   └─ Grandchild run (parent_run_id=B, run_id=C)
      ├── on_chain_start → fires with parent_run_id=B
      └── [tracer] still observes via inheritable_handlers
```

Run IDs and parent relationships establish the execution tree visible in tracing systems.

### Handler Ordering

Handlers are called in iteration order from the handlers list:

```python
# Multiple handlers execute in sequence
handlers = [handler1, handler2, handler3]
for handler in handlers:
    # Call handler.on_llm_start(...) in order
```

An exception in one handler is caught, logged, and execution continues with the next (unless `raise_error=True`).

## Custom Events

Applications can dispatch custom events for application-specific instrumentation via `dispatch_custom_event` (sync) or `adispatch_custom_event` (async):

```python
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_core.runnables.config import RunnableConfig

@RunnableLambda
def my_function(x: int, config: RunnableConfig):
    dispatch_custom_event(
        "user_action",
        {"action": "login", "user_id": 123},
        config=config
    )
    return x * 2
```

Custom event handlers implement `on_custom_event`:

```python
class CustomEventHandler(BaseCallbackHandler):
    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        print(f"Event {name}: {data}")
```

Async dispatch:

```python
from langchain_core.callbacks.manager import adispatch_custom_event

await adispatch_custom_event("async_event", {"data": "value"}, config=config)
```

**Custom event contract:**
- `name`: Identifier for the event type
- `data`: Arbitrary event data (must be serializable for LangSmith tracing)
- Receives `run_id`, `tags`, and `metadata` from the current execution context
- Handlers can filter by checking `ignore_custom_event` property
- Custom events require a parent run ID and must be called from within an active run (tool, RunnableLambda, etc.)

## Configuring Callbacks

### Via RunnableConfig

Runnables accept callbacks in their config:

```python
from langchain_core.runnables import RunnableLambda

chain = RunnableLambda(lambda x: x.upper())

result = chain.invoke(
    "hello",
    config={
        "callbacks": [handler1, handler2],
        "tags": ["production"],
        "metadata": {"user_id": "alice"}
    }
)
```

### CallbackManager Construction and Configuration

**Direct construction:**

```python
from langchain_core.callbacks import CallbackManager, BaseCallbackHandler

manager = CallbackManager(
    handlers=[debug_handler],           # Handlers for this run only
    inheritable_handlers=[tracer],      # Propagated to child runs
    tags=["local"],
    inheritable_tags=["app"],
    metadata={"session": "123"},
    inheritable_metadata={"user": "alice"},
    parent_run_id=parent_uuid
)
```

**Class method `configure()` for flexible setup:**

```python
from langchain_core.callbacks import CallbackManager

manager = CallbackManager.configure(
    inheritable_callbacks=[tracer],
    local_callbacks=[debug_handler],
    inheritable_tags=["system"],
    local_tags=["this-run"],
    inheritable_metadata={"version": "1.0"},
    local_metadata={"request_id": "456"},
    verbose=True
)
```

**Key behaviors:**
- `configure()` aggregates callbacks from multiple sources with environment variable integration
- When `verbose=True`, a `StdOutCallbackHandler` is automatically added for chain execution tracing
- Inheritable callbacks/tags/metadata flow to child runs; local ones apply only to the current run
- `langsmith_inheritable_metadata` and `langsmith_inheritable_tags` are applied to `LangChainTracer` handlers

### Context Variables

Callbacks can access context via the implicit `RunnableConfig`:

```python
from langchain_core.runnables import RunnableLambda, RunnableConfig

@RunnableLambda
async def fetch_context(x: str, config: RunnableConfig):
    callbacks = config.get("callbacks")
    run_id = config.get("run_id")
    tags = config.get("tags", [])
    metadata = config.get("metadata", {})
    return x.upper()
```

## Callback Integration with Runnables

### Config Threading Through _call_with_config

Callbacks flow through runnables via `_call_with_config`, which:

1. **Extracts the callback manager** from the RunnableConfig via `get_callback_manager_for_config(config)`
2. **Fires on_chain_start** to notify all handlers of the execution beginning
3. **Creates a child manager** via `run_manager.get_child()` that inherits tags and inheritable handlers
4. **Patches the config** with the child manager as callbacks for nested runnables
5. **Passes the patched config** to nested runnables so they inherit observability context
6. **Fires on_chain_end or on_chain_error** to close the run and propagate results

The child manager created by `get_child()` has:
- Empty handlers list (inheritable_handlers not propagated as handlers)
- Inheritable handlers preserved for further nesting
- Inherited tags plus any tag passed to `get_child(tag="...")`
- Inherited metadata

This design ensures:
- Inheritable handlers reach all descendants
- Non-inheritable handlers apply only to their current run
- Each level of nesting creates a new run with distinct callbacks

**Key insight**: Every child runnable gets a fresh CallbackManager configured from the child config, enabling automatic context propagation without explicit handler passing.

### Fallback for Chat Models

When a handler doesn't implement `on_chat_model_start`, the system falls back to `on_llm_start` with prompts converted from messages:

```python
def on_chat_model_start(self, serialized, messages, **kwargs):
    raise NotImplementedError  # Triggers fallback to on_llm_start
```

The conversion uses `get_buffer_string()` to transform message objects into plain text prompts, ensuring backward compatibility with handlers built for text LLMs.

## Tracing Nested Executions

### Automatic Parent-Child Tracking

Run managers track parent-child relationships:

```python
# Top-level execution
chain_manager = callback_manager.on_chain_start(...)

# chain_manager.run_id = <uuid-a>
# chain_manager.parent_run_id = None (or parent's UUID)

# Child execution
child_manager = chain_manager.get_child(tag="step1")

# Callbacks see: parent_run_id=<uuid-a>
tool_manager = child_manager.on_tool_start(...)
# tool_manager.parent_run_id = <uuid-a>
```

### Grouping Runs: trace_as_chain_group

`trace_as_chain_group` context manager groups unrelated calls as a single logical chain:

```python
from langchain_core.callbacks.manager import trace_as_chain_group

with trace_as_chain_group(
    "multi_step_process",
    inputs={"start": "value"},
    tags=["important"],
) as group_manager:
    result1 = llm1.invoke("prompt1", config={"callbacks": group_manager})
    result2 = llm2.invoke("prompt2", config={"callbacks": group_manager})
    
    group_manager.on_chain_end({"results": [result1, result2]})
```

In LangSmith, all calls within the group appear as children of a single parent run.
