---
type: Concept
title: "Callbacks: Event Handler Architecture"
description: "Callbacks are an event-driven architecture for monitoring, tracing, and instrumenting LangChain applications. Handlers respond to component lifecycle events (start/end/error), managers coordinate dispatch and context propagation (tags, metadata, run IDs), and integration with LangSmith enables production tracing."
tags: [callbacks, observability, tracing, instrumentation, event-driven, langsmith]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-07T08:15:07.286Z
sources:
  - id: openwiki-source-96c73d2c05223b0b46abdbe9
    resource: repo://libs/core/langchain_core/callbacks/__init__.py
  - id: openwiki-source-c9313cf42f0120d86b20245f
    resource: repo://libs/core/langchain_core/callbacks/base.py
  - id: openwiki-source-1c233fccf5a66b84d0045366
    resource: repo://libs/core/langchain_core/callbacks/manager.py
  - id: openwiki-source-55497dabc655f803e40dc13a
    resource: repo://libs/core/langchain_core/callbacks/stdout.py
  - id: openwiki-source-2685400b7962e4c90cefe9df
    resource: repo://libs/core/langchain_core/callbacks/usage.py
  - id: openwiki-source-bfd8b1aa6ad00852a2e99762
    resource: repo://libs/core/langchain_core/tracers/context.py
  - id: openwiki-source-79eb321fa44e679f85ad5c48
    resource: repo://libs/core/langchain_core/tracers/langchain.py
generated: { by: "openwiki/0.5.0", at: "2026-11-07T08:15:07.286Z" }
---

## Overview

LangChain's callback system is an event-driven instrumentation architecture that enables monitoring, tracing, debugging, and custom logic in LLM applications. Callbacks fire at key points in the execution lifecycle of LLMs, chat models, chains, tools, retrievers, and agents, providing access to inputs, outputs, and errors for observability and extensibility.

The callback system provides three primary concerns:

- **Event handlers** that respond to component lifecycle events via mixin-based interfaces
- **Manager patterns** that coordinate handler dispatch, context propagation (tags, metadata, run IDs), and nested execution hierarchies
- **Built-in integrations** with LangSmith for production tracing and custom event support for application-specific instrumentation

## Core Concepts

### Event Lifecycle

Every callable component in LangChain generates a sequence of callback events during execution:

| Component | Start | End | Error | Streaming |
|-----------|-------|-----|-------|-----------|
| **LLM** (text) | `on_llm_start` | `on_llm_end` | `on_llm_error` | `on_llm_new_token` |
| **Chat Model** | `on_chat_model_start` | `on_llm_end` | `on_llm_error` | `on_llm_new_token`, `on_stream_event` |
| **Chain** | `on_chain_start` | `on_chain_end` | `on_chain_error` | — |
| **Tool** | `on_tool_start` | `on_tool_end` | `on_tool_error` | — |
| **Retriever** | `on_retriever_start` | `on_retriever_end` | `on_retriever_error` | — |
| **Agent** | `on_chain_start` (action) | `on_chain_end` | `on_chain_error` | `on_agent_action`, `on_agent_finish` |

All events carry contextual parameters: `run_id` (unique execution identifier), `parent_run_id` (parent in execution tree), `tags` (propagated metadata for filtering), and `metadata` (custom execution context).

### BaseCallbackHandler

The `BaseCallbackHandler` is the core interface for implementing custom callbacks. It combines mixins (`LLMManagerMixin`, `ChainManagerMixin`, `ToolManagerMixin`, `RetrieverManagerMixin`, `CallbackManagerMixin`, `RunManagerMixin`) that define optional event methods for each component type.

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
        print(f"LLM: {serialized.get('name')}, Tags: {tags}")
    
    def on_llm_end(self, response, **kwargs: Any) -> None:
        """Called when an LLM finishes."""
        print(f"LLM finished with {len(response.generations[0])} generations")
    
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Called when a chain starts."""
        print(f"Chain: {serialized.get('name')}")
```

**Handler properties for lifecycle control:**

- `raise_error` (bool, default=False): Whether to raise exceptions from the handler instead of logging them
- `run_inline` (bool, default=False): Whether to run handlers synchronously in the calling thread (for async handlers in async contexts; default executes async handlers concurrently via `asyncio.gather`)
- `ignore_*` properties: Conditionally skip event types (`ignore_llm`, `ignore_chain`, `ignore_retriever`, etc.)

For async handlers that may be called from sync code, use `AsyncCallbackHandler` and set `run_inline=True` to execute synchronously, or leave it False to use a thread pool executor.

### Callback Event Parameters

All callback events propagate execution context:

- **`run_id`** (UUID): Unique identifier for the current execution run
- **`parent_run_id`** (UUID | None): ID of the parent run in nested execution trees
- **`tags`** (list[str] | None): Tags propagated from parent runs for filtering, correlation, and tracing
- **`metadata`** (dict[str, Any] | None): Custom metadata propagated through nested execution for observability

These parameters enable tracing frameworks to construct execution graphs, filter spans by tags, and associate arbitrary metadata with runs.

### Chat Model Fallback Mechanism

When a handler's `on_chat_model_start` raises `NotImplementedError`, the callback system automatically falls back to `on_llm_start` with message lists converted to prompt strings. This allows handlers to support both chat and legacy text-completion LLMs through a single interface. Handlers implementing `on_chat_model_start` must declare it explicitly without `*args` to avoid IndexError in the fallback.

## Callback Manager Patterns

Callback managers (`CallbackManager`, `AsyncCallbackManager`) coordinate handler dispatch, manage context (handlers, tags, metadata), and return specialized run managers for nested callbacks. They enforce the start/end/error lifecycle and propagate inheritable context to child runs.

### CallbackManager (Synchronous)

The synchronous manager dispatches events to handlers and returns bound run managers:

```python
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager
from uuid import UUID

handler = BaseCallbackHandler()
manager = CallbackManager(handlers=[handler])

# Start an LLM run
llm_managers = manager.on_llm_start(
    serialized={"name": "my-llm"},
    prompts=["What is AI?"],
    tags=["user-query"],
    metadata={"user_id": "123"}
)

# Each manager tracks one execution
for llm_manager in llm_managers:
    # ... LLM execution ...
    llm_manager.on_llm_end(response)
```

**Return types from start methods:**

- `on_llm_start()` → `list[CallbackManagerForLLMRun]` (one per prompt)
- `on_chat_model_start()` → `list[CallbackManagerForLLMRun]` (one per message list)
- `on_chain_start()` → `CallbackManagerForChainRun` (single manager for chain)
- `on_tool_start()` → `CallbackManagerForToolRun` (single manager for tool)
- `on_retriever_start()` → `CallbackManagerForRetrieverRun` (single manager for retriever)

### AsyncCallbackManager

The async variant dispatches events concurrently to multiple handlers, respecting handler `run_inline` settings:

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

Handlers with `run_inline=True` execute sequentially; those with `run_inline=False` are gathered concurrently. This enables mixing blocking and non-blocking instrumentation in async contexts.

### Run Managers

Run managers bind to a specific execution run and extend their corresponding handler mixins. They propagate context to child runs via `get_child()`.

**Parent Run Managers** (`CallbackManagerForChainRun`, etc.) support creating child managers:

```python
# In a chain or tool, create a child manager that inherits context
child_manager = parent_run_manager.get_child(tag="step_1")

# Child inherits inheritable_handlers, inheritable_tags, and inheritable_metadata
# New tags can be added with tag parameter (non-inheritable)
```

**Run Manager Hierarchy:**

```
BaseRunManager
├── RunManager (sync)
│   └── ParentRunManager
│       ├── CallbackManagerForChainRun
│       ├── CallbackManagerForToolRun
│       └── CallbackManagerForRetrieverRun
└── AsyncRunManager (abstract)
    └── AsyncParentRunManager
        ├── AsyncCallbackManagerForChainRun
        ├── AsyncCallbackManagerForToolRun
        └── AsyncCallbackManagerForRetrieverRun
```

## Event Dispatch Mechanics

### Synchronous Event Dispatch (`handle_event`)

The `handle_event` function dispatches events to handlers, with special handling for mixed sync/async handlers:

1. **Iterate handlers**: For each handler, invoke the event method (`on_llm_start`, etc.)
2. **Detect coroutines**: If a handler method returns a coroutine (async handler called from sync code), collect it
3. **Fallback on NotImplementedError**: For `on_chat_model_start`, convert messages to prompts and call `on_llm_start`
4. **Execute coroutines**: If no event loop is running, execute collected coroutines in a new event loop. If a loop is running, schedule them on a background thread pool to avoid deadlock.

This mechanism enables async handlers (e.g., `UsageMetadataCallbackHandler`) to work transparently in sync contexts.

### Asynchronous Event Dispatch (`ahandle_event`)

The async dispatcher honors `run_inline` settings:

1. **Execute inline handlers sequentially** (awaited immediately)
2. **Gather non-inline handlers concurrently** (via `asyncio.gather`)
3. **Fallback on NotImplementedError**: Same message-to-prompt conversion for `on_chat_model_start`

This allows sync-blocking handlers to run sequentially without blocking async operations.

## Context Management and Grouping

### trace_as_chain_group / atrace_as_chain_group

These context managers group unrelated LangChain calls as children of a parent run, enabling logical grouping without modifying component code:

```python
from langchain_core.callbacks.manager import trace_as_chain_group

with trace_as_chain_group(
    "multi_step_process",
    inputs={"query": "..."},
    tags=["important"],
    metadata={"user": "alice"}
) as manager:
    # Multiple unrelated calls grouped together
    result1 = llm1.invoke(prompt1, {"callbacks": manager})
    result2 = tool.invoke(input2, {"callbacks": manager})
    manager.on_chain_end({"results": [result1, result2]})
```

The manager provides a `parent_run_manager` that tracks the group, enabling clean error handling and explicit end-of-group semantics. Runs are visible in LangSmith as children of the group.

### CallbackManager.configure()

The `configure` class method aggregates callbacks, tags, and metadata from multiple sources with environment variable integration:

```python
manager = CallbackManager.configure(
    inheritable_callbacks=[tracer],
    local_callbacks=[debug_handler],
    verbose=True,  # Adds StdOutCallbackHandler if True
    inheritable_tags=["production"],
    local_tags=["request_123"],
    inheritable_metadata={"version": "1.0"},
)
```

This method is used by runnables to construct callback managers from config, enabling declarative instrumentation through the runnable invocation chain.

## Built-in Handlers

LangChain provides production-ready handlers for common instrumentation patterns:

### StdOutCallbackHandler

Prints execution events to stdout with formatting. Useful for debugging and development.

```python
from langchain_core.callbacks import StdOutCallbackHandler

handler = StdOutCallbackHandler(color="green")
chain.invoke(inputs, {"callbacks": [handler]})
# Output:
# > Entering new ChainName chain...
# > Finished chain.
```

### StreamingStdOutCallbackHandler

Streams LLM tokens to stdout as they arrive during streaming calls. Only works with models that support streaming.

```python
from langchain_core.callbacks import StreamingStdOutCallbackHandler

handler = StreamingStdOutCallbackHandler()
llm.stream(prompt, {"callbacks": [handler]})  # Tokens printed as they arrive
```

### FileCallbackHandler

Writes execution events to a file. Supports context manager usage:

```python
from langchain_core.callbacks import FileCallbackHandler

with FileCallbackHandler("output.log") as handler:
    chain.invoke(inputs, {"callbacks": [handler]})
```

### UsageMetadataCallbackHandler

Aggregates token usage metadata (input/output token counts) from chat model responses:

```python
from langchain_core.callbacks import UsageMetadataCallbackHandler

callback = UsageMetadataCallbackHandler()
llm.invoke(prompt, {"callbacks": [callback]})
print(callback.usage_metadata)  # {model_name: UsageMetadata}
```

### LangChainTracer

Integrates with LangSmith for production tracing. Automatically enabled via `LANGCHAIN_TRACING_V2=true` environment variable:

```python
from langchain_core.tracers.context import tracing_v2_enabled

with tracing_v2_enabled(project_name="my-project") as cb:
    chain.invoke(inputs)
    run_url = cb.get_run_url()  # Get LangSmith trace URL
```

## Custom Events

Applications can emit custom instrumentation events with arbitrary data using `dispatch_custom_event` (sync) or `adispatch_custom_event` (async). Handlers implement `on_custom_event` to react:

```python
from langchain_core.callbacks import dispatch_custom_event, BaseCallbackHandler
from langchain_core.runnable import RunnableLambda

class CustomHandler(BaseCallbackHandler):
    def on_custom_event(self, name: str, data: Any, **kwargs):
        print(f"Custom event: {name} = {data}")

def my_step(inputs):
    dispatch_custom_event("step_started", {"step": "validation"})
    # ... do work ...
    dispatch_custom_event("step_completed", {"duration_ms": 150})
    return inputs

chain = RunnableLambda(my_step)
chain.invoke({}, {"callbacks": [CustomHandler()]})
```

Custom events are associated with the parent run ID (the calling component's run), enabling correlation with the broader execution trace.

## Testing and Debugging

### RunCollectorCallbackHandler and collect_runs()

For testing and debugging without LangSmith, collect traced runs in memory:

```python
from langchain_core.tracers.context import collect_runs

with collect_runs() as runs_cb:
    chain.invoke(inputs)
    
    # Inspect all traced runs
    for run in runs_cb.traced_runs:
        print(f"Run: {run.name}, Duration: {run.end_time - run.start_time}")
```

This enables unit tests to verify execution flow, event ordering, and handler behavior without external dependencies.

## Error Handling and Control Flow

Callbacks are designed for observability with graceful error handling:

- **Handler exceptions** are logged and swallowed by default unless `raise_error=True`
- **NotImplementedError** on `on_chat_model_start` triggers automatic fallback to `on_llm_start`
- **Async handler errors** in sync contexts are collected and run in a thread pool; exceptions are logged but do not propagate (a pre-existing asymmetry for async handlers)
- **Parent run ID propagation** is automatic in nested runs through `parent_run_id` passed to managers

This design ensures that instrumentation failures do not crash application code.

## Integration with Runnables

Callbacks integrate with LangChain's runnable protocol via the `callbacks` config key:

```python
from langchain_core.runnables import RunnableLambda

chain = RunnableLambda(lambda x: x.upper())
chain.invoke("hello", {"callbacks": [handler], "tags": ["my-tag"]})
```

Runnables internally use `CallbackManager.configure()` to construct managers, propagate tags and metadata, and ensure child components inherit parent context. This enables end-to-end tracing without explicit manager plumbing.
