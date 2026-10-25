---
type: "Reference"
title: "Create an LLM run"
openwiki_generated: true
verified:
  - by: openwiki/0.5.0
    at: 2026-10-25T08:23:03.589Z
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
generated: { by: "openwiki/0.5.0", at: "2026-10-25T08:23:03.589Z" }
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

- `raise_error` (bool, default=False): Whether to raise exceptions from the handler
- `run_inline` (bool, default=False): Whether to run the handler synchronously in the calling thread (vs. thread pool for async handlers)
- `ignore_*` properties: Filter callbacks for specific component types (e.g., `ignore_llm`, `ignore_chain`)

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

Async handlers with `run_inline=False` are scheduled concurrently; those with `run_inline=True` run sequentially.

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
    metadata={"version": "1.0"}         # Non-inheritable metadata
)

# When on_chain_start is called, it returns a CallbackManagerForChainRun
chain_manager = manager.on_chain_start(...)

# The child manager created by chain_manager.get_child() inherits:
child = chain_manager.get_child(tag="step1")
# child.handlers = [tracer]           (inheritable only)
# child.inheritable_handlers = [tracer]
# child.inheritable_tags = ["user:alice", "step1"]
```

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

### Merging Callback Managers

Managers can be merged to combine handlers and tags:

```python
from langchain_core.callbacks.manager import CallbackManager

manager1 = CallbackManager(handlers=[handler1], tags=["tag1"])
manager2 = CallbackManager(handlers=[handler2], tags=["tag2"])

merged = manager1.merge(manager2)
# merged.handlers = [handler1, handler2]
# merged.tags = ["tag1", "tag2"]  (deduplicated)
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

Logs events to a file:

```python
from langchain_core.callbacks import FileCallbackHandler

handler = FileCallbackHandler(filename="trace.log")
```

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

With these set, a `LangChainTracer` callback handler is automatically added to all executions, sending traces to the configured LangSmith project.

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

## Custom Events

Applications can dispatch custom events for application-specific instrumentation:

```python
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_core.runnables.config import RunnableConfig

@runnables.RunnableLambda
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

Dispatch with async:

```python
from langchain_core.callbacks.manager import adispatch_custom_event

await adispatch_custom_event("async_event", {"data": "value"}, config=config)
```

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

### CallbackManager.configure()

Use `configure()` for flexible callback setup with environment variable integration:

```python
from langchain_core.callbacks import CallbackManager

manager = CallbackManager.configure(
    inheritable_callbacks=[tracer],
    local_callbacks=[debug_handler],
    inheritable_tags=["system"],
    local_tags=["this-run"],
    inheritable_metadata={"version": "1.0"},
    verbose=True
)
```

When `verbose=True`, a `StdOutCallbackHandler` is automatically added for chain execution tracing.

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

## Event Dispatch Mechanics

### Synchronous Dispatch

The `handle_event` function dispatches synchronous and async handlers:

```
handle_event(handlers, "on_llm_start", "ignore_llm", ...)
├─ For each handler not matching ignore condition:
│  ├─ Call handler.on_llm_start(...) 
│  ├─ If result is a coroutine:
│  │  └─ Collect for async execution
│  └─ If exception and handler.raise_error:
│     └─ Re-raise
└─ If async coroutines collected:
   └─ Execute in thread pool (avoid blocking event loop)
```

Async handlers invoked from sync code run in a new event loop via `ThreadPoolExecutor`.

### Asynchronous Dispatch

The `ahandle_event` function handles async dispatch:

```
ahandle_event(handlers, "on_llm_start", "ignore_llm", ...)
├─ Inline handlers (run_inline=True):
│  └─ await sequentially
└─ Non-inline handlers (run_inline=False):
   └─ asyncio.gather() concurrently
```

This ensures inline handlers maintain strict ordering while non-inline handlers execute concurrently for performance.

### Fallback for Chat Models

When a handler doesn't implement `on_chat_model_start`, the system falls back to `on_llm_start` with prompts converted from messages:

```python
def on_chat_model_start(self, serialized, messages, **kwargs):
    raise NotImplementedError  # Triggers fallback to on_llm_start
```

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

Async version:

```python
from langchain_core.callbacks.manager import atrace_as_chain_group

async with atrace_as_chain_group("async_group") as group_manager:
    result1 = await llm1.ainvoke("prompt", config={"callbacks": group_manager})
    result2 = await llm2.ainvoke("prompt", config={"callbacks": group_manager})
```

## Event Filtering and Composition

### Tag-based Filtering

Callbacks in the observability layer can filter events by tags:

```python
class TagFilterHandler(BaseCallbackHandler):
    def __init__(self, allowed_tags: list[str]):
        self.allowed_tags = allowed_tags
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        tags = kwargs.get("tags", [])
        if any(tag in self.allowed_tags for tag in tags):
            print(f"LLM start: {prompts[0]}")
```

Tags are inherited from parent runs, enabling hierarchical filtering.

### Metadata-based Filtering

Metadata can filter events by feature flags, user, or deployment:

```python
class FeatureGateHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        metadata = kwargs.get("metadata", {})
        if metadata.get("feature_flag:debugging"):
            print(f"Debug mode enabled for: {serialized.get('name')}")
```

## Error Handling

### Handler Exceptions

By default, exceptions in handlers are logged and swallowed:

```python
class FaultyHandler(BaseCallbackHandler):
    def on_llm_start(self, *args, **kwargs):
        raise ValueError("Something went wrong")
        # Logged as warning, execution continues

handler = FaultyHandler()
result = llm.invoke("input", config={"callbacks": [handler]})  # Succeeds
```

Enable error propagation with `raise_error=True`:

```python
handler = FaultyHandler()
handler.raise_error = True
result = llm.invoke("input", config={"callbacks": [handler]})  # Raises ValueError
```

### Async Handler Exceptions in Sync Execution

Exceptions from async handlers invoked via sync `handle_event` are caught generically and logged—they do not honor the `raise_error` flag. This is a pre-existing asymmetry between sync and async paths.

## Streaming and Tokens

### Token Streaming

The `on_llm_new_token` callback fires for each token during streaming:

```python
class TokenHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []
    
    def on_llm_new_token(self, token: str, **kwargs):
        self.tokens.append(token)
        print(token, end="", flush=True)

handler = TokenHandler()
result = llm.stream("Tell me a story", config={"callbacks": [handler]})
```

The `chunk` parameter provides structured token information:

```python
def on_llm_new_token(self, token, chunk=None, **kwargs):
    if chunk:
        print(f"Token: {chunk.text}, Type: {type(chunk).__name__}")
```

### Stream Events (v3)

For high-fidelity event streaming, use `stream_events(version="v3")` with `on_stream_event`:

```python
class StreamEventHandler(BaseCallbackHandler):
    def on_stream_event(self, event, **kwargs):
        print(f"Event: {event['type']}")
        # Event types: message-start, content-block-start, 
        #              content-block-delta, content-block-finish, message-finish
```

## Testing Callbacks

Use `collect_runs()` to test callback behavior without LangSmith:

```python
from langchain_core.tracers.context import collect_runs
import pytest

def test_chain_tracing():
    with collect_runs() as collector:
        result = chain.invoke("input")
    
    assert len(collector.traced_runs) == 1
    run = collector.traced_runs[0]
    assert run.name == "my_chain"
    assert run.inputs == {"input": "input"}
    assert run.outputs == {"output": result}
```

For callback handler unit tests:

```python
def test_custom_handler():
    handler = MyCallbackHandler()
    manager = CallbackManager(handlers=[handler])
    
    managers = manager.on_llm_start(
        serialized={},
        prompts=["test"],
        tags=["test"]
    )
    
    assert handler.was_called
```

## Best Practices

1. **Use inheritable handlers for cross-cutting concerns** (tracing, error reporting) and non-inheritable handlers for run-specific logic (debugging a single chain).

2. **Set `run_inline=True` for handlers that must execute synchronously** (e.g., collecting metrics), and leave `False` for fire-and-forget handlers (e.g., remote logging).

3. **Filter by `ignore_*` properties** rather than checking event types in handlers to reduce noise and improve performance.

4. **Propagate tags and metadata** through your application for hierarchical filtering: `config={"tags": ["user:alice"], "metadata": {"request_id": "req-123"}}`.

5. **Use `collect_runs()` in tests** and `tracing_v2_enabled()` in development for local debugging without LangSmith setup.

6. **Catch exceptions in handlers gracefully**; set `raise_error=True` only for critical handlers that must fail fast.

7. **Use custom events sparingly**; reserve for application-specific milestones (user interactions, business events) rather than component-level instrumentation (which callbacks already provide).

8. **Configure verbose mode in development** to surface chain execution flow via `StdOutCallbackHandler`, but disable in production to avoid log noise.
