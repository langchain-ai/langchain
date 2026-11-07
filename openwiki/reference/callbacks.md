---
type: "Reference"
title: "Callback Manager Interfaces and Run Managers"
description: "Technical reference for callback manager classes, run manager lifecycle, event dispatch mechanisms, and handler configuration patterns in LangChain's instrumentation layer."
tags: ["callbacks", "managers", "handlers", "run-managers", "event-dispatch", "reference"]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-07T08:15:07.286Z
sources:
  - id: openwiki-source-c9313cf42f0120d86b20245f
    resource: repo://libs/core/langchain_core/callbacks/base.py
  - id: openwiki-source-1c233fccf5a66b84d0045366
    resource: repo://libs/core/langchain_core/callbacks/manager.py
generated: { by: "openwiki/0.5.0", at: "2026-11-07T08:15:07.286Z" }
---

## Overview

LangChain's callback system provides event-driven instrumentation through two primary abstractions:

- **Callback Managers**: Coordinate handler registration, lifecycle, and tag/metadata inheritance
- **Run Managers**: Bind callbacks to specific operations (LLM, chain, tool, retriever), providing context-scoped methods for end/error events

This reference covers the core manager and run manager interfaces, event dispatch mechanisms, and patterns for implementing and configuring custom handlers.

## CallbackManager (Synchronous)

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L1377-L1726`

The primary synchronous callback manager that coordinates handlers and dispatches events to them during operation execution.

### Construction and Configuration

```python
from langchain_core.callbacks.manager import CallbackManager

# Minimal initialization
manager = CallbackManager(handlers=[handler1, handler2])

# With inheritance and metadata
manager = CallbackManager(
    handlers=[local_handler],
    inheritable_handlers=[global_tracer],
    parent_run_id=uuid7(),
    tags=["operation_tag"],
    inheritable_tags=["global_tag"],
    metadata={"env": "prod"},
    inheritable_metadata={"session": "abc123"},
)

# Via configure() factory method
manager = CallbackManager.configure(
    inheritable_callbacks=[tracer],
    local_callbacks=[stdout_handler],
    inheritable_tags=["app"],
    local_tags=["expensive"],
    inheritable_metadata={"env": "prod"},
    verbose=True,  # Also registers StdOutCallbackHandler
)
```

### Attributes

- **handlers**: Non-inheritable callbacks for current operation only
- **inheritable_handlers**: Callbacks passed to all child operations (e.g., global tracer)
- **parent_run_id**: UUID reference to parent operation (set by parent manager)
- **tags**: Current operation tags (inheritable tags copied at creation)
- **inheritable_tags**: Tags passed to children, accumulate in hierarchy
- **metadata**: Current operation metadata (inheritable metadata copied at creation)
- **inheritable_metadata**: Metadata passed to children, merged recursively

### Core Methods

**Event Start Methods** — Return specialized run managers for the given operation:

```python
# LLM/Chat operations
llm_run = manager.on_llm_start(
    serialized={"model": "gpt-4"},
    prompts=["Hello world"],
    run_id=uuid7(),
)
# -> CallbackManagerForLLMRun

chat_run = manager.on_chat_model_start(
    serialized={"model": "gpt-4"},
    messages=[[Message(...)]],
    run_id=uuid7(),
)
# -> CallbackManagerForChatModelRun

# Chain/Agent operations
chain_run = manager.on_chain_start(
    serialized={"name": "AgentChain"},
    inputs={"input": "query"},
    run_id=uuid7(),
)
# -> CallbackManagerForChainRun

# Tool operations
tool_run = manager.on_tool_start(
    serialized={"name": "GoogleSearch"},
    input_str="what is langchain",
    run_id=uuid7(),
)
# -> CallbackManagerForToolRun

# Retriever operations
retriever_run = manager.on_retriever_start(
    serialized={"name": "VectorStore"},
    query="langchain overview",
    run_id=uuid7(),
)
# -> CallbackManagerForRetrieverRun
```

**Handler Management**:

```python
manager.add_handler(handler, inherit=True)     # Add to both lists
manager.add_handler(handler, inherit=False)    # Add to handlers only
manager.remove_handler(handler)
manager.set_handlers([h1, h2], inherit=True)   # Replace all handlers
```

**Tag and Metadata Management**:

```python
manager.add_tags(["tag1", "tag2"], inherit=True)
manager.remove_tags(["tag1"])
manager.add_metadata({"key": "val"}, inherit=True)
manager.remove_metadata(["key"])
```

**Manager Operations**:

```python
manager.copy()  # Shallow copy of handlers/tags/metadata
manager.merge(other_manager)  # Union handlers, merge metadata/tags
```

## AsyncCallbackManager (Asynchronous)

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L1859`

The async equivalent of `CallbackManager`, with all methods returning coroutines. Event start methods return async run managers.

```python
from langchain_core.callbacks.manager import AsyncCallbackManager

async def process():
    manager = AsyncCallbackManager(handlers=[async_handler])
    
    llm_run = await manager.on_llm_start(
        serialized={...},
        prompts=[...],
        run_id=uuid7(),
    )
    # -> AsyncCallbackManagerForLLMRun
    
    await llm_run.on_llm_end(response)
```

Async managers support the same handler/tag/metadata operations as sync managers, but all event dispatch methods are awaitable and return async run managers.

## Run Managers

Run managers are returned from event start methods and provide context-scoped methods for reporting end/error/intermediate events. They hold the `run_id` and `parent_run_id` necessary for trace hierarchy.

### RunManager (Synchronous Base)

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L500-L608`

Abstract base for all sync run managers. Provides common methods:

```python
class RunManager(BaseRunManager, ABC):
    # All run managers have these:
    def on_text(self, text: str, **kwargs) -> None:
        """Emit arbitrary text to handlers."""
    
    def on_retry(self, retry_state: RetryCallState, **kwargs) -> None:
        """Emit retry event when tenacity retries."""
    
    def on_custom_event(
        self,
        name: str,
        data: Any,
        **kwargs
    ) -> None:
        """Emit application-specific event."""
```

### ParentRunManager (Synchronous)

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L610-L630`

Base class for sync managers that support parent-child hierarchies. Provides:

```python
child_manager = parent_run_manager.get_child(tag: str | None = None) -> CallbackManager
```

Creates a child callback manager that:
- Inherits all `inheritable_handlers` from parent
- Inherits all `inheritable_tags` from parent
- Inherits all `inheritable_metadata` from parent
- Sets `parent_run_id` to the parent's `run_id`
- Optionally adds a single additional tag to the child

This enables handlers registered at parent level to be invoked for all child operations, creating a unified trace tree.

### CallbackManagerForLLMRun

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L716-L777`

Specialized run manager for LLM operations. In addition to base methods, provides:

```python
class CallbackManagerForLLMRun(RunManager, LLMManagerMixin):
    def on_llm_new_token(
        self,
        token: str | list[str | dict[str, Any]],
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        **kwargs: Any,
    ) -> None:
        """Emit token during streaming."""
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Emit completion when LLM returns."""
    
    def on_llm_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Emit error if LLM raises."""
    
    def on_stream_event(
        self,
        event: MessagesData,
        **kwargs: Any,
    ) -> None:
        """Emit v3 protocol events (message-start, content-block-*, etc)."""
```

### CallbackManagerForChainRun

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L780-L880`

Specialized run manager for chain/agent operations. Extends `ParentRunManager` to support child chains:

```python
class CallbackManagerForChainRun(ParentRunManager, ChainManagerMixin):
    def on_chain_end(self, output: Any, **kwargs: Any) -> None:
        """Emit completion when chain returns."""
    
    def on_chain_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Emit error if chain raises."""
    
    def on_agent_action(
        self,
        action: AgentAction,
        **kwargs: Any,
    ) -> None:
        """Emit action when agent selects a tool."""
    
    def on_agent_finish(
        self,
        finish: AgentFinish,
        **kwargs: Any,
    ) -> None:
        """Emit final output when agent finishes."""
```

### CallbackManagerForToolRun

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L883-L920`

Specialized run manager for tool operations:

```python
class CallbackManagerForToolRun(RunManager, ToolManagerMixin):
    def on_tool_end(
        self,
        output: str,
        **kwargs: Any,
    ) -> None:
        """Emit tool result."""
    
    def on_tool_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Emit tool error."""
```

### CallbackManagerForRetrieverRun

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L923-L960`

Specialized run manager for retriever operations:

```python
class CallbackManagerForRetrieverRun(RunManager, RetrieverManagerMixin):
    def on_retriever_end(
        self,
        documents: Sequence[Document],
        **kwargs: Any,
    ) -> None:
        """Emit retrieved documents."""
    
    def on_retriever_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Emit retrieval error."""
```

### Async Run Managers

For each sync run manager, an async equivalent exists with all methods awaitable:

- `AsyncRunManager` (base for async)
- `AsyncParentRunManager` (supports `await get_child()`)
- `AsyncCallbackManagerForLLMRun`
- `AsyncCallbackManagerForChainRun`
- `AsyncCallbackManagerForToolRun`
- `AsyncCallbackManagerForRetrieverRun`

All have identical method signatures to their sync counterparts, with methods being coroutines.

## Event Dispatch Mechanisms

### Synchronous Dispatch: handle_event

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L296-L379`

Generic event handler that iterates over all handlers and invokes a named event method. Responsible for:

- **Conditional invocation**: Checks handler's `ignore_*` properties (e.g., `ignore_llm`, `ignore_chain`) to skip irrelevant events
- **Async collection**: Detects coroutines returned by async handlers and collects them
- **Exception handling**: Catches exceptions, logs them (respecting `raise_error` flag), and continues
- **Chat model fallback**: If `on_chat_model_start` is not implemented, converts to `on_llm_start` with message strings
- **Async coroutine execution**: Runs collected coroutines in a new event loop (Python 3.11+) or via `asyncio.run()` (Python <3.11), preserving context variables

**Usage pattern** (internal to managers):

```python
handle_event(
    handlers=self.handlers,
    event_name="on_llm_end",
    ignore_condition_name="ignore_llm",
    response,  # positional args passed to handler
    run_id=run_id,  # keyword args passed to handler
    parent_run_id=self.parent_run_id,
    tags=self.tags,
)
```

### Asynchronous Dispatch: ahandle_event

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L464-L499`

Async event handler that separates inline and non-inline handlers:

- **Inline handlers** (`run_inline=True`): Awaited directly in caller's context (sequential)
- **Non-inline handlers** (`run_inline=False`): Submitted to executor or scheduled as concurrent tasks via `asyncio.gather()`

This allows handlers that must run in the caller's context (e.g., updating local state) to execute sequentially, while I/O-bound handlers (e.g., sending to external services) execute concurrently.

### Context Preservation: shielded() Decorator

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L232-L264`

Wraps async handler methods to shield them from cancellation while preserving context variables. Critical for:

- Preserving `tracing_v2_callback_var` (the active LangChainTracer)
- Preserving `run_collector_var` (trace collection context)
- Preventing task cancellation from interrupting callback execution

```python
@shielded
async def _ahandle_event_for_handler(...) -> None:
    # Will not be cancelled even if parent task is cancelled
    # Context variables are preserved
    ...
```

## BaseCallbackHandler

**Location**: `repo://libs/core/langchain_core/callbacks/base.py#L496-L546`

The base class all custom handlers must extend. Inherits from multiple event mixins that define the complete event interface.

### Class Attributes

```python
class BaseCallbackHandler(...):
    raise_error: bool = False
    """If True, exceptions in this handler propagate and stop execution."""
    
    run_inline: bool = False
    """If True, async handler methods run sequentially in caller's context.
    
    Set to True if handler accesses local state or needs caller-context
    semantics. Set to False for I/O-bound handlers to enable concurrency.
    """
```

### Ignore Conditions (Properties)

Handlers can skip event types by implementing `@property` methods that return `True`:

```python
@property
def ignore_llm(self) -> bool:
    """Skip all LLM events."""
    return False

@property
def ignore_chain(self) -> bool:
    """Skip all chain/agent events."""
    return False

@property
def ignore_retriever(self) -> bool:
    """Skip all retriever events."""
    return False

@property
def ignore_retry(self) -> bool:
    """Skip retry events."""
    return False

@property
def ignore_chat_model(self) -> bool:
    """Skip chat model events (even if on_chat_model_start is defined)."""
    return False

@property
def ignore_custom_event(self) -> bool:
    """Skip custom events via on_custom_event."""
    return False
```

### Event Mixins

Handlers inherit event methods from these mixins:

- **LLMManagerMixin**: `on_llm_start`, `on_llm_new_token`, `on_llm_end`, `on_llm_error`, `on_stream_event`
- **ChainManagerMixin**: `on_chain_start`, `on_chain_end`, `on_chain_error`, `on_agent_action`, `on_agent_finish`
- **ToolManagerMixin**: `on_tool_start`, `on_tool_end`, `on_tool_error`
- **RetrieverManagerMixin**: `on_retriever_start`, `on_retriever_end`, `on_retriever_error`
- **RunManagerMixin**: `on_text`, `on_retry`, `on_custom_event`
- **CallbackManagerMixin**: Methods for manager lifecycle (not typically overridden)

### Minimal Handler Implementation

```python
from langchain_core.callbacks import BaseCallbackHandler

class MyHandler(BaseCallbackHandler):
    """Custom instrumentation handler."""
    
    run_inline = True  # Execute in caller's context
    
    @property
    def ignore_chain(self) -> bool:
        """Only care about LLM events."""
        return True
    
    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        """Called when LLM starts."""
        print(f"LLM starting with prompt: {prompts[0]}")
    
    def on_llm_end(self, response, *, run_id, **kwargs):
        """Called when LLM completes."""
        print(f"LLM finished, tokens: {len(response.generations[0][0].text)}")
```

## Configuration Patterns

### Via RunnableConfig

All runnables accept a `config` parameter with callbacks:

```python
from langchain_core.runnables.config import RunnableConfig

config = RunnableConfig(
    callbacks=[handler1, handler2],
    tags=["production", "v1.0"],
    metadata={"user_id": "123", "session": "abc"},
    run_name="my_operation",
    run_id=uuid7(),  # Optional explicit run ID
)

result = chain.invoke(input, config=config)
```

The **RunnableConfig** TypedDict (`repo://libs/core/langchain_core/runnables/config.py#L57-L129`) fields:

- **callbacks**: Handler list or `CallbackManager` instance
- **tags**: List of inheritable labels for filtering
- **metadata**: Dict of inheritable context (JSON-serializable values)
- **run_name**: Override default operation name for traces
- **run_id**: Explicit UUID for the operation (auto-generated if not provided)

### Via configure() Factory

```python
manager = CallbackManager.configure(
    inheritable_callbacks=[global_handler],
    local_callbacks=[operation_handler],
    inheritable_tags=["app"],
    local_tags=["expensive_op"],
    inheritable_metadata={"env": "prod"},
    verbose=True,  # Also registers StdOutCallbackHandler
)

chain.invoke(input, config={"callbacks": manager})
```

### Inheritance Rules

When a parent operation creates a child via `get_child()`:

1. **inheritable_handlers**: All passed to child
2. **handlers**: NOT passed to child (operation-specific only)
3. **inheritable_tags**: Accumulated in child (parent's tags + child's tags)
4. **inheritable_metadata**: Merged in child (parent's metadata overlaid by child's)
5. **tags** and **metadata**: Each operation has its own non-inheritable versions copied from inheritable versions

This enables:

```
Root: inheritable_handlers=[tracer], tags=["root"]
  Child1: inheritable_handlers=[tracer], tags=["root", "child1"]
    Grandchild: inheritable_handlers=[tracer], tags=["root", "child1", "grandchild"]
  Child2: inheritable_handlers=[tracer], tags=["root", "child2"]
```

The global tracer is invoked for all operations, but each operation has its own tag set reflecting its position in the hierarchy.

## Handler Lifecycle and Execution Context

### Sync Handler Execution

```
1. Manager.on_llm_start() -> dispatches to handlers via handle_event()
2. handle_event() checks ignore_llm property
3. If sync method: calls directly in current thread
4. If async method: collects coroutine
5. After all handlers: runs collected coroutines in new event loop
```

**Critical behavior**: If an async event loop is running (sync context from async operation), coroutines are submitted to thread pool executor to avoid deadlock. Context variables are preserved via `copy_context().run()`.

### Async Handler Execution

```
1. Manager.on_llm_start() -> returns coroutine
2. await manager.on_llm_start() -> dispatches to handlers via ahandle_event()
3. ahandle_event() separates inline/non-inline handlers
4. Inline handlers (run_inline=True): awaited sequentially
5. Non-inline handlers (run_inline=False): gathered and awaited concurrently
6. Both protected by shielded() to preserve context and prevent cancellation
```

### Execution Context Variables

The callback system preserves these context variables across thread boundaries:

- **tracing_v2_callback_var**: Current active LangChainTracer (if tracing enabled)
- **run_collector_var**: Current active RunCollectorCallbackHandler (if collecting)
- **var_child_runnable_config**: Child runnable config to propagate to nested operations

Handlers can access these via `contextvars.get()`:

```python
from langchain_core.tracers.context import tracing_v2_callback_var

class MyHandler(BaseCallbackHandler):
    def on_llm_end(self, response, *, run_id, **kwargs):
        tracer = tracing_v2_callback_var.get()
        if tracer:
            # We're inside a tracing context
            pass
```

## Async Handler Patterns

### Inline Handlers (run_inline=True)

For handlers that need to access or modify local state in the caller's context:

```python
class StateTrackingHandler(BaseCallbackHandler):
    run_inline = True
    
    def __init__(self):
        self.call_count = 0
    
    def on_llm_start(self, **kwargs):
        """Runs in caller's context, safe to increment local state."""
        self.call_count += 1
```

Inline handlers execute sequentially in the async context of the caller, preserving execution order and local state semantics.

### Non-Inline Handlers (run_inline=False)

For I/O-bound handlers that can execute concurrently:

```python
class ExternalServiceHandler(BaseCallbackHandler):
    run_inline = False  # Default
    
    async def on_llm_end(self, response, **kwargs):
        """Runs on executor or concurrently, non-blocking."""
        await send_to_external_service(response)
```

Non-inline handlers are submitted to an executor or scheduled as concurrent tasks, allowing other handlers to run while waiting for I/O.

## Exception Handling and Error Propagation

### Handler Exception Control

```python
class StrictHandler(BaseCallbackHandler):
    raise_error = True  # Propagate exceptions
    
    def on_llm_start(self, **kwargs):
        if something_wrong:
            raise ValueError("Stop execution!")

class LenientHandler(BaseCallbackHandler):
    raise_error = False  # Log and continue (default)
    
    def on_llm_start(self, **kwargs):
        # If this raises, it's logged but execution continues
        risky_operation()
```

**Behavior differences**:

- **raise_error=True** (sync handlers): Exception propagates, halts operation
- **raise_error=False** (sync handlers): Exception logged, operation continues
- **Async handlers driven by sync code**: Exceptions always logged and swallowed (pre-existing asymmetry)

### Exception Flow

```python
try:
    operation()  # Triggers on_llm_start, etc.
except Exception as e:
    # Sync handlers with raise_error=True throw here
    # Async handlers with raise_error=True may not (run in thread pool)
    raise
```

## Chain Groups and Grouping Operations

### trace_as_chain_group Context Manager

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L57-L139`

For treating multiple independent calls as a single logical operation in trace trees:

```python
from langchain_core.callbacks.manager import trace_as_chain_group

with trace_as_chain_group(
    "data_processing",
    tags=["batch", "overnight"],
    metadata={"batch_id": "20241125_001"},
) as manager:
    result1 = llm.invoke("query1", config={"callbacks": manager})
    result2 = chain.invoke(input2, config={"callbacks": manager})
    # Both calls share the same run_id parent, appearing as one operation
```

All calls within the context use the same `parent_run_id`, creating a unified trace tree. The manager tracks completion state and calls `on_chain_end` or `on_chain_error` on the parent manager when exiting.

### Async Equivalent

```python
from langchain_core.callbacks.manager import atrace_as_chain_group

async with atrace_as_chain_group("group_name") as manager:
    result1 = await async_chain.ainvoke(input1, config={"callbacks": manager})
    result2 = await async_chain.ainvoke(input2, config={"callbacks": manager})
```

## Callback Manager Merging

For combining multiple callback managers:

```python
manager1 = CallbackManager.configure(
    inheritable_callbacks=[tracer],
    inheritable_tags=["tag1"],
)

manager2 = CallbackManager(
    handlers=[stdout_handler],
    tags=["tag2"],
)

merged = manager1.merge(manager2)
# merged has all handlers from both
# merged tags combine unique values from both
# merged metadata merges both dicts
```

Merging uses set operations to deduplicate handlers and unique tags, with metadata dicts merged via update.
