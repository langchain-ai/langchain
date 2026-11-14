---
type: Reference
title: Callbacks API Reference
description: Complete API reference for callback handlers, managers, and run managers in LangChain's instrumentation system.
tags: [callbacks, API, reference, handlers, managers]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-13T08:16:35.902Z
sources:
  - id: openwiki-source-c9313cf42f0120d86b20245f
    resource: repo://libs/core/langchain_core/callbacks/base.py
  - id: openwiki-source-1c233fccf5a66b84d0045366
    resource: repo://libs/core/langchain_core/callbacks/manager.py
generated: { by: "openwiki/0.5.0", at: "2026-11-13T08:16:35.902Z" }
---

## Handler Interfaces

### BaseCallbackHandler

The `BaseCallbackHandler` class is the root interface for all callback implementations. It inherits from multiple mixins that define event methods for different component types.

**Location**: `repo://libs/core/langchain_core/callbacks/base.py#L496-L546`

**Core attributes**:
- `raise_error: bool = False` — Whether to raise exceptions from callback methods
- `run_inline: bool = False` — Whether to run async callbacks synchronously in the calling thread

**Filter properties** (all return `bool`, default `False`):
- `ignore_llm` — Skip LLM events
- `ignore_chat_model` — Skip chat model events
- `ignore_chain` — Skip chain events
- `ignore_agent` — Skip agent events
- `ignore_tool` — Skip tool events
- `ignore_retriever` — Skip retriever events
- `ignore_retry` — Skip retry events
- `ignore_custom_event` — Skip custom events

### LLMManagerMixin

Methods for LLM event handling.

**Location**: `repo://libs/core/langchain_core/callbacks/base.py#L62-L167`

```python
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
) -> Any:
    """Called when an LLM starts executing."""

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
    """Called when a new token is generated during streaming."""

def on_llm_end(
    self,
    response: LLMResult,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Called when an LLM completes execution."""

def on_llm_error(
    self,
    error: BaseException,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Called when an LLM raises an exception."""

def on_stream_event(
    self,
    event: MessagesData,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Called for each protocol event from stream_events(version="v3")."""
```

### ChainManagerMixin

Methods for chain and agent event handling.

**Location**: `repo://libs/core/langchain_core/callbacks/base.py#L169-L239`

```python
def on_chain_start(
    self,
    serialized: dict[str, Any],
    inputs: dict[str, Any],
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Called when a chain starts executing."""

def on_chain_end(
    self,
    outputs: dict[str, Any],
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any:
    """Called when a chain completes execution."""

def on_chain_error(
    self,
    error: BaseException,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any:
    """Called when a chain raises an exception."""

def on_agent_action(
    self,
    action: AgentAction,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any:
    """Called when an agent takes an action."""

def on_agent_finish(
    self,
    finish: AgentFinish,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any:
    """Called when an agent finishes execution."""
```

### ToolManagerMixin

Methods for tool execution tracking.

**Location**: `repo://libs/core/langchain_core/callbacks/base.py#L241-L277`

```python
def on_tool_start(
    self,
    serialized: dict[str, Any],
    input_str: str,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Called when a tool starts executing."""

def on_tool_end(
    self,
    output: Any,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any:
    """Called when a tool completes execution."""

def on_tool_error(
    self,
    error: BaseException,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any:
    """Called when a tool raises an exception."""
```

### RetrieverManagerMixin

Methods for retrieval tracking.

**Location**: `repo://libs/core/langchain_core/callbacks/base.py#L24-L60`

```python
def on_retriever_start(
    self,
    serialized: dict[str, Any],
    query: str,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Called when a retriever starts executing."""

def on_retriever_end(
    self,
    documents: Sequence[Document],
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any:
    """Called when a retriever completes execution."""

def on_retriever_error(
    self,
    error: BaseException,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any:
    """Called when a retriever raises an exception."""
```

### RunManagerMixin

Methods for arbitrary text, retry, and custom event handling.

**Location**: `repo://libs/core/langchain_core/callbacks/base.py#L435-L494`

```python
def on_text(
    self,
    text: str,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any:
    """Called to output arbitrary text."""

def on_retry(
    self,
    retry_state: RetryCallState,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any:
    """Called when an operation is retried."""

def on_custom_event(
    self,
    name: str,
    data: Any,
    *,
    run_id: UUID,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Called for custom application events."""
```

### CallbackManagerMixin

Methods for component start events.

**Location**: `repo://libs/core/langchain_core/callbacks/base.py#L279-L433`

```python
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
) -> Any:
    """Called when a text completion LLM starts."""

def on_chat_model_start(
    self,
    serialized: dict[str, Any],
    messages: list[list[BaseMessage]],
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Called when a chat model starts. Must declare explicit signature."""

def on_chain_start(
    self,
    serialized: dict[str, Any],
    inputs: dict[str, Any],
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Called when a chain starts."""

def on_tool_start(
    self,
    serialized: dict[str, Any],
    input_str: str,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Called when a tool starts."""

def on_retriever_start(
    self,
    serialized: dict[str, Any],
    query: str,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Called when a retriever starts."""
```

### AsyncCallbackHandler

Async variant of `BaseCallbackHandler` with async method signatures.

**Location**: `repo://libs/core/langchain_core/callbacks/base.py#L548-L1002`

All methods are `async def` with identical signatures to the sync versions. Key async methods:

```python
async def on_llm_start(...) -> None: ...
async def on_chat_model_start(...) -> Any: ...
async def on_llm_new_token(...) -> None: ...
async def on_llm_end(...) -> None: ...
async def on_llm_error(...) -> None: ...
async def on_stream_event(...) -> None: ...
async def on_chain_start(...) -> None: ...
async def on_chain_end(...) -> None: ...
async def on_chain_error(...) -> None: ...
async def on_tool_start(...) -> None: ...
async def on_tool_end(...) -> None: ...
async def on_tool_error(...) -> None: ...
async def on_retriever_start(...) -> None: ...
async def on_retriever_end(...) -> None: ...
async def on_retriever_error(...) -> None: ...
async def on_text(...) -> None: ...
async def on_retry(...) -> Any: ...
async def on_agent_action(...) -> None: ...
async def on_agent_finish(...) -> None: ...
async def on_custom_event(...) -> None: ...
```

## Manager Classes

### BaseCallbackManager

Base class managing handler lifecycle and context propagation.

**Location**: `repo://libs/core/langchain_core/callbacks/base.py#L1004-L1227`

**Constructor**:
```python
def __init__(
    self,
    handlers: list[BaseCallbackHandler],
    inheritable_handlers: list[BaseCallbackHandler] | None = None,
    parent_run_id: UUID | None = None,
    *,
    tags: list[str] | None = None,
    inheritable_tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    inheritable_metadata: dict[str, Any] | None = None,
) -> None: ...
```

**Attributes**:
- `handlers: list[BaseCallbackHandler]` — Non-inheritable callbacks for current operation
- `inheritable_handlers: list[BaseCallbackHandler]` — Callbacks passed to child operations
- `parent_run_id: UUID | None` — Parent operation ID for trace hierarchy
- `tags: list[str]` — Non-inheritable tags
- `inheritable_tags: list[str]` — Tags passed to children
- `metadata: dict[str, Any]` — Non-inheritable metadata
- `inheritable_metadata: dict[str, Any]` — Metadata passed to children
- `is_async: bool` — Whether manager is async (property)

**Core methods**:

```python
def copy(self) -> Self:
    """Return a deep copy of the callback manager."""

def merge(self, other: BaseCallbackManager) -> Self:
    """Merge with another manager, combining handlers, tags, and metadata."""

def add_handler(
    self,
    handler: BaseCallbackHandler,
    inherit: bool = True,
) -> None:
    """Add a handler to the manager."""

def remove_handler(self, handler: BaseCallbackHandler) -> None:
    """Remove a handler from the manager."""

def set_handlers(
    self,
    handlers: list[BaseCallbackHandler],
    inherit: bool = True,
) -> None:
    """Replace all handlers."""

def set_handler(
    self,
    handler: BaseCallbackHandler,
    inherit: bool = True,
) -> None:
    """Set a single handler as the only handler."""

def add_tags(
    self,
    tags: list[str],
    inherit: bool = True,
) -> None:
    """Add tags to the manager."""

def remove_tags(self, tags: list[str]) -> None:
    """Remove tags from the manager."""

def add_metadata(
    self,
    metadata: dict[str, Any],
    inherit: bool = True,
) -> None:
    """Add metadata to the manager."""

def remove_metadata(self, keys: list[str]) -> None:
    """Remove metadata keys from the manager."""
```

### CallbackManager

Synchronous callback manager for LangChain operations.

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L1364-L1726`

**Start event methods** — Return specialized run managers:

```python
def on_llm_start(
    self,
    serialized: dict[str, Any],
    prompts: list[str],
    run_id: UUID | None = None,
    **kwargs: Any,
) -> list[CallbackManagerForLLMRun]:
    """Start LLM execution, returns one manager per prompt."""

def on_chat_model_start(
    self,
    serialized: dict[str, Any],
    messages: list[list[BaseMessage]],
    run_id: UUID | None = None,
    **kwargs: Any,
) -> list[CallbackManagerForLLMRun]:
    """Start chat model execution, returns one manager per message list."""

def on_chain_start(
    self,
    serialized: dict[str, Any] | None,
    inputs: dict[str, Any] | Any,
    run_id: UUID | None = None,
    **kwargs: Any,
) -> CallbackManagerForChainRun:
    """Start chain execution, returns chain run manager."""

def on_tool_start(
    self,
    serialized: dict[str, Any] | None,
    input_str: str,
    run_id: UUID | None = None,
    parent_run_id: UUID | None = None,
    inputs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> CallbackManagerForToolRun:
    """Start tool execution, returns tool run manager."""

def on_retriever_start(
    self,
    serialized: dict[str, Any] | None,
    query: str,
    run_id: UUID | None = None,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> CallbackManagerForRetrieverRun:
    """Start retriever execution, returns retriever run manager."""
```

### AsyncCallbackManager

Asynchronous callback manager for async LangChain operations.

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L1859+`

All start methods are `async` with identical signatures to `CallbackManager`, returning async variants:

```python
async def on_llm_start(...) -> list[AsyncCallbackManagerForLLMRun]: ...
async def on_chat_model_start(...) -> list[AsyncCallbackManagerForLLMRun]: ...
async def on_chain_start(...) -> AsyncCallbackManagerForChainRun: ...
async def on_tool_start(...) -> AsyncCallbackManagerForToolRun: ...
async def on_retriever_start(...) -> AsyncCallbackManagerForRetrieverRun: ...
```

## Run Managers

Run managers are returned from start events and provide context for end/error callbacks.

### BaseRunManager

Base class for all run managers.

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L477-L531`

**Attributes**:
- `run_id: UUID` — Unique identifier for this run
- `parent_run_id: UUID | None` — Parent operation ID
- `handlers: list[BaseCallbackHandler]` — Non-inheritable handlers
- `inheritable_handlers: list[BaseCallbackHandler]` — Handlers for child operations
- `tags: list[str]` — Non-inheritable tags
- `inheritable_tags: list[str]` — Tags for child operations
- `metadata: dict[str, Any]` — Non-inheritable metadata
- `inheritable_metadata: dict[str, Any]` — Metadata for child operations

**Methods**:
```python
@classmethod
def get_noop_manager(cls) -> Self:
    """Return a manager that performs no operations."""
```

### RunManager

Synchronous run manager with text and retry support.

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L533-L584`

```python
def on_text(self, text: str, **kwargs: Any) -> None:
    """Emit arbitrary text."""

def on_retry(self, retry_state: RetryCallState, **kwargs: Any) -> None:
    """Handle retry event."""
```

### ParentRunManager

Adds child manager creation to `RunManager`.

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L586-L630`

```python
def get_child(self, tag: str | None = None) -> CallbackManager:
    """Create a child callback manager with inherited handlers and context."""
```

### CallbackManagerForLLMRun

Run manager for LLM operations.

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L632-L810`

```python
def on_llm_new_token(
    self,
    token: str | list[str | dict[str, Any]],
    *,
    chunk: GenerationChunk | ChatGenerationChunk | None = None,
    **kwargs: Any,
) -> None:
    """Handle new token from LLM stream."""

def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
    """Handle LLM completion."""

def on_llm_error(
    self,
    error: BaseException,
    **kwargs: Any,
) -> None:
    """Handle LLM error."""

def on_stream_event(self, event: MessagesData, **kwargs: Any) -> None:
    """Handle protocol event from stream_events(version="v3")."""
```

### CallbackManagerForChainRun

Run manager for chain operations.

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L915-L1048`

```python
def on_chain_end(self, outputs: dict[str, Any] | Any, **kwargs: Any) -> None:
    """Handle chain completion."""

def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
    """Handle chain error."""

def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
    """Handle agent action."""

def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
    """Handle agent finish."""
```

### CallbackManagerForToolRun

Run manager for tool operations.

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L1050-L1180`

```python
def on_tool_end(self, output: Any, **kwargs: Any) -> None:
    """Handle tool completion."""

def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
    """Handle tool error."""
```

### CallbackManagerForRetrieverRun

Run manager for retriever operations.

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L1235-L1287`

```python
def on_retriever_end(
    self,
    documents: Sequence[Document],
    **kwargs: Any,
) -> None:
    """Handle retriever completion."""

def on_retriever_error(
    self,
    error: BaseException,
    **kwargs: Any,
) -> None:
    """Handle retriever error."""
```

### Async Run Managers

Async variants exist for all run managers with `Async` prefix:
- `AsyncRunManager`
- `AsyncParentRunManager`
- `AsyncCallbackManagerForLLMRun`
- `AsyncCallbackManagerForChainRun`
- `AsyncCallbackManagerForToolRun`
- `AsyncCallbackManagerForRetrieverRun`

All methods are `async def` with identical signatures to sync versions.

## Event Dispatch

### handle_event

Synchronous event handler dispatch function.

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L287-L370`

```python
def handle_event(
    handlers: list[BaseCallbackHandler],
    event_name: str,
    ignore_condition_name: str | None,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Dispatch a callback event to all handlers.
    
    Args:
        handlers: Handlers to dispatch to
        event_name: Method name to call (e.g., 'on_llm_start')
        ignore_condition_name: Handler property to check for skip condition
        *args: Positional arguments to pass to handler method
        **kwargs: Keyword arguments to pass to handler method
    """
```

**Behavior**:
- Skips handlers where `getattr(handler, ignore_condition_name)` is `True`
- Handles async handlers by collecting coroutines and running them via thread pool or event loop
- Falls back from `on_chat_model_start` to `on_llm_start` on `NotImplementedError`
- Respects `handler.raise_error` to propagate or suppress exceptions
- Logs non-fatal errors

### ahandle_event

Asynchronous event handler dispatch function.

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L440-L475`

```python
async def ahandle_event(
    handlers: list[BaseCallbackHandler],
    event_name: str,
    ignore_condition_name: str | None,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Async dispatch a callback event to all handlers."""
```

**Behavior**:
- Runs inline handlers (`handler.run_inline == True`) sequentially
- Runs other handlers concurrently via `asyncio.gather()`
- Falls back from `on_chat_model_start` to `on_llm_start` on `NotImplementedError`
- Respects `handler.raise_error` for exception propagation
- Logs non-fatal errors

### shielded Decorator

Preserves context variables in async handlers during cancellation.

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L232-L256`

```python
def shielded(func: Func) -> Func:
    """Shield async function from cancellation while preserving context variables.
    
    Captures context, creates task with that context, and shields from cancellation.
    """
```

## Helper Functions

### trace_as_chain_group

Context manager for grouping operations as a single trace.

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L56-L139`

```python
@contextmanager
def trace_as_chain_group(
    group_name: str,
    callback_manager: CallbackManager | None = None,
    *,
    inputs: dict[str, Any] | None = None,
    project_name: str | None = None,
    address: AgentAddress | None = None,
    example_id: str | UUID | None = None,
    run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Generator[CallbackManagerForChainGroup, None, None]:
    """Create a callback manager for a named group of operations."""
```

### atrace_as_chain_group

Async variant of `trace_as_chain_group`.

**Location**: `repo://libs/core/langchain_core/callbacks/manager.py#L141-L227`

```python
@asynccontextmanager
async def atrace_as_chain_group(
    group_name: str,
    callback_manager: AsyncCallbackManager | None = None,
    *,
    inputs: dict[str, Any] | None = None,
    project_name: str | None = None,
    address: AgentAddress | None = None,
    example_id: str | UUID | None = None,
    run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AsyncGenerator[AsyncCallbackManagerForChainGroup, None]:
    """Create an async callback manager for a named group of operations."""
```

## Type Aliases

```python
Callbacks = list[BaseCallbackHandler] | BaseCallbackManager | None
```

The `Callbacks` type is used throughout LangChain to accept callback configuration in multiple forms: a list of handlers, a single manager, or `None` for no callbacks.
