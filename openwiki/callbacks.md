---
type: "Reference"
title: "> Entering new SequentialChain chain..."
openwiki_generated: true
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:18:34.589Z
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
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:18:34.589Z" }
---


## Overview

LangChain's callback system provides a unified mechanism for observability, logging, and tracing of all operations—LLM calls, chain execution, tool usage, and retrieval. Callbacks allow developers to monitor execution flow, collect metrics, stream output, and integrate with external observability platforms like LangSmith without modifying core application code.

The system is built on a hierarchical run structure where parent-child relationships are tracked through run IDs, enabling comprehensive trace trees. Each operation generates events that are dispatched to one or more registered handlers, which can act on them synchronously or asynchronously.

## Architecture

### Core Components

**BaseCallbackHandler** (`repo://libs/core/langchain_core/callbacks/base.py#L496-L546`) is the base class for all callback implementations. It inherits from multiple mixins that define event methods for different operation types:

- **LLMManagerMixin**: `on_llm_start`, `on_llm_new_token`, `on_llm_end`, `on_llm_error`, `on_stream_event`
- **ChainManagerMixin**: `on_chain_start`, `on_chain_end`, `on_chain_error`
- **ToolManagerMixin**: `on_tool_start`, `on_tool_end`, `on_tool_error`
- **RetrieverManagerMixin**: `on_retriever_start`, `on_retriever_end`, `on_retriever_error`
- **AgentManagerMixin**: `on_agent_action`, `on_agent_finish`
- **RunManagerMixin**: `on_text`, `on_retry`, `on_custom_event`
- **CallbackManagerMixin**: start methods for all operation types

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

**handle_event** (`repo://libs/core/langchain_core/callbacks/manager.py#L285-L368`, sync) iterates over handlers and calls the corresponding event method. It handles:

- Skipping handlers based on ignore conditions (e.g., `ignore_llm`, `ignore_chain`)
- Catching and logging exceptions (respecting `raise_error`)
- Collecting async coroutines and running them via executor pool or event loop
- Converting `on_chat_model_start` to `on_llm_start` fallback when not implemented

**ahandle_event** (`repo://libs/core/langchain_core/callbacks/manager.py#L453-L488`, async) separates inline (sequential) and non-inline (concurrent) handlers, using `asyncio.gather()` for parallelism.

The **shielded** decorator (`repo://libs/core/langchain_core/callbacks/manager.py#L221-L254`) preserves context variables in async handlers when cancellation occurs, avoiding task cancellation deadlocks.

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

**Automatic trace callback injection** (`repo://libs/core/langchain_core/tracers/context.py#L105-L130`) happens via `_get_trace_callbacks`, which:

1. Checks if tracing is enabled via `_tracing_v2_is_enabled()`
2. Creates or reuses an existing LangChainTracer
3. Adds it to the callback manager without duplication
4. Returns the configured callback manager for use in operations

### Run Collection

For programmatic access to trace data without LangSmith:

```python
from langchain_core.tracers.context import collect_runs

with collect_runs() as runs_cb:
    chain.invoke("hello")
    for run in runs_cb.traced_runs:
        print(f"Run ID: {run.id}, Type: {run.run_type}")
```

The **RunCollectorCallbackHandler** gathers all Run objects in a list, enabling offline analysis.

## Configuration and Registration

### Via RunnableConfig

Runnables accept callbacks through the `config` parameter:

```python
from langchain_core.runnables.config import RunnableConfig

config = RunnableConfig(
    callbacks=[handler1, handler2],
    tags=["production", "v1"],
    metadata={"user_id": "123", "session": "abc"},
    run_name="my_run",
)
chain.invoke(input, config=config)
```

The **RunnableConfig** TypedDict (`repo://libs/core/langchain_core/runnables/config.py#L57-L129`) supports:

- **callbacks**: Handler list or callback manager
- **tags**: Inheritable labels for filtering
- **metadata**: Inheritable context (JSON-serializable)
- **run_name**: Override default operation name
- **run_id**: Explicit run identifier (UUID)

### Manager Configuration

The static `configure` method creates a fully initialized callback manager:

```python
from langchain_core.callbacks.manager import CallbackManager

manager = CallbackManager.configure(
    inheritable_callbacks=[tracer],
    local_callbacks=[stdout_handler],
    inheritable_tags=["app"],
    local_tags=["expensive_op"],
    inheritable_metadata={"env": "prod"},
    verbose=True,
)
chain.invoke(input, config={"callbacks": manager})
```

### Handler Management

The callback manager API for dynamic handler registration:

```python
manager = CallbackManager(handlers=[])
manager.add_handler(handler, inherit=True)   # Add to both handler lists
manager.add_handler(handler, inherit=False)  # Add only to current operation
manager.remove_handler(handler)
manager.set_handlers([h1, h2], inherit=True) # Replace all handlers
```

### Tag and Metadata Management

```python
manager.add_tags(["tag1", "tag2"], inherit=True)
manager.remove_tags(["tag1"])

manager.add_metadata({"key": "value"}, inherit=True)
manager.remove_metadata(["key"])
```

Tags and metadata are passed to all callbacks via keyword arguments in event methods.

## Callback Lifecycle and Execution

### Run Hierarchy

Operations form a parent-child hierarchy where each child receives a **parent_run_id** linking it to the parent:

```
LLMResult -> parent_run_id=run_id_chain -> parent_run_id=run_id_outer
```

This enables LangSmith to reconstruct the full call tree. The **ParentRunManager** (`repo://libs/core/langchain_core/callbacks/manager.py#L599-L619`) provides `get_child()` to create child callback managers that inherit tags and inheritable handlers.

### Inheritance Rules

- **inheritable_handlers**: Passed to all child operations
- **handlers**: Only used for current operation (not passed to children)
- **inheritable_tags**: Passed to children, accumulated
- **inheritable_metadata**: Passed to children, merged

This allows global handlers (e.g., LangSmith tracer) while supporting operation-specific ones.

### Async Execution Control

The `run_inline` attribute on a handler determines execution:

- **run_inline=True**: Execute in the current async context (sequential)
- **run_inline=False**: Schedule on thread pool or concurrent tasks

```python
class MyInlineHandler(BaseCallbackHandler):
    run_inline = True  # Always runs in caller's context

class MyConcurrentHandler(BaseCallbackHandler):
    run_inline = False  # Runs in executor or concurrently
```

Inline handlers respect the caller's execution context (ContextVar), while non-inline handlers use `copy_context().run()` to preserve context variables across thread boundaries.

## Thread Safety and Context Variables

### Context Preservation

The callback system uses Python's `contextvars` module to propagate context across sync/async boundaries:

- **tracing_v2_callback_var**: Current LangChainTracer (if tracing enabled)
- **run_collector_var**: Current RunCollectorCallbackHandler (if collecting)
- **var_child_runnable_config**: Child runnable config from parent

When `handle_event` runs async coroutines from sync context, it preserves context variables via `copy_context().run()`.

### Thread Safety

Handlers should implement thread-safe state management if accessed by multiple handlers concurrently. The **UsageMetadataCallbackHandler** uses `threading.Lock()` for aggregation.

## Streaming and Token Events

### LLM Token Streaming

When an LLM supports streaming, the callback manager calls `on_llm_new_token` for each token:

```python
class TokenCollector(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []
    
    def on_llm_new_token(self, token, **kwargs):
        self.tokens.append(token)

collector = TokenCollector()
llm.stream("Hello", config={"callbacks": [collector]})
print("".join(collector.tokens))
```

The **chunk** parameter provides the full `GenerationChunk` or `ChatGenerationChunk` object with additional metadata.

### Protocol Events (v3)

For native streaming providers using the v3 protocol:

```python
class ProtocolEventHandler(BaseCallbackHandler):
    def on_stream_event(self, event, **kwargs):
        # event is MessagesData: message-start, content-block-start, etc.
        print(f"Event type: {event['type']}")
```

Protocol events fire at finer granularity than `on_llm_new_token`, with explicit lifecycle boundaries.

## Error Handling

### Exception Propagation

By default, handler exceptions are logged and swallowed:

```python
class BuggyHandler(BaseCallbackHandler):
    def on_chain_start(self, **kwargs):
        raise ValueError("Oops!")  # Logged but doesn't stop execution

handler = BuggyHandler()
chain.invoke(input, config={"callbacks": [handler]})  # Still runs
```

To enforce strict error checking:

```python
handler = BuggyHandler()
handler.raise_error = True
chain.invoke(input, config={"callbacks": [handler]})  # Raises ValueError
```

### Ignore Conditions

Handlers can opt out of specific event types:

```python
class LLMOnlyHandler(BaseCallbackHandler):
    @property
    def ignore_chain(self):
        return True  # Skip all chain events
    
    @property
    def ignore_retriever(self):
        return True  # Skip all retriever events
```

Available properties: `ignore_llm`, `ignore_chain`, `ignore_agent`, `ignore_tool`, `ignore_retriever`, `ignore_retry`, `ignore_chat_model`, `ignore_custom_event`.

## Custom Event Dispatch

### on_custom_event

For application-specific events beyond LLM/chain/tool:

```python
manager = CallbackManager(handlers=[custom_handler])
manager.on_custom_event(
    name="user_interaction",
    data={"user_id": 123, "action": "clicked_button"},
)
```

Handlers receive:

```python
def on_custom_event(self, name, data, run_id, tags, metadata, **kwargs):
    print(f"Event: {name}, Data: {data}")
```

## Chain Groups

For grouping multiple separate calls as a single logical operation:

```python
from langchain_core.callbacks.manager import trace_as_chain_group

with trace_as_chain_group("data_processing", tags=["batch"]) as manager:
    result1 = llm.invoke("query1", config={"callbacks": manager})
    result2 = chain.invoke(input2, config={"callbacks": manager})
    # Both treated as a single chain in LangSmith trace
```

The manager tracks completion state and calls parent's `on_chain_end` or `on_chain_error` when exiting.

## Best Practices

1. **Use context managers for file handlers**: FileCallbackHandler should be used with `with` statement to ensure proper cleanup.

2. **Register global handlers via inheritable_callbacks**: Use `CallbackManager.configure(inheritable_callbacks=[...])` for handlers that should apply everywhere.

3. **Enable LangSmith in production**: Set `LANGCHAIN_TRACING_V2=true` and `LANGSMITH_API_KEY` for automatic trace collection.

4. **Use tags for filtering**: Add semantic tags ("production", "experiment", "expensive") to filter runs in LangSmith.

5. **Include metadata context**: Embed user IDs, session IDs, and environment info in metadata for better observability.

6. **Implement thread-safe handlers**: If your handler accesses shared state, use locks or thread-local storage.

7. **Handle exceptions gracefully**: Set `raise_error=True` only for critical handlers; others should log and continue.

8. **Respect ignore conditions**: If your handler only cares about LLM calls, set `ignore_chain=True` to skip irrelevant events.

9. **Use streaming handlers for real-time feedback**: StreamingStdOutCallbackHandler enables interactive token-by-token output.

10. **Collect token usage**: Use UsageMetadataCallbackHandler to track costs across multi-model applications.
