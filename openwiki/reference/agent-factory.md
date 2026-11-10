---
type: API Reference
title: Agent Factory (create_agent) Reference
description: Complete reference documentation for the create_agent function signature, all overloads, parameters, return types, and exception types for building LangChain agents.
tags: [agents, factory, api-reference, create_agent, state-machine]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-10T08:17:41.265Z
sources:
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
  - id: openwiki-source-ec30ab6256dd50cc670919f6
    resource: repo://libs/langchain_v1/langchain/agents/structured_output.py
generated: { by: "openwiki/0.5.0", at: "2026-11-10T08:17:41.265Z" }
---

## Function Signature Overview

The `create_agent` function is the primary factory entry point for constructing LangChain agents. It accepts a language model, optional tools, and middleware to create a compiled LangGraph state machine that orchestrates conversation flow.

```python
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4o",
    tools=[my_tool],
    system_prompt="You are a helpful assistant.",
)
```

## Function Overloads

The function has three overloaded signatures to provide proper type inference based on the `response_format` parameter:

### Overload 1: No Structured Output

When `response_format` is `None` (default), structured output is disabled and the function returns an agent with untyped responses:

```python
@overload
def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: None = None,
    state_schema: None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache[Any] | None = None,
    transformers: Sequence[TransformerFactory] | None = None,
) -> CompiledStateGraph[AgentState[Any], ContextT, InputAgentState, OutputAgentState[Any]]: ...
```

**Returns**: `CompiledStateGraph` with `ResponseT` bound to `Any` (no structured output).

### Overload 2: Raw Dict Schema

When `response_format` is a raw `dict[str, Any]`, structured output is enabled with an untyped schema:

```python
@overload
def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: dict[str, Any],
    state_schema: type[AgentState[dict[str, Any]]] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache[Any] | None = None,
    transformers: Sequence[TransformerFactory] | None = None,
) -> CompiledStateGraph[
    AgentState[dict[str, Any]], ContextT, InputAgentState, OutputAgentState[dict[str, Any]]
]: ...
```

**Returns**: `CompiledStateGraph` with `ResponseT` bound to `dict[str, Any]`.

### Overload 3: Typed Schema (Preferred)

When `response_format` is a `ResponseFormat` subclass or type (Pydantic model, dataclass, TypedDict), structured output is enabled with proper type inference:

```python
@overload
def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | None = None,
    state_schema: type[AgentState[ResponseT]] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache[Any] | None = None,
    transformers: Sequence[TransformerFactory] | None = None,
) -> CompiledStateGraph[
    AgentState[ResponseT], ContextT, InputAgentState, OutputAgentState[ResponseT]
]: ...
```

**Returns**: `CompiledStateGraph` with `ResponseT` properly inferred from the schema type.

## Parameter Reference

### Core Parameters

#### `model`
**Type**: `str | BaseChatModel`  
**Default**: Required  
**Description**: The language model for the agent. Can be:
- A string identifier (e.g., `"openai:gpt-4o"`, `"anthropic:claude-sonnet-4-5"`)
- A direct `BaseChatModel` instance (e.g., `ChatOpenAI()`, `ChatAnthropic()`)

String identifiers are resolved via `init_chat_model`.

#### `tools`
**Type**: `Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None`  
**Default**: `None`  
**Description**: Tools available to the agent. Supports three formats:
1. **BaseTool instances** (preferred): Full control over execution and metadata
2. **Raw callables**: Automatically wrapped into `BaseTool` instances
3. **Dict-based provider tools**: Native tools from the model's provider (e.g., OpenAI code interpreter); not executed client-side

When `None` or empty, the agent consists of a model node without a tool-calling loop.

#### `system_prompt`
**Type**: `str | SystemMessage | None`  
**Default**: `None`  
**Description**: Optional system prompt for the language model. Can be:
- A string (converted to `SystemMessage` internally)
- A `SystemMessage` instance directly

The system message is prepended to the message list on every model call.

### Middleware and Composition

#### `middleware`
**Type**: `Sequence[AgentMiddleware[StateT_co, ContextT]]`  
**Default**: `()`  
**Description**: Ordered sequence of middleware instances to customize agent behavior. Middleware can intercept and modify agent behavior at various stages:
- `before_agent` / `abefore_agent`: Run once at start before any model calls
- `before_model` / `abefore_model`: Run before each model invocation
- `wrap_model_call` / `awrap_model_call`: Wrap the model invocation itself (retry, fallback, caching)
- `after_model` / `aafter_model`: Run after each model invocation
- `after_agent` / `aafter_agent`: Run once at the end

First middleware in the list becomes the outermost composition layer. Middleware can extend agent state via `state_schema` attributes.

### Structured Output

#### `response_format`
**Type**: `ResponseFormat[ResponseT] | type[ResponseT] | dict[str, Any] | None`  
**Default**: `None`  
**Description**: Configuration for structured responses. Accepts:
- `None`: No structured output (Overload 1)
- `dict[str, Any]`: Raw JSON schema (Overload 2)
- `ResponseFormat` subclass (`ToolStrategy`, `ProviderStrategy`, `AutoStrategy`): Explicit strategy
- Pydantic model class, dataclass, or TypedDict: Auto-detected strategy (Overload 3)

Raw schemas are wrapped in `AutoStrategy` which auto-detects the best strategy based on model capabilities:
1. If the model supports provider-native structured output, uses `ProviderStrategy`
2. Otherwise, uses `ToolStrategy` with a synthetic tool call

#### `state_schema`
**Type**: `type[AgentState[ResponseT]] | None`  
**Default**: `None`  
**Description**: Custom state base class extending `AgentState`. When provided, this schema is used instead of `AgentState` as the base schema for merging with middleware state schemas. Allows users to add custom state fields without creating custom middleware.

User-provided `state_schema` wins on field conflicts with middleware schemas.

#### `context_schema`
**Type**: `type[ContextT] | None`  
**Default**: `None`  
**Description**: Optional schema for runtime context. Passed to the underlying `StateGraph`.

### Persistence and Execution Control

#### `checkpointer`
**Type**: `Checkpointer | None`  
**Default**: `None`  
**Description**: Thread-level checkpoint saver for persisting agent state across turns (e.g., chat memory, conversation history). Enables resuming conversations from saved checkpoints.

#### `store`
**Type**: `BaseStore | None`  
**Default**: `None`  
**Description**: Cross-thread persistence store for data that spans multiple threads/conversations (e.g., user profiles, document stores, shared knowledge bases).

#### `interrupt_before`
**Type**: `list[str] | None`  
**Default**: `None`  
**Description**: List of node names to suspend execution before, enabling user intervention. Example: `["model"]` pauses before model invocation.

#### `interrupt_after`
**Type**: `list[str] | None`  
**Default**: `None`  
**Description**: List of node names to suspend execution after, enabling inspection and resumption. Example: `["tools"]` pauses after tool execution.

### Execution and Monitoring

#### `debug`
**Type**: `bool`  
**Default**: `False`  
**Description**: Enable verbose logging for graph execution. When enabled, prints detailed information about each node execution, state updates, and transitions. Useful for debugging middleware behavior and understanding agent execution flow.

#### `name`
**Type**: `str | None`  
**Default**: `None`  
**Description**: Name for the compiled graph. Used in LangSmith tracing and when adding the agent as a subgraph node in multi-agent systems.

#### `cache`
**Type**: `BaseCache[Any] | None`  
**Default**: `None`  
**Description**: Optional execution cache (LangGraph feature) to cache graph execution results.

#### `transformers`
**Type**: `Sequence[TransformerFactory] | None`  
**Default**: `None`  
**Description**: Additional stream transformer factories to register on the compiled graph. Each factory is invoked as `factory(scope)` to receive a fresh instance. The final transformer order is:
1. `ToolCallTransformer`
2. Transformers from middleware (via `AgentMiddleware.transformers`)
3. Transformers supplied here

## Return Type

All three overloads return a `CompiledStateGraph` object:

```python
CompiledStateGraph[
    AgentState[ResponseT],           # Internal state type
    ContextT,                        # Runtime context type
    InputAgentState,                 # Input schema type
    OutputAgentState[ResponseT]      # Output schema type
]
```

The returned graph is a fully compiled, executable state machine ready for use with `.stream()`, `.invoke()`, or `.astream()`.

## Exception Types

The `create_agent` function and the resulting agent graph may raise the following exceptions:

### Configuration Errors

#### `AssertionError`
**Raised when**: Duplicate middleware instances are provided in the `middleware` sequence.

**Message**: `"Please remove duplicate middleware instances."`

**Recovery**: Ensure each middleware instance in the list is unique.

```python
# ❌ Invalid: duplicate middleware
middleware = [my_middleware, my_middleware]  # AssertionError

# ✅ Valid: unique instances
middleware = [MyMiddlewareA(), MyMiddlewareB()]
```

### Structured Output Errors

#### `StructuredOutputValidationError`
**Raised when**: Structured output tool call arguments fail to parse according to the schema.

**Type**: `StructuredOutputError` (base class)

**Attributes**:
- `tool_name`: The name of the tool that failed
- `source`: The underlying exception that occurred
- `ai_message`: The AI message that contained the invalid structured output

**Recovery**: Depends on `response_format.handle_errors` configuration. If `True`, the agent automatically retries with error context. If `False`, the error is raised to the caller.

#### `MultipleStructuredOutputsError`
**Raised when**: The model returns multiple structured output tool calls when only one is expected.

**Type**: `StructuredOutputError` (base class)

**Attributes**:
- `tool_names`: List of tool names called
- `ai_message`: The AI message that contained the multiple tool calls

**Recovery**: Depends on `response_format.handle_errors` configuration. Defaults to retrying with error context.

### Tool and Model Binding Errors

#### `ValueError`
**Raised when**:
1. Middleware returns unknown client-side tool names not registered upfront
2. `ToolStrategy` specifies tools not declared when creating the agent

**Message Example**:
```
Middleware added tools that the agent doesn't know how to execute.

Unknown tools: [tool_a, tool_b]
Registered tools: [tool_c, tool_d]

This happens when middleware modifies `request.tools` in `wrap_model_call` to include
tools that weren't passed to `create_agent()`.
```

**Recovery**: 
- Register tools at agent creation: `create_agent(tools=[...])`
- Or implement `wrap_tool_call` middleware to handle dynamic tools

#### `NotImplementedError`
**Raised when**:
- A middleware implements only sync `wrap_model_call` but async execution is requested
- A middleware implements only async `awrap_model_call` but sync execution is requested
- Middleware returns unsupported command types (e.g., `Command.goto`, `Command.resume`) from `wrap_model_call`

**Recovery**: Implement both sync and async versions of hooks, or use only the execution mode matching the middleware implementation.

## Quick Example

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather forecast for a location."""
    return f"It's sunny in {location}"

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

# Basic agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=[get_weather, search_web],
    system_prompt="You are a helpful assistant.",
)

# Invoke the agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]
})

# Stream responses
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Search for Python 3.13 features"}]},
    stream_mode="updates"
):
    print(chunk)
```

## Type Parameters

- **`StateT_co`** (Covariant): The agent state type (extends `AgentState[ResponseT]`)
- **`ContextT`** (Contravariant): The runtime context type passed to middleware and nodes
- **`ResponseT`** (Variance depends on overload): The structured response type:
  - Overload 1: `Any` (no structured output)
  - Overload 2: `dict[str, Any]` (untyped dict)
  - Overload 3: Inferred from `response_format` parameter

## Related Documentation

For comprehensive coverage of agent architecture, middleware, structured output strategies, and state management, see:

- **[Agent Execution Flow and Loop Control](/openwiki/workflows/agent-execution.md)**: Complete agent lifecycle, node graph structure, middleware hooks, tool handling, and state management
- **[Agent Factory Detailed Guide](/openwiki/agent-factory.md)**: Deep dives on factory features, middleware composition, and integration patterns
