---
type: Reference
title: Composability API Reference
description: Technical reference for agent composition patterns, middleware hooks, request/response types, and subagent lifecycle in LangChain agents.
tags: [composability, agents, middleware, subagents, API-reference, request-response]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-07T08:15:07.286Z
sources:
  - id: openwiki-source-2bb5c49861e72a9099b51970
    resource: repo://libs/langchain_v1/langchain/agents/_subagent_transformer.py
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
  - id: openwiki-source-03e8ca0eebe37feda8566793
    resource: repo://libs/langchain_v1/langchain/agents/middleware/types.py
generated: { by: "openwiki/0.5.0", at: "2026-11-07T08:15:07.286Z" }
---

## Overview

Agent composability is built on three foundational mechanisms: **middleware composition** (layered interception via hooks), **request/response types** (structured data flowing through the execution pipeline), and **subagent lifecycle management** (detecting and surfacing nested agent runs). This reference documents the core types, hooks, and control flow that enable these patterns.

## Middleware Hook Lifecycle

All agent execution follows a standard sequence of middleware hooks, invoked in order:

1. **`before_agent(state, runtime)`** – Runs once at agent start, before any model call
2. **`before_model(state, runtime)`** – Runs before each model invocation; can skip to tools/end via `jump_to`
3. **`wrap_model_call(request, handler)`** – Intercepts the actual model invocation; middleware chains compose here (first in list = outermost)
4. **`after_model(state, runtime)`** – Runs after model response; can filter/modify tool calls or jump
5. **`wrap_tool_call(request, handler)`** – Intercepts each tool execution independently
6. **`after_agent(state, runtime)`** – Runs once at agent end, after the loop terminates

Hooks have both sync and async versions. Implementing either sync or async satisfies the interface; attempting to call the unimplemented version raises `NotImplementedError` directing the user to implement both or invoke with the matching execution model.

## State Objects

### AgentState

The core state dictionary passed to all hooks and returned by agents.

```python
class AgentState(TypedDict, Generic[ResponseT]):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]
```

**Fields:**

- **`messages`**: Conversation history (list of `BaseMessage`, `AIMessage`, `ToolMessage`, etc.). Updated via `add_messages` reducer, appending new messages rather than replacing.
- **`jump_to`**: Ephemeral routing directive set by middleware to override default control flow:
  - `"model"` – Loop back to model call instead of tools
  - `"tools"` – Skip model, go directly to tool execution
  - `"end"` – Exit agent loop immediately
  - `None` (default) – Follow normal routing (model → tools → end)
- **`structured_response`**: Parsed structured output when `response_format` is configured. Omitted from input schema (set only on output).

Middleware can extend `AgentState` by defining a `state_schema` class attribute. When multiple middleware declare state extensions, schemas merge left-to-right (first middleware overrides later ones for conflicting fields).

### InputAgentState and OutputAgentState

Input and output schemas derived from `AgentState`:

```python
class InputAgentState(TypedDict):
    messages: Required[Annotated[list[AnyMessage | dict[str, Any]], add_messages]]

class OutputAgentState(TypedDict, Generic[ResponseT]):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    structured_response: NotRequired[ResponseT]
```

Input accepts message dicts (auto-converted to `BaseMessage`). Output includes structured responses only when configured. The `jump_to` field is ephemeral and never appears in schema.

## Request and Response Types

### ModelRequest

Encapsulates all data needed to invoke a model.

```python
@dataclass(init=False)
class ModelRequest(Generic[ContextT]):
    model: BaseChatModel
    messages: list[AnyMessage]  # excluding system message
    system_message: SystemMessage | None
    tool_choice: Any | None
    tools: list[BaseTool | dict[str, Any]]
    response_format: ResponseFormat[Any] | None
    state: AgentState[Any]
    runtime: Runtime[ContextT]
    model_settings: dict[str, Any]
```

**Key methods:**

- **`override(**overrides) → ModelRequest`**: Returns a new `ModelRequest` with specified fields replaced. Supports overriding `model`, `system_message`, `messages`, `tools`, `tool_choice`, `response_format`, `model_settings`, and `state`. Immutable pattern—original request unchanged.

**Direct attribute assignment is deprecated.** Middleware must use `override()` to create modified requests rather than mutating the original.

### ModelResponse

Result from model invocation, including optional structured output.

```python
@dataclass
class ModelResponse(Generic[ResponseT]):
    result: list[BaseMessage]
    structured_response: ResponseT | None = None
```

**Fields:**

- **`result`**: List of messages from model (typically a single `AIMessage`, but may include `ToolMessage` if structured output tool was invoked).
- **`structured_response`**: Parsed structured output if `response_format` was used and a valid response was produced. `None` otherwise.

### ExtendedModelResponse

Model response with an optional `Command` for additional state updates.

```python
@dataclass
class ExtendedModelResponse(Generic[ResponseT]):
    model_response: ModelResponse[ResponseT]
    command: Command[Any] | None = None
```

Used by `wrap_model_call` middleware when returning a response alongside a graph command. Commands are applied as state updates after the model node completes, using graph reducers. For reducer fields (like `messages`), later commands add alongside earlier ones; for non-reducer fields, later commands overwrite.

**Unsupported command types** (raise `NotImplementedError` if used): `goto`, `resume`, `graph`.

### ToolCallRequest

Request for tool execution, used by `wrap_tool_call` middleware.

```python
@dataclass
class ToolCallRequest:
    tool_call: dict[str, Any]  # {id, name, args}
    tool: BaseTool | None
    state: AgentState[Any]
    runtime: Runtime[ContextT]
```

**Methods:**

- **`override(tool_call=..., tool=...) → ToolCallRequest`**: Returns modified request. Middleware can rewrite arguments or replace the tool instance.

## Middleware Composition

### Hook Ordering and Composition

When multiple middleware are registered, they compose into a **stack** where the **first in the list becomes the outermost layer**:

```python
middleware=[LoggingMiddleware(), RetryMiddleware()]
```

Control flow for model calls:

```
LoggingMiddleware.wrap_model_call(request, handler)
  → RetryMiddleware.wrap_model_call(request, handler)
    → actual model execution
```

Inner middleware see normalized `ModelResponse` (not `ExtendedModelResponse`), so their commands are isolated. Outer middleware can wrap inner commands and add their own. Commands are accumulated in inner-first order.

### Handler Callback Pattern

All `wrap_*` hooks receive a `handler` callback—the next layer in the middleware stack. Middleware **must** invoke the handler to proceed; skipping it short-circuits execution:

```python
def wrap_model_call(self, request, handler):
    response = handler(request)  # Execute next layer
    return response
```

Handlers can be called multiple times (for retry logic) or skipped (for caching/fallbacks). Each call is independent; middleware manages state for multi-call scenarios (e.g., accumulating retry counts).

### State Updates from Hooks

Hooks returning `dict[str, Any] | None`:

- State updates are merged using graph reducers (e.g., `add_messages` for `messages` field)
- `None` return applies no updates
- Updates are applied before returning control to the caller

## Subagent Detection and Lifecycle

### SubagentTransformer

Automatically detects nested agent runs and surfaces them via `run.subagents` when consuming a run stream.

```python
class SubagentTransformer(_TasksLifecycleBase):
    """Promote nested named agents into typed handles on `run.subagents`."""
```

**Detection criteria** – A nested run is surfaced as a subagent when:

1. It is created via `create_agent(name=...)` with an explicit name
2. Its `lc_agent_name` differs from its parent's (same-named nested agents are surfaced; unnamed agents with `None` are excluded)
3. It was triggered by a tool call (joined via `cause` metadata)

### SubagentRunStream / AsyncSubagentRunStream

Typed handles for accessing nested agent runs:

```python
class SubagentRunStream(SubgraphRunStream):
    @property
    def name(self) -> str | None:
        """Subagent name (the nested run's `lc_agent_name`)."""

    @property
    def cause(self) -> LifecycleCause | None:
        """Causation edge — the tool call that triggered this subagent.
        
        Returns: LifecycleCause dict with type: "toolCall" and tool_call_id, or None.
        """
```

**Properties:**

- **`name`**: The subagent's configured name (from `create_agent(name=...)`).
- **`cause`**: Metadata about the triggering tool call. A dict with `{"type": "toolCall", "tool_call_id": "..."}` if the originating tool call could be joined, else `None`.
- **`status`**: Final status ("success", "failed", or running).
- **`error`**: Optional error message if failed.

## create_agent Factory

The `create_agent` factory function assembles the agent graph and composes all middleware.

### Core Parameters

**Model and Tools:**

- **`model: str | BaseChatModel`** – Language model instance or string identifier (e.g., `"openai:gpt-4o"`). Resolved via `init_chat_model()`.
- **`tools: Sequence[BaseTool | Callable | dict[str, Any]] | None`** – Tools available to the agent. Callables are converted to `BaseTool`. Dicts (provider-native tools) are passed to the provider. Defaults to empty.

**Middleware and Hooks:**

- **`middleware: Sequence[AgentMiddleware]`** – Middleware instances. First in list becomes outermost layer. Duplicate instances raise `AssertionError`.
- **`system_prompt: str | SystemMessage | None`** – System message prefix.

**Structured Output:**

- **`response_format: ResponseFormat[ResponseT] | type[ResponseT] | dict[str, Any] | None`** – Structured output config. Raw schemas are wrapped in `AutoStrategy` for auto-detection. Auto-detection chooses `ProviderStrategy` if model supports it, else `ToolStrategy`.

**State and Context:**

- **`state_schema: type[AgentState[ResponseT]] | None`** – Custom state extension. Used instead of base `AgentState` for schema merging with middleware states.
- **`context_schema: type[ContextT] | None`** – Runtime context schema.

**Persistence and Interrupts:**

- **`checkpointer: Checkpointer | None`** – Checkpoint store for resuming interrupted runs.
- **`store: BaseStore | None`** – LangGraph persistent memory store.
- **`interrupt_before: list[str] | None`** – Node names to interrupt before (e.g., `["tools"]`).
- **`interrupt_after: list[str] | None`** – Node names to interrupt after.

**Observability:**

- **`debug: bool`** – Enable verbose logging for graph execution (node execution, state updates, transitions).
- **`name: str | None`** – Graph name. Used when adding the agent as a subgraph (e.g., in multi-agent systems).

**Caching and Transformers:**

- **`cache: BaseCache[Any] | None`** – Enable caching of graph execution.
- **`transformers: Sequence[TransformerFactory] | None`** – Custom stream transformer factories (invoked as `factory(scope)` at compile time). Order: `ToolCallTransformer`, then middleware transformers, then user-supplied.

### Graph Structure and Execution

The compiled graph has two main nodes:

1. **`agent` node**: Calls the model with the current messages (plus system message). If the model returns tool calls, the graph transitions to tools; otherwise, to END.
2. **`tools` node**: Executes each tool call independently (parallelized via LangGraph's `Send` command). Results are added as `ToolMessage` objects. Transitions back to agent.

The loop repeats until the model returns no tool calls.

### Schema Resolution

`create_agent` merges state schemas in the following order:

1. Collect middleware `state_schema` attributes (in registration order)
2. Append the user's `state_schema` parameter (or base `AgentState` if not provided)
3. Merge left-to-right: later entries override earlier for conflicting fields
4. Generate `input_schema` and `output_schema` by respecting `OmitFromSchema` annotations

This allows users to override middleware-declared state fields by passing an explicit `state_schema`.

## Middleware Composition Internals

### Handler Chaining

Middleware `wrap_model_call` and `awrap_model_call` handlers are chained via `_chain_model_call_handlers` (sync) and `_chain_async_model_call_handlers` (async).

**Composition strategy:**

- Handlers are composed right-to-left so first in list wraps all others
- Inner middleware see normalized `ModelResponse` (unwrapped from `ExtendedModelResponse`)
- Commands from inner middleware are accumulated in a list (inner-first order)
- Outer middleware can add commands; final list is passed to the graph

**Normalized result:**

All handler results are converted to `_ComposedExtendedModelResponse` internally:

```python
@dataclass
class _ComposedExtendedModelResponse(Generic[ResponseT]):
    model_response: ModelResponse[ResponseT]
    commands: list[Command[Any]]  # accumulated inner-first, then outer
```

### Tool Call Chaining

Similarly, `wrap_tool_call` and `awrap_tool_call` handlers are chained via `_chain_tool_call_wrappers` and `_chain_async_tool_call_wrappers`. Tool call wrappers compose first in list = outermost, and their results are accumulated in a list.

## Dynamic Tool Handling

When middleware modifies `request.tools` in `wrap_model_call` to add tools not in the original list, validation occurs:

- If **`wrap_tool_call` or `awrap_tool_call` is defined**, dynamic tool handling is assumed. Middleware can intercept unknown tools and execute them via handler or override.
- If **no tool wrapping is defined**, unknown tools raise `ValueError` with a diagnostic message listing expected vs. received tool names.

**Recommended patterns:**

1. Register all expected tools via `create_agent(tools=[...])` or `middleware.tools`.
2. For runtime-generated tools, implement `wrap_tool_call` to handle them.

## Response Format and Structured Output

### Auto-Detection Strategy

When `response_format` is provided as a raw schema (Pydantic class or dict), `create_agent` wraps it in `AutoStrategy`:

- **At graph compile time**: Wrapped in `AutoStrategy` to defer decision.
- **At first model call**: Auto-detects based on model capabilities.
  - If model supports provider-native structured output (OpenAI, Anthropic), use `ProviderStrategy`
  - Else, use `ToolStrategy` (structured output tool)

Middleware can override `response_format` in `wrap_model_call`, triggering re-detection.

### Handling Structured Output Responses

The `_handle_model_output` function processes model output:

- **ProviderStrategy**: Parses the response directly to structured format.
- **ToolStrategy**: Detects if a structured output tool was called. If so, parses arguments to structured format.
- **Errors**: Validation errors trigger retry logic or exception based on `error_on_failure` config.

## Tracing and Observability

### Trace Policy

Middleware can define a `trace_policy` to control what data is included in LangSmith spans:

```python
class AgentMiddleware(Generic[StateT, ContextT, ResponseT]):
    trace_policy: TracePolicy | None = None
```

The policy is applied to hooks via `traceable` decorator wrapping. Process-wide or middleware-specific policies can shape `process_inputs` and `process_outputs` transformations.

### Subagent Identification

Each agent invocation via `create_agent(name=...)` sets `lc_agent_name` in the run config. The `SubagentTransformer` detects when this name changes (indicating a subagent boundary) and surfaces the nested run on `run.subagents`.

## State Isolation and Context Passing

### Subagent State

Subagents created from delegating tools have isolated `AgentState`. State does not automatically merge with the parent. The delegating tool must extract and format results for the parent.

### Runtime Context

The `runtime` parameter passed to hooks provides access to:

- **`runtime.state`** – Current agent state (read-only via `CONFIG_KEY_READ`)
- **`runtime.checkpointer`** – Checkpoint store for resuming
- **`runtime.store`** – LangGraph persistent memory store for cross-thread data
- **`runtime.context`** – User-provided context object (type `ContextT`)

Delegating tools can read `runtime.state` and `runtime.store` to access parent data, then construct subagent inputs.

## Invariants and Failure Modes

### Middleware Ordering Invariant

Middleware are processed in declaration order. Duplicate middleware instances raise `AssertionError` during agent creation. Implementing the same middleware multiple times requires distinct instances.

### Hook Execution Invariant

If a middleware defines `before_agent` or `abefore_agent`, exactly one is invoked per agent run. Same for `after_agent`/`aafter_agent`. If both sync and async are defined, the appropriate version is called based on execution context.

### Handler Must Be Called (Except for Short-Circuits)

In `wrap_model_call` and `wrap_tool_call`, the handler callback must be invoked to proceed to the next layer, unless middleware is intentionally short-circuiting (caching, fallback, etc.). Forgetting to call the handler may appear to work but leaves the execution incomplete.

### State Update Timing

State updates from hooks are applied **after** the hook returns, before the next hook or node runs. Reducers (e.g., `add_messages`) are invoked; non-reducer fields are overwritten.

## Error Handling Patterns

### ModelRequest / ToolCallRequest Construction

`ModelRequest` and `ToolCallRequest` handle immutability via `override()`. Direct attribute assignment issues deprecation warnings and may fail in strict contexts.

### NotImplementedError on Sync/Async Mismatch

Hooks default to `NotImplementedError` for unimplemented sync/async versions. Middleware must implement both sync and async (or handle async-only via async context manager), or users must invoke agents in the matching execution mode.

### Dynamic Tool Validation

Tools added by middleware in `wrap_model_call` are validated at bind time. If validation fails and no `wrap_tool_call` is defined, a `ValueError` with a diagnostic message is raised.

### Structured Output Validation

Errors during structured output parsing (both `ProviderStrategy` and `ToolStrategy`) raise `StructuredOutputValidationError` or `StructuredOutputError`. Error handling is configurable via `error_on_failure` in response format config.

## See Also

- [Agent Composability: Building Complex Agents](/openwiki/composability.md) – Conceptual guide and patterns
- LangChain Agents documentation – Detailed usage examples
- LangGraph documentation – Graph primitives and `Send` command reference
