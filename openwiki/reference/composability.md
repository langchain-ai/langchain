---
type: API Reference
title: Agent Composability API Reference
description: Complete reference documentation for agent composition APIs including middleware, subagents, and hook patterns in LangChain agents.
tags: [composability, agents, middleware, subagents, API-reference, hook-patterns]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-13T08:16:35.902Z
sources:
  - id: openwiki-source-ef89c5b8bd7c35a091f64f0a
    resource: repo://libs/langchain_v1/langchain/agents/__init__.py
  - id: openwiki-source-2bb5c49861e72a9099b51970
    resource: repo://libs/langchain_v1/langchain/agents/_subagent_transformer.py
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
  - id: openwiki-source-03e8ca0eebe37feda8566793
    resource: repo://libs/langchain_v1/langchain/agents/middleware/types.py
  - id: openwiki-source-363977308087d996c7530860
    resource: repo://libs/langchain_v1/tests/unit_tests/agents/test_subagent_transformer.py
generated: { by: "openwiki/0.5.0", at: "2026-11-13T08:16:35.902Z" }
---

## Overview

Agent composability in LangChain is built on three primary mechanisms: **middleware layers** for request/response interception, **subagent delegation** for tool-based hierarchical dispatch, and **hook patterns** for lifecycle control. This reference documents the complete API surface.

## Core Entry Point

### `create_agent()`

```python
def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    response_format: ResponseFormat | type | dict | None = None,
    state_schema: type[AgentState] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
    transformers: Sequence[TransformerFactory] | None = None,
) -> CompiledStateGraph[AgentState, ContextT, InputAgentState, OutputAgentState]
```

Creates a compiled agent graph supporting middleware composition, tool execution, and optional subagent dispatch. 

**Key Parameters:**
- `model`: LLM instance or model identifier (e.g., `"openai:gpt-5.5"`)
- `tools`: List of BaseTool, callables, or provider tool dicts; may be extended dynamically by middleware
- `middleware`: Sequence of `AgentMiddleware` instances applied in order (first = outermost layer)
- `response_format`: Structured output configuration (ToolStrategy, ProviderStrategy, or raw schema)
- `state_schema`: Custom state TypedDict extending `AgentState` to add fields without middleware boilerplate
- `name`: Agent identifier for subagent tracking (passed to LangGraph via `lc_agent_name` config)
- `transformers`: Stream transformer factories composed after `ToolCallTransformer` and middleware transformers

**Returns:** Compiled graph accepting `{"messages": list}` input.

---

## Middleware System

### `AgentMiddleware` Base Class

```python
class AgentMiddleware(Generic[StateT, ContextT, ResponseT]):
    state_schema: type[StateT] = AgentState
    tools: Sequence[BaseTool] = ()
    trace_policy: TracePolicy | None = None
    transformers: Sequence[TransformerFactory] = ()
    
    @property
    def name(self) -> str:
        """Middleware identifier (defaults to class name, customizable)."""
```

All middleware hooks are optional. Implement only the methods needed for your cross-cutting concern.

#### Lifecycle Hooks

**`before_agent(state: StateT, runtime: Runtime[ContextT]) → dict | None`**  
Runs once at agent start, before first model call. Use for prerequisites, initialization, or early exit via `jump_to`.

**`before_model(state: StateT, runtime: Runtime[ContextT]) → dict | None`**  
Runs before each model invocation. Return state updates or `{"jump_to": "tools"/"model"/"end"}` for conditional routing.

**`after_model(state: StateT, runtime: Runtime[ContextT]) → dict | None`**  
Runs after model response; can filter tool calls, inspect structured output, or jump to end.

**`after_agent(state: StateT, runtime: Runtime[ContextT]) → dict | None`**  
Runs once at agent completion, after the loop terminates.

**Async variants:** `abefore_agent`, `abefore_model`, `aafter_model`, `aafter_agent` follow the same signature with `async def`.

#### Interception Hooks

**`wrap_model_call(request: ModelRequest[ContextT], handler: Callable[[ModelRequest], ModelResponse]) → ModelResponse | AIMessage | ExtendedModelResponse`**

Intercepts LLM calls at deepest level. Multiple middleware compose automatically (first = outermost). The `handler` callback executes the model; middleware can:
- Call `handler` multiple times for retry logic
- Skip calling `handler` to short-circuit
- Modify the request via `request.override(**changes)`
- Return `ModelResponse`, simplified `AIMessage`, or `ExtendedModelResponse` with optional `Command` for state updates

**`wrap_tool_call(request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]) → ToolMessage | Command`**

Intercepts each tool execution. The `ToolCallRequest` exposes `tool_call` dict, `BaseTool`, `state`, and `runtime`. Middleware can retry, validate arguments, or override results before execution.

**Async variants:** `awrap_model_call`, `awrap_tool_call`.

#### Type Parameters

- `StateT`: Agent state schema (defaults to `AgentState`)
- `ContextT`: Runtime context type (defaults to `None`)
- `ResponseT`: Structured output response type (defaults to `Any`)

### State Schema (`AgentState`)

```python
class AgentState(TypedDict, Generic[ResponseT]):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]
```

- `messages`: Conversation history (merged with `add_messages` reducer)
- `jump_to`: Ephemeral routing directive; one of `"model"`, `"tools"`, `"end"` (cleared after each step)
- `structured_response`: Populated by `response_format` strategies; typically omitted from input schema

**Middleware can extend** via `state_schema` class attribute or `state_schema` parameter in decorators.

### Request/Response Types

#### `ModelRequest[ContextT]`

```python
@dataclass
class ModelRequest(Generic[ContextT]):
    model: BaseChatModel
    messages: list[AnyMessage]                # Excluding system message
    system_message: SystemMessage | None
    tool_choice: Any | None
    tools: list[BaseTool | dict]             # May be modified by middleware
    response_format: ResponseFormat | None
    state: AgentState[Any]
    runtime: Runtime[ContextT]
    model_settings: dict[str, Any]
    
    def override(**overrides: Unpack[_ModelRequestOverrides]) -> ModelRequest[ContextT]:
        """Return new request with specified fields replaced (immutable pattern)."""
```

#### `ModelResponse[ResponseT]`

```python
@dataclass
class ModelResponse(Generic[ResponseT]):
    result: list[BaseMessage]              # Usually single AIMessage
    structured_response: ResponseT | None  # If response_format was configured
```

#### `ExtendedModelResponse[ResponseT]`

```python
@dataclass
class ExtendedModelResponse(Generic[ResponseT]):
    model_response: ModelResponse[ResponseT]
    command: Command[Any] | None  # Optional state update after model completes
```

Use to return both the model response and additional state mutations (e.g., messages, custom fields) from `wrap_model_call`.

#### `ToolCallRequest`

```python
@dataclass
class ToolCallRequest:
    tool_call: dict[str, Any]     # {"id": ..., "name": ..., "args": ...}
    tool: BaseTool
    state: AgentState[Any]
    runtime: Runtime[ContextT]
    
    def override(tool_call: dict | None = None, tool: BaseTool | None = None) -> ToolCallRequest:
        """Return new request with tool_call or tool replaced."""
```

---

## Decorator-based Middleware

Lightweight middleware can be created with decorators; each wraps a function to auto-generate an `AgentMiddleware` subclass.

### `@before_model`

```python
@before_model(state_schema=MyState, tools=[...], can_jump_to=["end"], name="MyHook")
def my_hook(state: MyState, runtime: Runtime[ContextT]) -> dict | Command | None:
    if some_condition(state):
        return {"jump_to": "end"}  # Skip to end
    return None
```

- `state_schema`: Custom state type (defaults to `AgentState`)
- `tools`: List of tools registered by this middleware
- `can_jump_to`: Valid destinations for routing (enables conditional edges in graph)
- `name`: Middleware identifier

### `@after_model`

```python
@after_model
async def after_hook(state: AgentState, runtime: Runtime) -> None:
    last_msg = state["messages"][-1]
    print(f"Model returned: {last_msg}")
```

Same parameters as `@before_model`.

### `@before_agent` / `@after_agent`

```python
@before_agent
async def init_agent(state: AgentState, runtime: Runtime) -> None:
    runtime.stream_writer({"type": "status", "message": "Starting..."})
```

Runs once at agent start/end. Useful for setup, teardown, and streaming custom events.

### `@dynamic_prompt`

```python
@dynamic_prompt
def get_system_message(request: ModelRequest[ContextT]) -> str | SystemMessage:
    return f"You are an expert in {request.state.get('domain', 'general')}"
```

Returns the system message to inject before each model call (overriding static `system_prompt`).

### `@wrap_model_call`

```python
@wrap_model_call
def retry_on_error(request: ModelRequest, handler: Callable) -> ModelResponse:
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}")
```

Standalone function decorator for intercepting model calls without a full middleware class.

### `@wrap_tool_call`

```python
@wrap_tool_call
def validate_tool_args(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
    args = request.tool_call["args"]
    if not is_valid(args):
        return ToolMessage(content="Invalid args", tool_call_id=request.tool_call["id"], status="error")
    return handler(request)
```

Standalone function decorator for intercepting tool execution.

---

## Subagent Delegation

Subagents are nested `create_agent()` instances dispatched via tools from a parent agent. Detection is automatic: any agent created with `name=...` and invoked from a tool appears on `run.subagents`.

### Subagent Observability

When streaming with `stream_events()` or `astream_events()`:

```python
supervisor = create_agent(model=..., tools=[call_agent_tool], name="supervisor")
run = supervisor.stream_events({"messages": [...]}, version="v3")

for handle in run.subagents:  # SubagentRunStream instances
    print(f"Subagent: {handle.name}")
    print(f"Cause: {handle.cause}")  # {"type": "toolCall", "tool_call_id": "..."}
    for event in handle:
        print(event)
```

**`SubagentRunStream` / `AsyncSubagentRunStream` properties:**
- `name: str | None` – Subagent's `lc_agent_name` (set via `create_agent(name=...)`)
- `cause: LifecycleCause | None` – Causation edge: `{"type": "toolCall", "tool_call_id": "..."}`
- Inherits from LangGraph's `SubgraphRunStream` for event streaming

---

## Middleware Composition Rules

### Stack Order

When multiple middleware are registered:

```python
create_agent(
    model=...,
    middleware=[
        LoggingMiddleware(),      # Outermost
        RetryMiddleware(),        # Middle
        ValidationMiddleware(),   # Innermost
    ]
)
```

**Request flow:** LoggingMiddleware.wrap_model_call → RetryMiddleware.wrap_model_call → ValidationMiddleware.wrap_model_call → actual model

**Response flow:** model → ValidationMiddleware result → RetryMiddleware result → LoggingMiddleware result

### Hook Execution Sequence

Per agent invocation:

1. `before_agent()` (once, all middleware)
2. Loop until no tool calls:
   - `before_model()` (all middleware)
   - `wrap_model_call()` (composed chain, outermost first)
   - `after_model()` (all middleware)
   - Tool dispatch via `Send` (parallel per tool call)
     - `wrap_tool_call()` (composed chain, outermost first)
3. `after_agent()` (once, all middleware)

### State Updates

All hooks can return `dict[str, Any]` to update state. Updates are merged using LangGraph reducers:
- `messages` field uses `add_messages` (appends)
- Other fields use last-write-wins

Alternatively, return a `Command` for more complex control (e.g., `Command(update={"field": value}, goto="node_name")`).

---

## Advanced Patterns

### Retry with Middleware

```python
class RetryMiddleware(AgentMiddleware):
    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts
    
    def wrap_model_call(self, request, handler):
        for attempt in range(self.max_attempts):
            try:
                return handler(request)
            except Exception:
                if attempt == self.max_attempts - 1:
                    raise
                print(f"Retry {attempt + 1}/{self.max_attempts}")
```

### Dynamic Tool Injection

```python
class DynamicToolMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        # Add tools dynamically based on state
        if "search_enabled" in request.state:
            filtered = [t for t in request.tools if t.name in {"search", "calculate"}]
            return handler(request.override(tools=filtered))
        return handler(request)
```

### Tool Validation

```python
class ToolValidationMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        args = request.tool_call["args"]
        try:
            # Validate against tool schema
            request.tool.validate_args(args)
        except ValueError as e:
            return ToolMessage(
                content=f"Validation error: {e}",
                tool_call_id=request.tool_call["id"],
                status="error"
            )
        return handler(request)
```

### Context-aware Model Selection

```python
class ModelRouterMiddleware(AgentMiddleware):
    def __init__(self, fast_model, expensive_model):
        self.fast_model = fast_model
        self.expensive_model = expensive_model
    
    def wrap_model_call(self, request, handler):
        # Route based on context
        if len(request.state["messages"]) > 10:
            return handler(request.override(model=self.fast_model))
        return handler(request.override(model=self.expensive_model))
```

### State Extensions via Middleware

```python
from typing_extensions import TypedDict
from langchain.agents.middleware import OmitFromInput

class ConversationMetadataState(AgentState):
    session_id: str
    attempt_count: NotRequired[int]
    error_log: NotRequired[Annotated[list[str], OmitFromOutput]]

class SessionMiddleware(AgentMiddleware):
    state_schema = ConversationMetadataState
    
    def before_agent(self, state, runtime):
        return {"attempt_count": 0, "error_log": []}
```

---

## Error Handling

### Sync/Async Mismatch

Middleware must provide both sync and async implementations OR use async-capable decorators:

```python
# ❌ Will error in sync context
class BadMiddleware(AgentMiddleware):
    async def awrap_model_call(self, request, handler):  # Only async!
        return await handler(request)

# ✅ Decorator auto-handles both paths
@wrap_model_call
async def good_middleware(request, handler):
    return await handler(request)

# ✅ Provide both implementations
class GoodMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        return handler(request)
    
    async def awrap_model_call(self, request, handler):
        return await handler(request)
```

### Middleware Deduplication

`create_agent()` enforces unique middleware instances:

```python
# ❌ AssertionError: Duplicate middleware
m = LoggingMiddleware()
create_agent(model=..., middleware=[m, m])

# ✅ Create separate instances
create_agent(model=..., middleware=[LoggingMiddleware(), LoggingMiddleware()])
```

---

## Configuration & Operations

### Tracing and Observability

```python
from langgraph.types import omit_payload, TracePolicy

class SilentMiddleware(AgentMiddleware):
    trace_policy = TracePolicy(process_inputs=omit_payload, process_outputs=omit_payload)
    # Wraps are traced but inputs/outputs are omitted from spans
```

### Stream Transformers

```python
class CustomTransformer:
    def __init__(self, scope: tuple[str, ...]) -> None:
        self.scope = scope
    
    def transform(self, input):
        """Transform stream events."""
        return input

agent = create_agent(
    model=...,
    transformers=[CustomTransformer]  # Factories; invoked as factory(scope)
)
```

### Tool Registration via Middleware

```python
class MyMiddleware(AgentMiddleware):
    tools = [search_tool, calculate_tool]  # These are registered on the agent
```

---

## Common Entry Points by Use Case

| Use Case | Pattern |
|----------|---------|
| Logging all calls | `@before_model` + `@after_model` decorators |
| Retry with backoff | `@wrap_model_call` decorator or class method |
| Tool filtering | `@before_model` to inspect state, `wrap_model_call` to filter tools |
| Structured output validation | `@after_model` to inspect `structured_response` |
| Dynamic prompting | `@dynamic_prompt` decorator |
| Tool pre-execution checks | `@wrap_tool_call` decorator |
| Custom state tracking | Custom `AgentMiddleware` with `state_schema` |
| Multi-agent orchestration | `create_agent(name=...)` + tool-based dispatch from parent |
| Context propagation | `runtime.store` or middleware `state_schema` extensions |

---

## Key Invariants

1. **Middleware ordering is significant**: First registered = outermost layer in both request and response chains
2. **Ephemeral state**: `jump_to` field is cleared after each step; persistent routing requires state schema extension
3. **Parallel tool execution**: Multiple tool calls from one model response execute concurrently via LangGraph `Send`
4. **State reducer semantics**: `messages` appends; other fields use last-write-wins
5. **Subagent names are stable**: Agents created with `name="X"` always surface under that name; unnamed agents inherit parent's name
6. **Commands in wrap_model_call**: Only `update` field is supported; `goto`, `resume`, `graph` raise `NotImplementedError`
7. **Tool calls must be executed**: If middleware intercepts and doesn't call `handler`, no `ToolMessage` is produced
