---
type: Reference
title: "Agent Middleware System"
description: "Reference guide to agent middleware hooks, composition semantics, request/response types, and handler lifecycle in LangChain agents."
tags: [agent-middleware, lifecycle-hooks, request-interception, composition]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-10T08:17:41.265Z
sources:
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
  - id: openwiki-source-03e8ca0eebe37feda8566793
    resource: repo://libs/langchain_v1/langchain/agents/middleware/types.py
generated: { by: "openwiki/0.5.0", at: "2026-11-10T08:17:41.265Z" }
---

## Overview

Agent middleware in LangChain provides a composable, layered approach to intercepting and modifying agent behavior at well-defined execution points without changing core agent logic. Middleware registers handlers for lifecycle hooks spanning initialization (`before_agent`), per-iteration model invocation (`before_model`, `wrap_model_call`, `after_model`), tool execution (`wrap_tool_call`), and completion (`after_agent`). Multiple middleware instances compose by wrapping handlers in registration order, with the first-registered middleware becoming the outermost layer, enabling cross-cutting concerns such as automatic retries with backoff, error handling and recovery, human-in-the-loop approval workflows, tool argument validation, and request/response transformation.

## Core Hooks

### Lifecycle Hooks

Lifecycle hooks run synchronously with the agent graph and return optional state updates:

- **`before_agent(state, runtime)`** / **`abefore_agent(...)`**: Runs once when agent execution begins, before the first model call. Returns optional state updates.
- **`before_model(state, runtime)`** / **`abefore_model(...)`**: Runs before each model invocation. Can update state or set `jump_to` for control flow.
- **`after_model(state, runtime)`** / **`aafter_model(...)`**: Runs after model response is received. Can filter tool calls or update state.
- **`after_agent(state, runtime)`** / **`aafter_agent(...)`**: Runs after agent loop terminates (no more tool calls or explicit exit). Returns optional state updates.

All lifecycle hooks return `dict[str, Any] | None` to optionally update agent state.

### Interception Hooks

Interception hooks (`wrap_model_call`, `wrap_tool_call`) receive a handler callback that executes the wrapped operation and can call it zero or more times, modify the request/response, or short-circuit entirely:

- **`wrap_model_call(request, handler)`** / **`awrap_model_call(...)`**: Intercepts model execution. Receives `ModelRequest` and handler callback. Can retry, modify request, transform response, or bypass execution.
- **`wrap_tool_call(request, handler)`** / **`awrap_tool_call(...)`**: Intercepts tool execution. Receives `ToolCallRequest` and handler callback. Can retry, validate arguments, or handle errors.

## Request and Response Types

### ModelRequest

Contains all information needed for model invocation:

```python
@dataclass
class ModelRequest(Generic[ContextT]):
    model: BaseChatModel                    # The model to invoke
    messages: list[AnyMessage]              # Messages (excluding system message)
    system_message: SystemMessage | None    # System instructions
    tool_choice: Any | None                 # Tool selection strategy
    tools: list[BaseTool | dict]            # Available tools
    response_format: ResponseFormat[Any] | None  # Structured output format
    state: AgentState[Any]                  # Full agent state
    runtime: Runtime[ContextT]              # Runtime context
    model_settings: dict[str, Any]          # Additional model parameters
```

**Immutable Pattern**: Middleware must not mutate `ModelRequest` directly. Use `request.override(**changes)` to create a new request with modifications.

### ModelResponse

Result from model execution:

```python
@dataclass
class ModelResponse(Generic[ResponseT]):
    result: list[BaseMessage]               # Messages (typically one AIMessage)
    structured_response: ResponseT | None = None  # Parsed structured output if response_format was specified
```

### ExtendedModelResponse

Model response with optional LangGraph Command for additional state updates:

```python
@dataclass
class ExtendedModelResponse(Generic[ResponseT]):
    model_response: ModelResponse[ResponseT]
    command: Command[Any] | None = None
```

Middleware can return `ExtendedModelResponse` to apply a command that modifies state after the model node completes. Commands are applied through state reducers, so messages in commands are **added** to existing messages (not replaced). Non-reducer state fields follow outermost-wins semantics: when multiple middleware return commands, the outermost middleware's values for non-reducer fields override inner ones.

### ToolCallRequest

Request context for tool execution:

```python
@dataclass
class ToolCallRequest:
    tool_call: dict                         # Tool call dict with 'id', 'name', 'args'
    tool: BaseTool | None                   # Resolved BaseTool instance
    state: AgentState[Any]                  # Agent state at time of call
    runtime: ToolRuntime                    # Runtime context with tool-specific info
```

## Composition and Execution Order

### Middleware Stack

When middleware is registered as `[M1, M2, M3]`, the first becomes the outermost layer:

```
Execution order for wrap_model_call:
M1 (outermost) → M2 → M3 (innermost) → actual model execution
Results flow back: M3 can transform → M2 can transform → M1 can transform and return to agent
```

**Key Point**: Innermost middleware (closest to model) sees the result first after execution; outermost middleware sees and can override all inner transformations.

### State Updates and Commands

Lifecycle hooks return state updates merged via LangGraph reducers:

- **Messages field**: Uses `add_messages` reducer, so messages accumulate across updates
- **Non-reducer fields**: Overwritten by the most recent update (last update wins)

**Command Accumulation**: When middleware returns `ExtendedModelResponse` with `Command`, multiple commands are accumulated in a list (inner-to-outer order) and applied sequentially after the model node completes through LangGraph state reducers.

## Registration and Composition

Middleware is passed to `create_agent()` as a list:

```python
agent = create_agent(
    model=ChatOpenAI(),
    tools=[search, calculator],
    middleware=[
        ModelRetryMiddleware(max_retries=2),        # M1: Outermost
        ToolErrorMiddleware(on_error=handler),      # M2: Middle
        HumanInTheLoopMiddleware(interrupt_on={}),  # M3: Innermost
    ],
)
```

**Composition Rule**: First in list = outermost (highest priority for interception and response transformation). The factory collects middleware with `wrap_model_call` hooks, composes them into a single middleware stack where first becomes outermost layer. Commands from each layer are accumulated in inner-to-outer order and applied sequentially.

## Sync and Async Execution

Middleware can provide sync-only, async-only, or both implementations:

- **Sync contexts** (`stream()`, `invoke()`): Calls sync versions (`wrap_model_call`, `wrap_tool_call`)
- **Async contexts** (`astream()`, `ainvoke()`): Calls async versions (`awrap_model_call`, `awrap_tool_call`)
- **Missing implementation**: Raises `NotImplementedError` with guidance

**Best Practice**: Implement both versions unless the middleware is inherently async-only (uses async I/O).

## Error Propagation

Exceptions raised inside middleware (or by the wrapped operation) propagate up the stack unless explicitly handled:

- Middleware can convert exceptions to messages/errors or re-raise them
- Control-flow signals (`interrupt`, `Command`) bypass error handling and always propagate
- Opt-in by design: don't silently swallow exceptions without explicitly converting them

## Common Patterns

### Retry with Exponential Backoff

```python
class RetryMiddleware(AgentMiddleware):
    def __init__(self, max_retries: int = 2, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def wrap_model_call(self, request, handler):
        for attempt in range(self.max_retries + 1):
            try:
                return handler(request)
            except (TimeoutError, ConnectionError) as e:
                if attempt < self.max_retries:
                    delay = self.backoff_factor ** attempt
                    time.sleep(delay)
                else:
                    return ModelResponse(
                        result=[AIMessage(content=f"Failed after retries: {e}")]
                    )
```

### Tool Argument Validation

```python
class ValidatingMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        tool_call = request.tool_call
        args = tool_call.get("args", {})
        
        # Validate and normalize arguments
        if tool_call["name"] == "search":
            args["query"] = args.get("query", "").lower()
            request = request.override(tool_call={**tool_call, "args": args})
        
        return handler(request)
```

### Request/Response Transformation

```python
class TransformingMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        # Transform request (e.g., add context)
        new_request = request.override(
            messages=[*request.messages, SystemMessage(content="Extra context")]
        )
        
        # Execute
        response = handler(new_request)
        
        # Transform response
        return ModelResponse(
            result=[AIMessage(content=f"[Processed] {response.result[0].content}")],
            structured_response=response.structured_response
        )
```

### Conditional Interception

```python
class ConditionalMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        # Check if tool should be approved based on state
        if self._should_block_tool(request):
            return ToolMessage(
                content="Tool execution blocked",
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
                status="error",
            )
        
        return handler(request)
    
    def _should_block_tool(self, request: ToolCallRequest) -> bool:
        # Custom logic based on state
        return False
```

## Key Invariants

- **Immutability**: `ModelRequest` and `ToolCallRequest` follow an immutable pattern; use `override()` or `replace()` methods
- **Execution Order**: Middleware runs in registration order (first = outermost). Multiple calls within a middleware (retries) do not reorder subsequent middleware
- **State Reducer Semantics**: State updates via dict returns merge using reducer semantics. Messages accumulate; non-reducer fields are overwritten
- **Command Flow**: `ExtendedModelResponse` commands are collected (inner-to-outer) and applied after the model node completes through LangGraph reducers
- **Sync/Async Consistency**: Execution path is determined at agent invocation time; middleware cannot switch contexts mid-execution
- **Exception Propagation**: Exceptions that are not explicitly handled propagate to the caller

## See Also

- [Agent Middleware: Composable Request/Response Processing](/openwiki/middleware.md) – Comprehensive guide including all middleware types, composition patterns, and implementation examples
- [Agent Execution Flow and Loop Control](/openwiki/agent-execution.md) – Detailed description of agent loop and middleware hook invocation points
- [Agent Factory and Graph Construction](/openwiki/agent-factory.md) – How the agent graph is built with middleware integration
