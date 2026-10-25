---
type: "Workflow"
title: "Middleware: Extensible Request Processing"
description: "Middleware system for composing agent behavior. Covers middleware interface, execution order, state management, and patterns for routing, logging, and tool control."
tags: ["agents", "middleware", "composition", "hooks", "extensibility", "request-handling"]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-25T08:23:03.589Z
sources:
  - id: openwiki-source-07e634f5cd5f00c636010306
    resource: repo://libs/langchain_v1/langchain/agents/middleware/__init__.py
  - id: openwiki-source-219798683681ccdced950f4d
    resource: repo://libs/langchain_v1/langchain/agents/middleware/tool_error.py
  - id: openwiki-source-03e8ca0eebe37feda8566793
    resource: repo://libs/langchain_v1/langchain/agents/middleware/types.py
generated: { by: "openwiki/0.5.0", at: "2026-10-25T08:23:03.589Z" }
---

# Middleware: Extensible Request Processing

Middleware in LangChain agents provides a composable system for intercepting and customizing agent behavior at multiple execution phases. Middleware enables cross-cutting concerns—logging, validation, caching, retries, security checks, and routing—without modifying core agent logic.

Middleware operates by defining handlers for distinct phases:
- **Lifecycle hooks** (`before_agent`, `after_agent`, `before_model`, `after_model`): Execute synchronously at state transition boundaries
- **Wrapper handlers** (`wrap_model_call`, `wrap_tool_call`): Intercept actual LLM and tool invocations with handler callbacks

Multiple middleware compose automatically, with the first in the list acting as the outermost layer, wrapping inner middleware in nested closures.

## Middleware Interface and Architecture

### Core Base Class

The [`AgentMiddleware`][langchain.agents.middleware.AgentMiddleware] base class defines all hookable methods:

```python
class AgentMiddleware(Generic[StateT, ContextT, ResponseT]):
    """Base middleware class for an agent.
    
    Type Parameters:
        StateT: The type of the agent state (default: AgentState[Any]).
        ContextT: The type of the runtime context (default: None).
        ResponseT: The type of the structured response (default: Any).
    """
    
    state_schema: type[StateT]
    """The schema for state passed to middleware nodes."""
    
    tools: Sequence[BaseTool]
    """Additional tools registered by the middleware."""
    
    trace_policy: TracePolicy | None
    """Trace policy for hook spans (omit_payload, etc.)."""
```

To create middleware, subclass `AgentMiddleware` and override any combination of:
- Lifecycle hooks: `before_agent`, `abefore_agent`, `after_agent`, `aafter_agent`, `before_model`, `abefore_model`, `after_model`, `aafter_model`
- Wrapper handlers: `wrap_model_call`, `awrap_model_call`, `wrap_tool_call`, `awrap_tool_call`

### Async/Sync Duality

Every middleware method has sync (`method`) and async (`amethod`) variants:

- **Sync methods** (`before_model`, `wrap_tool_call`, etc.) are called on the sync execution path when using `invoke()` or `stream()`.
- **Async methods** (`abefore_model`, `awrap_tool_call`, etc.) are called on the async path when using `ainvoke()` or `astream()`.

**You must implement at least one variant.** If you define only the async version and invoke synchronously, you get `NotImplementedError`. Use decorators (see below) to auto-bridge both paths.

## Lifecycle Hooks: State Transitions

Lifecycle hooks execute at agent loop boundaries, receiving the current state and runtime context and returning optional state updates or control flow commands.

### `before_agent` / `abefore_agent`

Runs once at the very start of agent execution, before any model call or tool invocation.

**Signature:**
```python
def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None
async def abefore_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None
```

**Use cases:**
- Initialize or reset agent-level state.
- Validate preconditions.
- Log session start with custom events via `runtime.stream_writer()`.

**Example:**
```python
class InitializationMiddleware(AgentMiddleware):
    def before_agent(self, state, runtime):
        # Emit a custom event
        runtime.stream_writer({"type": "status", "message": "Agent starting..."})
        # Initialize custom state
        return {"conversation_id": str(uuid.uuid4())}
```

### `before_model` / `abefore_model`

Runs before each model call (may run multiple times in a loop).

**Signature:**
```python
def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | Command[Any] | None
async def abefore_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | Command[Any] | None
```

**Returns:**
- `dict[str, Any]`: State updates merged into agent state.
- `Command`: Control flow command (e.g., `{"jump_to": "end"}` to short-circuit).
- `None`: No updates.

**Use cases:**
- Add contextual messages or modify system prompt.
- Implement conditional routing (short-circuit or jump to tools/model/end).
- Add depth/timeout guards.

**Example with conditional jumping:**
```python
from langchain.agents.middleware import before_model, hook_config

@before_model(can_jump_to=["end"])
def max_depth_guard(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    depth = len([m for m in state["messages"] if isinstance(m, AIMessage)])
    if depth > 10:
        return {"jump_to": "end"}  # Exit agent loop
    return None
```

### `after_model` / `aafter_model`

Runs after each model call, allowing inspection and modification of the response before tool invocation.

**Signature:**
```python
def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | Command[Any] | None
async def aafter_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | Command[Any] | None
```

**Use cases:**
- Log or validate model output.
- Detect and handle tool-calling failures.
- Skip tool execution conditionally.

**Example:**
```python
@after_model
def log_model_decision(state: AgentState, runtime: Runtime) -> None:
    last_msg = state["messages"][-1]
    has_tools = hasattr(last_msg, "tool_calls") and last_msg.tool_calls
    runtime.stream_writer({
        "type": "decision",
        "will_call_tools": has_tools
    })
```

### `after_agent` / `aafter_agent`

Runs once when the agent execution completes (no more tools or end condition met).

**Signature:**
```python
def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None
async def aafter_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None
```

**Use cases:**
- Clean up resources.
- Log completion metrics.
- Compute or store summary state.

## Wrapper Handlers: Interception with Control Flow

Wrapper handlers receive a request object and a handler callback, allowing retries, caching, monitoring, and request/response modification.

### `wrap_model_call` / `awrap_model_call`

Intercepts model execution with full control: retry, short-circuit, or modify request and response.

**Signature:**
```python
def wrap_model_call(
    self,
    request: ModelRequest[ContextT],
    handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]

async def awrap_model_call(
    self,
    request: ModelRequest[ContextT],
    handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]
```

**Request object** (`ModelRequest[ContextT]`):
- `model`: The `BaseChatModel` to invoke.
- `messages`: Conversation history (excluding system message).
- `system_message`: System instructions (`SystemMessage` instance).
- `tools`: Available tools for the model.
- `tool_choice`: Tool forcing configuration.
- `response_format`: Structured output schema if specified.
- `state`: Current agent state.
- `runtime`: Runtime context for streaming custom events.
- `model_settings`: Additional model configuration.

**Methods on `ModelRequest`:**
- `override(**kwargs)`: Create a new request with modified fields (immutable pattern).

**Return types:**
- `ModelResponse[ResponseT]`: Full response with messages and optional structured output.
- `AIMessage`: Simplified return (auto-converted to `ModelResponse`).
- `ExtendedModelResponse[ResponseT]`: Response plus optional `Command` for additional state updates.

**Use cases:**
- **Retry logic**: Call `handler()` multiple times on failure.
- **Fallback models**: Catch exceptions, override the model, and retry.
- **Caching**: Short-circuit with cached result if available.
- **Request rewriting**: Modify system message, tools, or messages before calling handler.
- **Response modification**: Transform the response before returning.

**Example: Retry with backoff**
```python
@wrap_model_call
def with_retry(request, handler):
    from anthropic import RateLimitError
    import time
    
    for attempt in range(3):
        try:
            return handler(request)
        except RateLimitError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
```

**Example: Fallback model**
```python
@wrap_model_call
def fallback(request, handler):
    try:
        return handler(request)
    except Exception:
        # Try fallback model
        fallback_model = ChatOpenAI(model="gpt-4-turbo")
        request = request.override(model=fallback_model)
        return handler(request)
```

**Example: Dynamic tool selection**
```python
@wrap_model_call
def select_tools_dynamically(request, handler):
    # Filter tools based on request state
    if "search_mode" in request.state:
        request = request.override(tools=[t for t in request.tools if "search" in t.name])
    return handler(request)
```

### `wrap_tool_call` / `awrap_tool_call`

Intercepts tool execution before invocation. Useful for retries, monitoring, argument modification, and caching.

**Signature:**
```python
def wrap_tool_call(
    self,
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
) -> ToolMessage | Command[Any]

async def awrap_tool_call(
    self,
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
) -> ToolMessage | Command[Any]
```

**Request object** (`ToolCallRequest`):
- `tool`: The `BaseTool` being called.
- `tool_call`: Dict with keys: `id`, `name`, `args` (the arguments).
- `state`: Current agent state (read/write for context).
- `runtime`: Runtime context for custom events.

**Methods on `ToolCallRequest`:**
- `override(**kwargs)`: Create a new request with modified fields.

**Return types:**
- `ToolMessage`: Final result for the model.
- `Command`: Control flow command (graph control, state updates).

**Use cases:**
- **Retry with backoff**: Call `handler()` multiple times.
- **Conditional retry**: Inspect `ToolMessage` result and retry based on status.
- **Argument validation/modification**: Change args before execution.
- **Caching**: Return cached result if available.
- **Monitoring**: Log execution time, input size, etc.

**Example: Retry on tool error**
```python
@wrap_tool_call
def retry_on_error(request, handler):
    for attempt in range(3):
        result = handler(request)
        # ToolMessage has status field: "success", "error", etc.
        if isinstance(result, ToolMessage) and result.status != "error":
            return result
        if attempt < 2:
            continue
        return result
```

**Example: Argument modification**
```python
@wrap_tool_call
def double_numeric_args(request, handler):
    # Modify tool arguments
    modified_call = {
        **request.tool_call,
        "args": {
            k: (v * 2 if isinstance(v, (int, float)) else v)
            for k, v in request.tool_call["args"].items()
        },
    }
    request = request.override(tool_call=modified_call)
    return handler(request)
```

## State Management and Custom Schemas

### Default Agent State

Every agent maintains an `AgentState` with three fields:

```python
class AgentState(TypedDict, Generic[ResponseT]):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]
```

- **messages**: Conversation history merged using `add_messages` reducer.
- **jump_to**: Routing hint (ephemeral, private to middleware).
- **structured_response**: Parsed output when `response_format` is configured.

### Custom State Schemas

Middleware can define custom state by setting `state_schema`:

```python
from typing import TypedDict, Annotated
from langchain.agents.middleware import AgentMiddleware, OmitFromInput

class MyCustomState(TypedDict):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    retry_count: NotRequired[int]
    error_log: NotRequired[list[str]]

class RetryTrackingMiddleware(AgentMiddleware):
    state_schema = MyCustomState
    
    def before_model(self, state: MyCustomState, runtime) -> dict[str, Any] | None:
        # Access custom fields
        state["retry_count"] = state.get("retry_count", 0) + 1
        return {"retry_count": state["retry_count"]}
```

When multiple middleware have different schemas, they must be compatible (share common fields). The agent's final state schema is inferred from decorators or the first middleware that specifies one.

### State Update and Merging

- Returning `dict[str, Any]` from hooks applies state updates via graph reducers.
- For **messages**, updates are merged (new messages append via `add_messages`).
- For other fields, later updates **overwrite** earlier ones (last writer wins).
- Middleware composition order matters: outermost middleware updates override inner ones.

## Decorator Factories

Rather than subclassing, use decorator functions to create lightweight middleware for single hooks:

### `@before_model`, `@after_model`, `@before_agent`, `@after_agent`

Create middleware from a single lifecycle hook function.

```python
from langchain.agents.middleware import before_model, hook_config
from langchain.agents.middleware import AgentState
from langgraph.runtime import Runtime

@before_model
def my_hook(state: AgentState, runtime: Runtime) -> None:
    print(f"Messages: {len(state['messages'])}")

# Can also use decorator with parameters
@before_model(can_jump_to=["end"])
def conditional_hook(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if len(state["messages"]) > 20:
        return {"jump_to": "end"}
    return None
```

**Decorator parameters:**
- `state_schema`: Custom state schema type.
- `tools`: List of additional tools to register.
- `can_jump_to`: List of valid jump destinations (`["tools", "model", "end"]`).
- `name`: Custom middleware class name.

### `@wrap_model_call`, `@wrap_tool_call`

Create middleware from handler callback functions.

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

@wrap_model_call
def my_wrapper(request: ModelRequest, handler) -> ModelResponse | AIMessage:
    print(f"Calling model with {len(request.messages)} messages")
    return handler(request)

@wrap_model_call(name="CustomModelWrapper")
async def async_wrapper(request: ModelRequest, handler) -> ModelResponse | AIMessage:
    result = await handler(request)
    # Transform result
    return result
```

### `@dynamic_prompt`

Create middleware that dynamically computes the system message.

```python
from langchain.agents.middleware import dynamic_prompt
from langchain_core.messages import SystemMessage

@dynamic_prompt
def system_from_state(request: ModelRequest) -> str | SystemMessage:
    # Inspect request.state to decide system message
    mode = request.state.get("mode", "default")
    if mode == "strict":
        return SystemMessage(content="Be very careful and explicit.")
    return SystemMessage(content="Be helpful and concise.")
```

## Execution Order and Composition

### Middleware Chain Structure

Multiple middleware compose as nested layers:

```python
from langchain.agents import create_agent

middleware = [
    RetryMiddleware(),      # Outermost (first in list)
    ToolSelectorMiddleware(),
    LoggingMiddleware(),    # Innermost (last in list)
]

agent = create_agent(model, tools, middleware=middleware)
```

**Execution order (first in list = outermost):**

For `wrap_model_call`:
```
RetryMiddleware.wrap_model_call
  → ToolSelectorMiddleware.wrap_model_call
    → LoggingMiddleware.wrap_model_call
      → handler (actual model call)
```

Each middleware can:
- Call `handler()` zero times (short-circuit)
- Call `handler()` once (normal flow)
- Call `handler()` multiple times (retry)
- Modify request before calling `handler()`
- Modify response after calling `handler()`

### Lifecycle Hook Order

For state-transition hooks, all middleware hooks run in order:

```python
before_agent 1, before_agent 2, before_agent 3
  → model call
before_model 1, before_model 2, before_model 3
  → handler (model invocation)
after_model 1, after_model 2, after_model 3
  → tool calls
...
after_agent 1, after_agent 2, after_agent 3
```

State updates are applied cumulatively. If multiple middleware return state updates, later ones **overwrite** earlier ones for the same key (except for **messages**, which merge).

## Practical Patterns

### Pattern: Input Validation and Sanitization

Validate tool arguments or sanitize sensitive data:

```python
@wrap_tool_call
def validate_args(request, handler):
    tool_name = request.tool_call["name"]
    args = request.tool_call["args"]
    
    # Validation example
    if tool_name == "database_query" and "password" in args:
        raise ValueError("Password cannot be passed as argument")
    
    # Sanitization example
    if tool_name == "log_message":
        args = {k: str(v)[:100] for k, v in args.items()}
        request = request.override(
            tool_call={**request.tool_call, "args": args}
        )
    
    return handler(request)
```

### Pattern: Conditional Tool Filtering

Select available tools based on agent state:

```python
@wrap_model_call
def filter_tools(request, handler):
    # Only allow read tools if in read-only mode
    if request.state.get("read_only", False):
        read_only_tools = [t for t in request.tools if "read" in t.name.lower()]
        request = request.override(tools=read_only_tools)
    
    return handler(request)
```

### Pattern: Distributed Tracing and Monitoring

Log execution timing and resource usage:

```python
import time
from langchain.agents.middleware import wrap_tool_call

@wrap_tool_call
def monitor_tool_execution(request, handler):
    start = time.time()
    tool_name = request.tool_call["name"]
    
    try:
        result = handler(request)
        duration = time.time() - start
        print(f"✓ {tool_name} completed in {duration:.2f}s")
        return result
    except Exception as e:
        duration = time.time() - start
        print(f"✗ {tool_name} failed after {duration:.2f}s: {e}")
        raise
```

### Pattern: Message History Truncation

Summarize old messages to manage token usage:

```python
@before_model
def truncate_old_messages(state, runtime):
    messages = state["messages"]
    
    # Keep only last 20 messages + system context
    if len(messages) > 20:
        # Keep first message (user's initial query)
        # Skip middle messages
        # Keep last 10 messages
        new_messages = [messages[0]] + messages[-10:]
        return {"messages": new_messages}
    
    return None
```

### Pattern: Rate Limiting

Throttle model calls or tool invocations:

```python
import time
from langchain.agents.middleware import wrap_model_call

class RateLimitMiddleware(AgentMiddleware):
    def __init__(self, calls_per_second: float = 1.0):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0
    
    def wrap_model_call(self, request, handler):
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        
        self.last_call = time.time()
        return handler(request)
```

## Built-In Middleware

LangChain provides production-ready middleware for common needs:

- **`ModelRetryMiddleware`**: Automatic retry with exponential backoff for model calls.
- **`ToolRetryMiddleware`**: Automatic retry for failed tool executions.
- **`ToolErrorMiddleware`**: Convert tool exceptions to error `ToolMessage`s (opt-in per exception).
- **`ModelCallLimitMiddleware`**: Enforce maximum model calls per agent run.
- **`ToolCallLimitMiddleware`**: Enforce maximum tool calls per agent run.
- **`ToolErrorMiddleware`**: Handle and format tool execution errors.
- **`HumanInTheLoopMiddleware`**: Interrupt execution for human review/approval.
- **`LLMToolSelectorMiddleware`**: Use a smaller LLM to filter to the most relevant tools.
- **`ModelFallbackMiddleware`**: Fallback to alternative models on primary model failure.
- **`PIIMiddleware`**: Detect and redact personally identifiable information.
- **`ShellToolMiddleware`**: Sandboxed shell execution with resource limits.
- **`SummarizationMiddleware`**: Summarize message history to manage context window.

For details, consult the API documentation and use examples in the middleware package.

## Error Handling and Propagation

### Exception Handling in Middleware

**Lifecycle hooks** (`before_*`, `after_*`):
- Exceptions propagate to the caller.
- Use try/except inside the hook to handle gracefully.

**Wrapper handlers** (`wrap_model_call`, `wrap_tool_call`):
- Exceptions in `handler()` propagate unless caught by middleware.
- Return a fallback value (e.g., error `AIMessage` or `ToolMessage`) to suppress.

**Langgraph control-flow exceptions** (interrupts, parent commands):
- Always propagate; middleware never catches them.

### Example: Graceful Error Recovery

```python
from langchain_core.messages import AIMessage, ToolMessage

@wrap_model_call
def fallback_on_error(request, handler):
    try:
        return handler(request)
    except Exception as e:
        # Return a safe fallback response
        return AIMessage(content=f"I encountered an error: {str(e)}")

@wrap_tool_call
def tool_error_recovery(request, handler):
    try:
        return handler(request)
    except ValueError as e:
        # Return error message to the model
        return ToolMessage(
            content=f"Invalid input: {str(e)}",
            tool_call_id=request.tool_call["id"],
            status="error"
        )
    except Exception:
        # Let other exceptions propagate
        raise
```

## Tracing and Observability

Middleware hooks are traced by the runtime, recording:
- Input state and request objects
- Execution time
- Exceptions (if raised)
- Output state and response

### Controlling Trace Verbosity

Set `trace_policy` on middleware to shape what gets recorded:

```python
from langchain.agents.middleware import TracePolicy, omit_payload

class SensitiveMiddleware(AgentMiddleware):
    # Don't record request payloads (messages, state) in spans
    trace_policy = TracePolicy(process_inputs=omit_payload)
    
    def wrap_model_call(self, request, handler):
        # Still receives full request, but span won't record it
        return handler(request)
```

This is useful for:
- Keeping trace logs lean (timing only, no payloads).
- Excluding sensitive data from observability systems.

## Integration with Agent Factory

Middleware is registered when creating an agent:

```python
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool, calculator_tool],
    system_prompt="You are helpful.",
    middleware=[
        ModelRetryMiddleware(max_retries=3),
        ToolErrorMiddleware(on_error=lambda exc, req: f"Tool error: {exc}"),
        before_model(can_jump_to=["end"])(depth_guard),
    ],
)

# Use the agent
result = agent.invoke({"messages": [HumanMessage("Search for AI news")]})
```

The agent factory:
1. Registers all middleware hooks into the graph.
2. Merges additional tools from middleware into the available tool set.
3. Merges stream transformers from middleware.
4. Applies trace policies to appropriate nodes.
5. Compiles the LangGraph state machine.

## Summary

Middleware provides a clean, composable way to extend agent behavior:

- **Lifecycle hooks** handle state transitions (before/after agent, before/after model).
- **Wrapper handlers** intercept actual LLM and tool calls with full control.
- **Composition** uses nested closures; first in list = outermost layer.
- **State updates** merge messages (via reducers) and overwrite other fields.
- **Decorators** simplify single-hook middleware without subclassing.
- **Built-in middleware** covers common patterns (retries, limits, human-in-loop, tool selection, error handling).
- **Tracing and observability** are integrated; control verbosity with `trace_policy`.

This architecture enables modular, testable, and reusable agent customizations across logging, validation, optimization, and control flow concerns.
