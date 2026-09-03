---
type: "Reference"
title: "Middleware"
openwiki_generated: true
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:18:34.589Z
sources:
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
  - id: openwiki-source-bc95cc8fff4e07f74e50ce8b
    resource: repo://libs/langchain_v1/langchain/agents/middleware/human_in_the_loop.py
  - id: openwiki-source-c29f7722b0a4bcc0d760335e
    resource: repo://libs/langchain_v1/langchain/agents/middleware/model_retry.py
  - id: openwiki-source-219798683681ccdced950f4d
    resource: repo://libs/langchain_v1/langchain/agents/middleware/tool_error.py
  - id: openwiki-source-03e8ca0eebe37feda8566793
    resource: repo://libs/langchain_v1/langchain/agents/middleware/types.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:18:34.589Z" }
---


## Overview

Middleware in LangChain agents provides a composable, layered approach to intercepting and modifying agent behavior without changing core agent logic. Middleware hooks into the agent execution loop to implement cross-cutting concerns: automatic retries, error handling, human-in-the-loop approval, PII redaction, structured output transformation, and tool caching.

### Core Architectural Principles

**Interception Model**: Each middleware instance registers sync/async handler pairs for lifecycle hooks. When the agent executes, hooks are invoked in registration order, with each middleware observing or transforming state before delegating to the next layer or the core execution logic.

**Composition Semantics**: Middleware registered first in the list becomes the outermost layer. When composing `wrap_model_call` or `wrap_tool_call` handlers, each middleware wraps the previous ones, establishing a chain where the first middleware intercepts first and last.

- **Model call stack example**: `[Middleware1, Middleware2, Middleware3]` becomes `Middleware1(Middleware2(Middleware3(model)))`, so `Middleware1` receives control first, calls the next handler (which is Middleware2's wrapper), receives the result, and can transform it before returning.
- **Tool call stack**: Identical composition order — first middleware is outermost.

**State and Command Flow**: Most hooks return optional state updates (`dict[str, Any]`). The `wrap_model_call` and `wrap_tool_call` interception points also support returning `Command` objects from LangGraph, allowing middleware to redirect execution or jump to different nodes (e.g., skip tool execution, loop back to model, or exit).

## AgentMiddleware Protocol

All middleware classes inherit from `AgentMiddleware`, a generic base that defines optional hooks and establishes the contract for interceptors.

### Type Parameters

```python
class AgentMiddleware(Generic[StateT, ContextT, ResponseT]):
    """
    StateT: Type of agent state (default: AgentState[Any]).
    ContextT: Type of runtime context (default: None).
    ResponseT: Type of structured response (default: Any).
    """
```

### Lifecycle Hooks

Middleware can implement any or all of these lifecycle methods; unimplemented methods default to no-op:

#### Before/After Agent

```python
def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """Runs once at the very start of agent execution, before the first model call."""

async def abefore_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """Async version of before_agent."""

def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """Runs once after the agent loop terminates (no more tool calls or explicit exit)."""

async def aafter_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """Async version of after_agent."""
```

#### Before/After Model

```python
def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """Runs before each model invocation; allows state transformation before the call."""

async def abefore_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """Async version of before_model."""

def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """Runs after each model response; allows approval, modification, or rejection of tool calls."""

async def aafter_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """Async version of after_model."""
```

#### Wrap Model Call (Core Interception)

```python
def wrap_model_call(
    self,
    request: ModelRequest[ContextT],
    handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
    """Intercept and control model execution.
    
    The handler callback executes the model call and returns ModelResponse.
    Middleware can:
    - Call handler once for normal execution.
    - Call handler multiple times for retry logic.
    - Skip calling handler to short-circuit (return a cached result).
    - Modify request before calling handler.
    - Transform response after handler returns.
    
    Returns ModelResponse, AIMessage (auto-wrapped), or ExtendedModelResponse (with Command).
    """

async def awrap_model_call(
    self,
    request: ModelRequest[ContextT],
    handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
    """Async version of wrap_model_call."""
```

#### Wrap Tool Call (Tool Interception)

```python
def wrap_tool_call(
    self,
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
) -> ToolMessage | Command[Any]:
    """Intercept tool execution.
    
    Called once per tool call with the full ToolCallRequest (tool name, args, call id, state, runtime).
    Middleware can:
    - Call handler for normal execution.
    - Call handler multiple times for retries.
    - Skip handler to return a cached result or mock.
    - Modify tool args before handler.
    - Convert exceptions to error ToolMessages.
    
    Returns ToolMessage or Command.
    """

async def awrap_tool_call(
    self,
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
) -> ToolMessage | Command[Any]:
    """Async version of wrap_tool_call."""
```

### Configuration

```python
class AgentMiddleware(Generic[StateT, ContextT, ResponseT]):
    state_schema: type[StateT] = AgentState[Any]
    """Optional custom state schema. Merged with base AgentState during graph compilation."""
    
    tools: Sequence[BaseTool] = ()
    """Additional tools registered by this middleware (e.g., ShellToolMiddleware adds shell_tool)."""
    
    trace_policy: TracePolicy | None = None
    """Optional trace policy controlling what is captured in spans for this middleware's hooks."""
    
    transformers: Sequence[TransformerFactory] = ()
    """Stream transformer factories for streaming customization."""
```

## Core Middleware Types and Patterns

### Model Execution Middleware

**ModelRetryMiddleware**: Automatically retries failed model calls with exponential backoff when transient errors (rate limits, API timeouts) occur. Supports custom exception filtering, custom failure handlers, and jitter to avoid thundering herd.

```python
from langchain.agents.middleware import ModelRetryMiddleware

retry = ModelRetryMiddleware(
    max_retries=3,
    retry_on=(RateLimitError, APITimeoutError),
    backoff_factor=2.0,
    initial_delay=1.0,
    on_failure="continue",  # or "error" to re-raise, or callable
)
```

**ModelFallbackMiddleware**: Wraps the primary model with a fallback model (typically smaller/faster/cheaper) when the primary fails or to diversify model selection strategies.

**ModelCallLimitMiddleware**: Enforces a maximum number of model calls per agent invocation, preventing runaway loops.

### Tool Execution Middleware

**ToolErrorMiddleware**: Converts tool execution exceptions (e.g., API errors, validation failures) into error `ToolMessage` objects sent back to the model, allowing the model to recover or clarify. Opt-in by exception type.

```python
from langchain.agents.middleware import ToolErrorMiddleware

def on_error(exc: Exception, request: ToolCallRequest) -> str | None:
    if isinstance(exc, ValueError):
        return f"Invalid argument for {request.tool_call['name']}: {exc}"
    return None  # Propagate other exceptions

middleware = ToolErrorMiddleware(on_error=on_error)
```

**ToolRetryMiddleware**: Retries failed tool calls with configurable backoff and validation logic.

**ToolCallLimitMiddleware**: Prevents infinite tool loops by enforcing a maximum number of tool calls.

### Control Flow & Approval Middleware

**HumanInTheLoopMiddleware**: Pauses after model-requested tool calls and sends an interrupt with action summaries to a human reviewer. Supports approval, editing, rejection, or human-answered "respond" decision types. Tool calls are modified based on human feedback before execution.

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

hitl = HumanInTheLoopMiddleware(
    interrupt_on={
        "delete_file": True,  # All decision types allowed
        "search": {
            "allowed_decisions": ["approve", "reject"],
            "description": "Searching online databases",
        },
    }
)
```

### Data Transformation & Privacy Middleware

**PIIMiddleware**: Detects personally identifiable information (PII) in prompts, model responses, and tool outputs; optionally redacts or transforms it. Supports configurable detectors and redaction rules.

**ContextEditingMiddleware**: Allows dynamic modification of agent context (system message, available tools) during execution.

**FileSearchMiddleware**: Integrates file search capabilities into the agent, retrieving relevant documents before model calls.

### System Tools Middleware

**ShellToolMiddleware**: Provides controlled shell command execution with configurable resource limits, sandboxing (host, Docker, Codex), timeout enforcement, and output truncation.

**TodoListMiddleware**: Adds persistent todo list management capability.

## Request and Response Types

### ModelRequest

```python
@dataclass
class ModelRequest(Generic[ContextT]):
    """Request context for model execution."""
    
    model: BaseChatModel          # The model to invoke
    messages: list[AnyMessage]    # Messages (excluding system message)
    system_message: SystemMessage | None  # System instructions
    tool_choice: Any              # Tool selection strategy
    tools: list[BaseTool | dict]  # Available tools
    response_format: ResponseFormat[Any] | None  # Structured output format
    state: AgentState[Any]        # Full agent state
    runtime: Runtime[ContextT]    # Runtime context
    model_settings: dict[str, Any]  # Additional model parameters
```

**Immutable Pattern**: Middleware should not mutate `ModelRequest` directly. Use `request.override(**changes)` to create a new request with modifications.

### ModelResponse

```python
@dataclass
class ModelResponse(Generic[ResponseT]):
    """Successful model execution result."""
    
    result: list[BaseMessage]                  # Messages (typically one AIMessage)
    structured_response: ResponseT | None = None  # Parsed structured output
```

### ExtendedModelResponse

```python
@dataclass
class ExtendedModelResponse(Generic[ResponseT]):
    """Model response with optional LangGraph Command for additional state updates."""
    
    model_response: ModelResponse[ResponseT]
    command: Command[Any] | None = None
```

Middleware can return `ExtendedModelResponse` to apply a command that modifies state after the model node completes. Commands are applied through state reducers, so messages in commands are **added** to existing messages (not replaced).

### ToolCallRequest

```python
@dataclass
class ToolCallRequest:
    """Request context for tool execution."""
    
    tool_call: dict  # Tool call dict with 'id', 'name', 'args'
    tool: BaseTool | None  # Resolved BaseTool instance (or None in batch mode)
    state: AgentState[Any]  # Agent state at time of call
    runtime: Runtime[ContextT]  # Runtime context (ToolRuntime with tool-specific info)
```

## Composition and Execution Order

### Middleware Stack Execution

When middleware is registered as `[M1, M2, M3]`:

**Model Call Stack**:
1. M1's `wrap_model_call` is entered first
2. M1 calls handler → M2's `wrap_model_call` is entered
3. M2 calls handler → M3's `wrap_model_call` is entered
4. M3 calls handler → actual model execution
5. M3 returns result to M2
6. M2 can transform result and returns to M1
7. M1 can transform result and returns to agent

**Result**: Innermost middleware (M3, closest to model) executes first; outermost (M1) sees and can override all inner results.

### State Updates and Commands

State updates from hooks are merged using LangGraph reducers. For the `messages` field (which uses `add_messages` reducer), updates accumulate rather than replace.

**Command Accumulation**: When middleware returns `ExtendedModelResponse` with `Command`, multiple commands are accumulated in a list (inner-first, then outer). The agent applies them sequentially after the model node completes.

**Reducer Semantics**: Non-reducer fields in later commands override earlier ones (outermost middleware wins). The `messages` field is special: reducer-based fields like `messages` accumulate through `add_messages`.

## Writing Custom Middleware

### Simple Example: Logging Middleware

```python
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        num_messages = len(state.get("messages", []))
        print(f"[before_model] {num_messages} messages in state")
        return None  # No state updates

    def after_model(self, state, runtime):
        last_msg = state["messages"][-1] if state["messages"] else None
        if hasattr(last_msg, 'tool_calls'):
            print(f"[after_model] Model requested {len(last_msg.tool_calls)} tools")
        return None
```

### Retry Logic with Exponential Backoff

```python
import asyncio
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

class CustomRetryMiddleware(AgentMiddleware):
    def __init__(self, max_retries: int = 2, backoff_factor: float = 2.0):
        super().__init__()
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
                    # Return error message instead of raising
                    return ModelResponse(
                        result=[AIMessage(content=f"Model call failed after {self.max_retries} retries: {e}")]
                    )
```

### Tool Argument Validation and Modification

```python
from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage

class ArgumentValidationMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        tool_call = request.tool_call
        args = tool_call["args"]
        
        # Example: ensure all string args are lowercase for a search tool
        if tool_call["name"] == "search":
            args = {k: v.lower() if isinstance(v, str) else v for k, v in args.items()}
            request = request.override(tool_call={**tool_call, "args": args})
        
        return handler(request)
```

### Conditional Tool Interception Based on State

```python
from langchain.agents.middleware import AgentMiddleware, ToolCallRequest

class ConditionalToolMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        # Inspect agent state
        state = request.state
        user_message = next(
            (m.content for m in reversed(state["messages"]) if hasattr(m, "content")),
            None
        )
        
        # Only allow certain tools if user explicitly requests them
        if "search" not in user_message.lower() and request.tool_call["name"] == "search":
            return ToolMessage(
                content="Search tool requires explicit user approval.",
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
                status="error",
            )
        
        return handler(request)
```

## Async and Sync Implementations

Middleware can provide sync-only, async-only, or both implementations. The agent will:

- Call sync versions (`wrap_model_call`, `wrap_tool_call`) in sync contexts (`stream()`, `invoke()`).
- Call async versions (`awrap_model_call`, `awrap_tool_call`) in async contexts (`astream()`, `ainvoke()`).
- Raise `NotImplementedError` if the required implementation is missing for the execution path.

**Best Practice**: Implement both versions unless the middleware is inherently async-only (e.g., uses async I/O).

## Integration with Agent Factory

Middleware is passed to `create_agent()` as a list:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolErrorMiddleware, HumanInTheLoopMiddleware

agent = create_agent(
    model=ChatOpenAI(),
    tools=[search, calculator],
    middleware=[
        ModelRetryMiddleware(max_retries=2),  # Outermost: retry model calls
        ToolErrorMiddleware(on_error=my_error_handler),  # Middle: handle tool errors
        HumanInTheLoopMiddleware(interrupt_on={"delete": True}),  # Innermost: approve risky tools
    ],
)
```

**Composition Rule**: First in the list = outermost (highest priority for interception and response transformation).

## Tracing and Observability

### Trace Policy

Middleware can configure what gets traced via `trace_policy`:

```python
from langchain.agents.middleware import TracePolicy, omit_payload

class MyMiddleware(AgentMiddleware):
    trace_policy = TracePolicy(process_inputs=omit_payload)
    # Omit input payloads from traces to reduce noise, keep spans and timing
```

### Hook Naming in Traces

Middleware hooks are automatically named in trace spans as `{middleware_name}.{hook_name}`, e.g., `ModelRetryMiddleware.wrap_model_call`, making it easy to identify which middleware handled each span.

## Common Patterns and Anti-Patterns

### ✅ Good Patterns

1. **Immutable Requests**: Use `request.override()` instead of mutating fields.
2. **Opt-in Error Handling**: Return `None` from error handlers to propagate exceptions (don't silently swallow).
3. **State-Driven Decisions**: Use `request.state` and `runtime` to make decisions; avoid global state.
4. **Clear Composition Order**: Document which middleware should run first, especially when order matters (e.g., retry before error handling).
5. **Fallback Behavior**: Implement sync and async versions to support both execution paths.

### ❌ Anti-Patterns

1. **Mutating Request/Response Objects**: Direct assignment to `ModelRequest` fields is deprecated; always use `override()`.
2. **Silent Failures**: Don't catch and suppress exceptions in `on_error` handlers without returning content; let exceptions propagate if not handled.
3. **Hard-Coded Assumptions**: Don't assume specific tool names or message formats without validation.
4. **Blocking Async**: Never use `asyncio.run()` or sync I/O in async middleware implementations.
5. **Over-Composition**: Don't add more middleware than necessary; each layer adds latency.

## Key Invariants and Lifecycle

- **Immutability**: `ModelRequest` and `ToolCallRequest` follow an immutable pattern; use override/replace methods.
- **Execution Order**: Middleware runs in registration order (first = outermost). Multiple calls within a middleware (retries) do not reorder subsequent middleware.
- **State Reducer Semantics**: State updates via dict returns merge using reducer semantics. Messages accumulate; non-reducer fields are overwritten by the most recent update.
- **Command Flow**: `ExtendedModelResponse` commands are collected (inner-to-outer) and applied after the model node completes.
- **Sync/Async Consistency**: Choosing execution path (sync or async) is determined at agent invocation time; middleware cannot switch contexts mid-execution.
- **Exception Propagation**: Exceptions that are not explicitly handled propagate to the caller, unless a middleware converts them to a message or command.

## See Also

- [Agent Execution Flow and Loop Control](/openwiki/agent-execution.md) – Detailed description of the agent loop, state management, and where middleware hooks are invoked.
- [Agent Factory and Graph Construction](/openwiki/agent-factory.md) – How the agent graph is built, including middleware integration and handler composition.
