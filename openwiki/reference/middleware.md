---
type: Reference
title: "Agent Middleware System"
description: "Reference guide to agent middleware hooks, composition semantics, request/response types, and handler lifecycle in LangChain agents."
tags: [agent-middleware, lifecycle-hooks, request-interception, composition]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-18T08:17:15.497Z
sources:
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
  - id: openwiki-source-03e8ca0eebe37feda8566793
    resource: repo://libs/langchain_v1/langchain/agents/middleware/types.py
generated: { by: "openwiki/0.5.0", at: "2026-11-18T08:17:15.497Z" }
---

## Overview

Agent middleware in LangChain provides a composable, layered approach to intercepting and modifying agent behavior at well-defined execution points without changing core agent logic. Middleware registers handlers for lifecycle hooks spanning initialization (`before_agent`), per-iteration model invocation (`before_model`, `wrap_model_call`, `after_model`), tool execution (`wrap_tool_call`), and completion (`after_agent`). Multiple middleware instances compose by wrapping handlers in registration order, with the first-registered middleware becoming the outermost layer, enabling cross-cutting concerns such as automatic retries with backoff, error handling and recovery, human-in-the-loop approval workflows, tool argument validation, and request/response transformation.

Middleware can be created by subclassing `AgentMiddleware` or by using convenience decorators (`@before_agent`, `@after_agent`, `@before_model`, `@after_model`, `@wrap_model_call`, `@wrap_tool_call`, `@dynamic_prompt`, `@hook_config`) that generate middleware instances from plain functions.

## AgentMiddleware Base Class

The `AgentMiddleware` class is the foundation for all middleware implementations. It provides virtual methods for each lifecycle hook and optional attributes for tools, trace policies, and stream transformers.

### Class Overview

```python
class AgentMiddleware(Generic[StateT, ContextT, ResponseT]):
    """Base middleware class for an agent."""
    
    state_schema: type[StateT] = AgentState
    """The schema for state passed to the middleware nodes."""
    
    tools: Sequence[BaseTool] = []
    """Additional tools registered by the middleware."""
    
    trace_policy: TracePolicy | None = None
    """Optional trace policy for hook spans."""
    
    transformers: Sequence[TransformerFactory] = ()
    """Stream transformer factories registered by the middleware."""
    
    @property
    def name(self) -> str:
        """The name of the middleware instance (defaults to class name)."""
```

### Hook Methods

All `before_*` and `after_*` hooks return `dict[str, Any] | None` for optional state updates. Both sync and async versions are available (e.g., `before_model` and `abefore_model`).

**Lifecycle Hooks:**
- `before_agent(state, runtime)` / `abefore_agent(...)`: Runs once when agent execution begins, before the first model call.
- `before_model(state, runtime)` / `abefore_model(...)`: Runs before each model invocation. Can update state or set `jump_to` for control flow.
- `after_model(state, runtime)` / `aafter_model(...)`: Runs after model response is received. Can filter tool calls or update state.
- `after_agent(state, runtime)` / `aafter_agent(...)`: Runs after agent loop terminates (no more tool calls or explicit exit).

**Interception Hooks:**
- `wrap_model_call(request, handler)` / `awrap_model_call(...)`: Intercepts model execution. Handler callback executes the model; middleware can retry, modify request, transform response, or short-circuit.
- `wrap_tool_call(request, handler)` / `awrap_tool_call(...)`: Intercepts tool execution. Handler callback executes the tool; middleware can retry, validate arguments, or handle errors.

## State and Control Flow

### AgentState Schema

The default agent state contains:
```python
class AgentState(TypedDict):
    messages: list[AnyMessage]  # required, uses add_messages reducer
    jump_to: JumpTo | None      # optional, ephemeral, internal
    structured_response: Any    # optional, omitted from input schema
```

Where `JumpTo` is `Literal["tools", "model", "end"]` for control flow.

### State Access

Middleware accesses state through:
- **Lifecycle hooks**: `state: StateT` parameter (read and update via return dict)
- **Request interception**: `request.state` (read-only within model/tool call)
- **Runtime context**: `request.runtime` or `runtime` parameter (context, stream writer, etc.)

### Jump-to Control Flow

Lifecycle hooks (`before_model`, `after_model`, etc.) can return a dict with `jump_to` field to control agent flow:

```python
@before_model(can_jump_to=["end"])
def conditional_exit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if should_exit_early(state):
        return {"jump_to": "end"}  # Skip to agent completion
    return None
```

Valid destinations are:
- `"model"`: Retry/jump back to model invocation
- `"tools"`: Jump to tool execution (skips model)
- `"end"`: Exit the agent loop immediately

**Note**: Jumping capability must be declared upfront via `can_jump_to` parameter on lifecycle hook decorators or via `@hook_config(can_jump_to=[...])` on class methods. Interception hooks (`wrap_model_call`, `wrap_tool_call`) cannot use `jump_to` directly; they must return model/tool responses or use `Command` for state updates.

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

**Immutable Pattern**: `ModelRequest` enforces immutability via `__setattr__` to discourage direct mutation. Middleware must use `request.override(**changes)` to create a new request with modifications:

```python
# Access fields
print(request.model, request.messages, request.system_message)

# Create modified request (creates new instance)
new_request = request.override(
    system_message=SystemMessage(content="New prompt"),
    model_settings={"temperature": 0.5}
)
```

Supported override keys: `model`, `system_message`, `messages`, `tool_choice`, `tools`, `response_format`, `model_settings`, `state`. Direct assignment like `request.messages = [...]` raises `DeprecationWarning` and directs users to `override()`.

### ModelResponse

Result from model execution:

```python
@dataclass
class ModelResponse(Generic[ResponseT]):
    result: list[BaseMessage]               # Messages (typically one AIMessage)
    structured_response: ResponseT | None = None  # Parsed structured output if response_format was specified
```

### ExtendedModelResponse

Model response with optional `Command` for additional state updates:

```python
@dataclass
class ExtendedModelResponse(Generic[ResponseT]):
    model_response: ModelResponse[ResponseT]
    command: Command[Any] | None = None
```

**Command Semantics**: Middleware can return `ExtendedModelResponse` to apply a `Command` that modifies state after the model node completes. Commands are applied through state reducers, so messages in commands are **added** to existing messages (not replaced). Non-reducer state fields follow outermost-wins semantics: when multiple middleware return commands, the outermost middleware's values for non-reducer fields override inner ones.

**Unsupported Commands**: Commands returned from `wrap_model_call` do not yet support `goto`, `resume`, or `graph` directives; attempting to use them raises `NotImplementedError`.

### ToolCallRequest

Request context for tool execution (from LangGraph):

```python
@dataclass
class ToolCallRequest:
    tool_call: dict                         # Tool call dict with 'id', 'name', 'args'
    tool: BaseTool | None                   # Resolved BaseTool instance
    state: AgentState[Any]                  # Agent state at time of call
    runtime: ToolRuntime                    # Runtime context with tool-specific info
    
    def override(self, tool_call: dict) -> ToolCallRequest:
        """Create modified request with new tool_call dict."""
```

## Composition and Execution Order

### Middleware Stack

When middleware is registered as `[M1, M2, M3]`, the first becomes the outermost layer:

```
Execution order for wrap_model_call:
M1 (outermost) → M2 → M3 (innermost) → actual model execution
Results flow back: M3 can transform → M2 can transform → M1 can transform and return to agent

Lifecycle hooks execute in same order (M1, M2, M3) since there's no nesting.
```

**Key Point**: Innermost middleware (closest to model) sees the result first after execution; outermost middleware sees and can override all inner transformations. For retries and error recovery, the outermost layer is typically best.

### State Updates and Commands

Lifecycle hooks return state updates merged via LangGraph reducers:

- **Messages field**: Uses `add_messages` reducer, so messages accumulate across updates
- **Non-reducer fields**: Overwritten by the most recent update (last update wins)

**Command Accumulation**: When middleware returns `ExtendedModelResponse` with `Command`, multiple commands are accumulated in a list (inner-to-outer order) and applied sequentially after the model node completes through LangGraph state reducers. This ensures messages from all middleware are captured, while non-reducer fields respect outermost-wins precedence.

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

- **Sync contexts** (`stream()`, `invoke()`): Calls sync versions (`wrap_model_call`, `wrap_tool_call`, `before_model`, etc.)
- **Async contexts** (`astream()`, `ainvoke()`): Calls async versions (`awrap_model_call`, `awrap_tool_call`, `abefore_model`, etc.)
- **Missing implementation**: Raises `NotImplementedError` with guidance on how to resolve

**Best Practice**: Implement both versions unless the middleware is inherently async-only (uses async I/O). Decorator-based middleware can detect sync vs async functions automatically and generate appropriate wrappers.

**Special Case - dynamic_prompt**: When a sync `@dynamic_prompt` function is provided, the decorator automatically generates both `wrap_model_call` (sync) and `awrap_model_call` (async via delegation to sync) methods, supporting both execution modes from a single sync function.

## Error Propagation

Exceptions raised inside middleware (or by the wrapped operation) propagate up the stack unless explicitly handled:

- Middleware can convert exceptions to messages/errors or re-raise them
- Control-flow signals (`jump_to`, `Command`) bypass error handling and always propagate
- Opt-in by design: don't silently swallow exceptions without explicitly converting them

Example error handling in `wrap_model_call`:

```python
@wrap_model_call
def graceful_fallback(request, handler):
    try:
        return handler(request)
    except TimeoutError:
        # Convert exception to response
        return ModelResponse(
            result=[AIMessage(content="Request timed out, using default response")]
        )
```

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

## Middleware Decorators

LangChain provides convenience decorators to create middleware from plain functions, eliminating the need to subclass `AgentMiddleware`. Each decorator generates a middleware class dynamically at runtime.

### Lifecycle Hook Decorators

#### `@before_agent` / `@after_agent`

Runs at the start or end of agent execution. Returns optional state updates or commands.

```python
@before_agent
def initialize_session(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    return {"messages": [SystemMessage(content="Session initialized")]}

# With conditional jumping
@before_agent(can_jump_to=["end"])
def early_exit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if should_exit_early(state):
        return {"jump_to": "end"}
    return None

@after_agent
def log_completion(state: AgentState, runtime: Runtime) -> None:
    print(f"Agent completed with {len(state['messages'])} messages")
```

**Parameters:**
- `state_schema: type[StateT] | None` – Custom state schema (default: `AgentState`)
- `tools: list[BaseTool] | None` – Additional tools to register
- `can_jump_to: list[JumpTo] | None` – Valid jump destinations (`["tools", "model", "end"]`)
- `name: str | None` – Generated middleware class name (default: function name)

#### `@before_model` / `@after_model`

Runs before/after each model invocation. Can update state, set `jump_to`, or emit streaming events.

```python
@before_model
async def notify_thinking(state: AgentState, runtime: Runtime) -> None:
    runtime.stream_writer({"type": "status", "message": "Thinking..."})

@after_model
def log_model_choice(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return {"log_entry": f"Called {len(last_msg.tool_calls)} tools"}
    return None
```

**Parameters:** Same as `@before_agent` / `@after_agent`.

### Interception Hook Decorators

#### `@wrap_model_call`

Intercepts model execution with handler callback. Can retry, modify request, transform response, or short-circuit.

```python
@wrap_model_call
def retry_with_backoff(request: ModelRequest, handler) -> ModelResponse:
    for attempt in range(3):
        try:
            return handler(request)
        except (TimeoutError, ConnectionError) as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise

# Async version
@wrap_model_call
async def async_retry(request: ModelRequest, handler) -> ModelResponse:
    for attempt in range(3):
        try:
            return await handler(request)
        except Exception:
            if attempt == 2:
                raise
```

**Parameters:**
- `state_schema: type[StateT] | None` – Custom state schema (default: `AgentState`)
- `tools: list[BaseTool] | None` – Additional tools to register
- `name: str | None` – Generated middleware class name (default: function name)

#### `@wrap_tool_call`

Intercepts tool execution with handler callback. Can retry, validate arguments, or handle errors.

```python
@wrap_tool_call
def validate_args(request: ToolCallRequest, handler) -> ToolMessage | Command:
    # Normalize tool arguments
    if request.tool_call["name"] == "search":
        args = request.tool_call.get("args", {})
        args["query"] = args.get("query", "").lower().strip()
        request = request.override(tool_call={**request.tool_call, "args": args})
    return handler(request)

# Async version with retry
@wrap_tool_call
async def async_retry(request: ToolCallRequest, handler) -> ToolMessage:
    for attempt in range(3):
        try:
            return await handler(request)
        except Exception:
            if attempt == 2:
                raise
```

**Parameters:** Same as `@wrap_model_call`.

#### `@dynamic_prompt`

Convenience decorator for dynamically generating system prompts. Receives `ModelRequest` and returns string or `SystemMessage`.

```python
@dynamic_prompt
def context_aware_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context.get("user_name", "User")
    msg_count = len(request.state["messages"])
    if msg_count > 10:
        return f"You are helping {user_name}. Be concise in long conversations."
    return f"You are a helpful assistant helping {user_name}."

# Async version
@dynamic_prompt
async def async_prompt(request: ModelRequest) -> str:
    # Can perform async operations like fetching user preferences
    preferences = await fetch_user_preferences(request.runtime.context["user_id"])
    return f"User preferences: {preferences}\nBe helpful."

# Using with agent
agent = create_agent(model, middleware=[context_aware_system_prompt])
```

**Behavior**: The decorator automatically creates a `wrap_model_call` middleware that intercepts the request, overrides the system message, and calls the handler. For sync functions, both sync and async methods are generated for compatibility with both execution modes.

#### `@hook_config`

Decorator to configure hook behavior on middleware class methods. Currently supports specifying jump destinations.

```python
class MyMiddleware(AgentMiddleware):
    @hook_config(can_jump_to=["end", "model"])
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if some_condition(state):
            return {"jump_to": "end"}
        return None
```

**Parameters:**
- `can_jump_to: list[JumpTo] | None` – Valid jump destinations

Alternative: Use the `can_jump_to` parameter in `@before_model`, `@after_model`, etc. decorators directly (simpler for decorator-based middleware).

## Built-in Middleware Implementations

LangChain provides several out-of-the-box middleware implementations for common patterns. See individual middleware documentation for details:

- **ModelRetryMiddleware**: Retries model calls with exponential backoff on transient errors
- **ModelFallbackMiddleware**: Falls back to alternative model on failure
- **ToolRetryMiddleware**: Retries tool execution with configurable logic
- **ToolErrorMiddleware**: Converts tool errors to messages or interrupts
- **HumanInTheLoopMiddleware**: Interrupts agent for human approval of tool calls or actions
- **ModelCallLimitMiddleware**: Enforces maximum number of model calls
- **ToolCallLimitMiddleware**: Enforces maximum number of tool calls
- **PIIMiddleware**: Detects and masks personally identifiable information
- **SummarizationMiddleware**: Summarizes long conversation histories
- **ModelRouterMiddleware**: Routes to different models based on conditions
- **ToolSelectorMiddleware**: Dynamically selects tools based on context
- **ToolEmulatorMiddleware**: Emulates tools with language model responses

## Examples

### Complete Retry and Fallback Middleware

```python
class RobustModelMiddleware(AgentMiddleware):
    def __init__(self, max_retries: int = 2, fallback_model=None):
        self.max_retries = max_retries
        self.fallback_model = fallback_model
    
    def wrap_model_call(self, request, handler):
        # Try primary model with retries
        for attempt in range(self.max_retries + 1):
            try:
                return handler(request)
            except (TimeoutError, ConnectionError) as e:
                if attempt < self.max_retries:
                    import time
                    time.sleep(0.5 * (2 ** attempt))
                elif self.fallback_model:
                    # Try fallback model
                    new_request = request.override(model=self.fallback_model)
                    return handler(new_request)
                else:
                    raise
```

### State-Aware Tool Validation

```python
@wrap_tool_call
def validate_tool_access(request: ToolCallRequest, handler) -> ToolMessage:
    # Check if user is authorized for this tool
    user_id = request.runtime.context.get("user_id")
    tool_name = request.tool_call["name"]
    
    if not is_user_authorized(user_id, tool_name):
        return ToolMessage(
            content=f"Access denied to tool {tool_name}",
            tool_call_id=request.tool_call["id"],
            name=tool_name,
            status="error"
        )
    
    return handler(request)
```

### Composing Middleware for Complex Behavior

```python
# Create agent with multiple middleware for different concerns
agent = create_agent(
    model=ChatOpenAI(),
    tools=[web_search, calculator],
    middleware=[
        # Outermost: Catch all errors and log
        error_logger_middleware,
        
        # Retry with backoff on transient errors
        ModelRetryMiddleware(max_retries=2),
        
        # Validate tool calls before execution
        validate_tool_access,
        
        # Innermost: Dynamic prompts based on context
        context_aware_prompt_middleware,
    ]
)
```

The order ensures that error logging wraps retries, which wrap validation, which wraps the dynamic prompt logic.
