---
type: Reference
title: "Middleware Types and Interfaces"
description: "Reference documentation for middleware types, interfaces, decorators, and composition patterns. Details AgentMiddleware protocol, lifecycle hooks, request/response types, state schema annotations, and decorator-based middleware construction."
tags: [agent-middleware, types, interfaces, decorators, state-schema]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-07T08:15:07.286Z
sources:
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
  - id: openwiki-source-03e8ca0eebe37feda8566793
    resource: repo://libs/langchain_v1/langchain/agents/middleware/types.py
generated: { by: "openwiki/0.5.0", at: "2026-11-07T08:15:07.286Z" }
---

## Overview

The middleware reference documents the core types, interfaces, and decorators that compose the LangChain agent middleware system. This page provides detailed API documentation for implementing custom middleware and understanding type structures. For comprehensive guides covering lifecycle hooks, composition semantics, error handling, and implementation patterns, see [Agent Middleware](/openwiki/middleware.md).

## AgentMiddleware Base Class

`AgentMiddleware` is the generic base class for all middleware. Subclass it and implement any subset of the defined hook methods to customize agent behavior.

```python
class AgentMiddleware(Generic[StateT, ContextT, ResponseT]):
    """Base middleware class for an agent.
    
    Type Parameters:
        StateT: The type of the agent state. Defaults to AgentState[Any].
        ContextT: The type of the runtime context. Defaults to None.
        ResponseT: The type of the structured response. Defaults to Any.
    """
```

### Type Parameters

- **StateT**: The agent state type passed to lifecycle hooks and interception handlers. Defaults to `AgentState[Any]`. Custom state schemas extend the default with additional fields.
- **ContextT**: The runtime context type available via `Runtime[ContextT]`. Defaults to `None` for simple cases; set to a custom type to pass application context through the middleware stack.
- **ResponseT**: The structured response type used in `ModelResponse[ResponseT]` when the model returns parsed structured output. Defaults to `Any`.

### Class Attributes

```python
state_schema: type[StateT] = AgentState[Any]
```
The state schema for this middleware. If custom, it is merged with base `AgentState` during graph compilation. Allows middleware to declare additional state fields.

```python
tools: Sequence[BaseTool] = ()
```
Additional tools registered by this middleware. For example, `ShellToolMiddleware` adds shell execution tools here.

```python
trace_policy: TracePolicy | None = None
```
Optional trace policy for hook spans (`wrap_model_call`, `wrap_tool_call`, and lifecycle node hooks). By default (`None`), spans are traced normally. Set to configure what is recorded—e.g., `TracePolicy(process_inputs=omit_payload)` to drop conversation payloads while preserving timing.

```python
transformers: Sequence[TransformerFactory] = ()
```
Stream transformer factories for customizing streaming behavior. Factories are invoked as `factory(scope)` at graph compile time to receive fresh instances.

```python
@property
def name(self) -> str
```
The name of the middleware instance. Defaults to the class name; override in subclasses for custom naming.

## Lifecycle Hooks

Lifecycle hooks run synchronously with the graph at discrete execution points. They receive state and runtime, optionally return state updates or commands, and compose by wrapping in registration order.

### before_agent / abefore_agent

Runs once before agent execution starts, after initial state is set.

```python
def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """Runs before agent execution begins.
    
    Args:
        state: The initial agent state.
        runtime: The runtime context.
        
    Returns:
        State updates to merge, Command for flow control, or None.
    """

async def abefore_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """Async version of before_agent."""
```

Use cases: initialization, resource setup, context validation, custom streaming events.

### before_model / abefore_model

Runs before each model invocation in the agent loop. Can read or update state, and optionally redirect execution via `jump_to`.

```python
def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """Runs before each model call.
    
    Can return {"jump_to": "tools" | "model" | "end"} to redirect control flow.
    """

async def abefore_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """Async version of before_model."""
```

Use cases: state logging, input validation, early exit logic, conditional flow control.

### after_model / aafter_model

Runs after each model invocation. Can inspect or filter tool calls, update state.

```python
def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """Runs after model response.
    
    Can inspect last message, filter tool calls, redirect via jump_to.
    """

async def aafter_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """Async version of after_model."""
```

Use cases: response filtering, tool call validation, fallback logic, state transformation.

### after_agent / aafter_agent

Runs after agent execution completes (no more tool calls or explicit exit).

```python
def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """Runs after agent loop terminates.
    
    Args:
        state: Final agent state with all messages.
        runtime: The runtime context.
        
    Returns:
        Final state updates or Command.
    """

async def aafter_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """Async version of after_agent."""
```

Use cases: cleanup, result formatting, final logging, resource release.

## Interception Hooks

Interception hooks receive a request object and a handler callback. The middleware controls execution by calling the handler zero or more times, modifying the request/response, or short-circuiting with cached results.

### wrap_model_call / awrap_model_call

Intercepts and controls model execution. Called before each model invocation with the complete request context.

```python
def wrap_model_call(
    self,
    request: ModelRequest[ContextT],
    handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
    """Intercept model execution via handler callback.
    
    The handler callback executes the model request and returns ModelResponse.
    Middleware can:
    - Call handler once for normal execution
    - Call handler multiple times for retry logic
    - Skip calling handler to short-circuit (return cached or fallback)
    - Modify request before calling handler via request.override(...)
    - Transform response after handler returns
    
    Returns:
        ModelResponse: Full response with messages and optional structured output
        AIMessage: Simplified return (auto-wrapped to ModelResponse)
        ExtendedModelResponse: Response with optional Command for state updates
    """

async def awrap_model_call(
    self,
    request: ModelRequest[ContextT],
    handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
    """Async version of wrap_model_call."""
```

Handler signature: accepts `ModelRequest`, returns `ModelResponse[ResponseT]`.

Multiple calls to handler support retry logic. Each call is independent and stateless. Exceptions raised by handler propagate up the middleware stack unless explicitly handled.

Return type flexibility: `AIMessage` is automatically converted to `ModelResponse`; `ExtendedModelResponse` allows returning a `Command` for additional state updates or control flow.

### wrap_tool_call / awrap_tool_call

Intercepts tool execution. Called once per tool call with the complete tool call context.

```python
def wrap_tool_call(
    self,
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
) -> ToolMessage | Command[Any]:
    """Intercept tool execution for retries, monitoring, or modification.
    
    The handler callback executes the tool and returns ToolMessage or Command.
    Middleware can:
    - Call handler for normal execution
    - Call handler multiple times for retries
    - Skip handler to short-circuit (return cached result or mock)
    - Modify tool call dict before calling handler via request.override(...)
    - Convert exceptions to error ToolMessages
    - Return Command to modify state or control flow
    
    Returns:
        ToolMessage: Tool execution result (success or error)
        Command: LangGraph command for state updates or flow control
    """

async def awrap_tool_call(
    self,
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
) -> ToolMessage | Command[Any]:
    """Async version of wrap_tool_call."""
```

Handler signature: accepts `ToolCallRequest`, returns `ToolMessage | Command[Any]`.

Each call to handler is independent and stateless. Exceptions propagate unless `handle_tool_errors` is configured on `ToolNode`.

## Request and Response Types

### ModelRequest

Encapsulates all information needed to execute a model invocation. Immutable—use `override()` to create modified copies.

```python
@dataclass
class ModelRequest(Generic[ContextT]):
    """Model request information for the agent.
    
    Type Parameters:
        ContextT: The type of the runtime context.
    """
    
    model: BaseChatModel
    """The chat model to invoke."""
    
    messages: list[AnyMessage]
    """List of conversation messages (excluding system message)."""
    
    system_message: SystemMessage | None
    """System message (preferred). Use override(system_message=...) to change."""
    
    tool_choice: Any | None
    """Tool choice configuration (e.g., 'auto', 'required', tool name)."""
    
    tools: list[BaseTool | dict[str, Any]]
    """List of available tools."""
    
    response_format: ResponseFormat[Any] | None
    """Structured output format specification."""
    
    state: AgentState[Any]
    """Full agent state at time of request."""
    
    runtime: Runtime[ContextT]
    """Runtime context with app-specific data."""
    
    model_settings: dict[str, Any] = field(default_factory=dict)
    """Additional model-specific parameters."""
```

**Immutability**: Direct attribute assignment is deprecated and raises `DeprecationWarning`. Use `override()` to create modified requests:

```python
# ✅ Correct
new_request = request.override(
    system_message=SystemMessage(content="New instructions"),
    tools=[filtered_tools]
)

# ❌ Deprecated: direct assignment raises DeprecationWarning
request.system_message = ...
```

**Override method signature**:

```python
def override(self, **overrides: Unpack[_ModelRequestOverrides]) -> ModelRequest[ContextT]:
    """Create new request with specified overrides.
    
    Supported override keys:
    - model: BaseChatModel
    - system_message: SystemMessage | None (preferred over system_prompt)
    - messages: list[AnyMessage]
    - tool_choice: Any | None
    - tools: list[BaseTool | dict[str, Any]]
    - response_format: ResponseFormat[Any] | None
    - model_settings: dict[str, Any]
    - state: AgentState[Any]
    """
```

### ModelResponse

Represents the result of model execution.

```python
@dataclass
class ModelResponse(Generic[ResponseT]):
    """Response from model execution including messages and optional structured output.
    
    Type Parameters:
        ResponseT: The type of the structured response.
    """
    
    result: list[BaseMessage]
    """List of messages from model execution (typically one AIMessage)."""
    
    structured_response: ResponseT | None = None
    """Parsed structured output if response_format was specified, None otherwise."""
```

### ExtendedModelResponse

Model response paired with an optional LangGraph `Command` for state updates or control flow.

```python
@dataclass
class ExtendedModelResponse(Generic[ResponseT]):
    """Model response with optional Command from wrap_model_call middleware.
    
    Type Parameters:
        ResponseT: The type of the structured response.
    """
    
    model_response: ModelResponse[ResponseT]
    """The underlying model response."""
    
    command: Command[Any] | None = None
    """Optional LangGraph Command to apply additional state updates."""
```

**Command application semantics**:
- Commands are applied through LangGraph state reducers after the model node completes
- Messages in commands are **added** to existing messages (via `add_messages` reducer), not replaced
- Non-reducer state fields follow outermost-wins semantics: when multiple middleware return commands, outer middleware values override inner ones
- Note: `goto`, `resume`, and `graph` are not yet supported on these commands; using them raises `NotImplementedError`

### ModelCallResult

Union type for all valid return types from `wrap_model_call`:

```python
ModelCallResult = ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]
```

### ToolCallRequest

Request context for tool execution.

```python
@dataclass
class ToolCallRequest:
    """Request context for tool execution.
    
    Attributes are read-only; use override() to create modified copies.
    """
    
    tool_call: dict
    """Tool call dict with keys: 'id' (str), 'name' (str), 'args' (dict)."""
    
    tool: BaseTool | None
    """Resolved BaseTool instance (or None in batch execution mode)."""
    
    state: AgentState[Any]
    """Agent state at time of tool call."""
    
    runtime: ToolRuntime
    """Runtime context with tool-specific information."""
```

## Agent State

### AgentState

The core state schema used by all agents. Middleware can define custom state by subclassing or using `TypedDict`.

```python
class AgentState(TypedDict, Generic[ResponseT]):
    """State schema for the agent."""
    
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    """Conversation messages. Uses add_messages reducer; new messages accumulate."""
    
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    """Control flow signal: jump to 'tools', 'model', or 'end'. Ephemeral (not persisted)."""
    
    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]
    """Parsed structured output if response_format was specified. Omitted from input schema."""
```

**Reducer semantics**:
- `messages` field uses `add_messages` reducer: new messages are appended, not replaced
- `jump_to` is ephemeral and not persisted across invocations
- Non-reducer fields (like `structured_response`) are overwritten by the most recent update

### InputAgentState

Input schema for agents. Accepts both `AnyMessage` and message dicts.

```python
class InputAgentState(TypedDict):
    """Input state schema for the agent."""
    
    messages: Required[Annotated[list[AnyMessage | dict[str, Any]], add_messages]]
    """Input messages. Can be Message objects or dicts."""
```

### OutputAgentState

Output schema for agents.

```python
class OutputAgentState(TypedDict, Generic[ResponseT]):
    """Output state schema for the agent."""
    
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    """Final conversation messages."""
    
    structured_response: NotRequired[ResponseT]
    """Structured response if the agent produced one."""
```

## State Schema Annotations

Annotations control how state fields are exposed in input/output schemas and whether they are agent-internal.

### OmitFromSchema

Annotation to control schema exposure.

```python
@dataclass
class OmitFromSchema:
    """Annotation to mark state attributes as omitted from input or output schemas."""
    
    input: bool = True
    """Whether to omit the attribute from the input schema."""
    
    output: bool = True
    """Whether to omit the attribute from the output schema."""
```

### Predefined Annotations

```python
OmitFromInput = OmitFromSchema(input=True, output=False)
"""Mark field as omitted from input schema but included in output."""

OmitFromOutput = OmitFromSchema(input=False, output=True)
"""Mark field as included in input schema but omitted from output."""

PrivateStateAttr = OmitFromSchema(input=True, output=True)
"""Mark field as purely internal for a given middleware."""
```

**Usage example**:

```python
class CustomState(AgentState):
    internal_counter: NotRequired[
        Annotated[int, PrivateStateAttr]
    ]
    """Internal field not exposed in schemas."""
```

## Decorator-Based Middleware Construction

Decorators provide a convenient way to construct `AgentMiddleware` instances from functions. Each decorator dynamically creates and returns a middleware instance.

### @before_model / @abefore_model

Creates middleware with a `before_model` hook from a decorated function.

```python
@before_model(
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
)
def my_hook(state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """Function to decorate."""
```

**Parameters**:
- `state_schema`: Custom state type. Defaults to `AgentState`.
- `tools`: Additional tools to register.
- `can_jump_to`: List of valid jump destinations (`'tools'`, `'model'`, `'end'`). Required if hook returns `jump_to`.
- `name`: Middleware class name. Defaults to function name.

**Return value**: State updates (`dict`), `Command` for flow control, or `None`.

**Example**:

```python
@before_model(can_jump_to=["end"])
def conditional_exit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if should_exit(state):
        return {"jump_to": "end"}
    return None
```

### @after_model / @aafter_model

Creates middleware with an `after_model` hook.

```python
@after_model(
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
)
def my_hook(state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """Function to decorate."""
```

Same parameters and return semantics as `@before_model`.

### @before_agent / @aafter_agent

Creates middleware with a `before_agent` hook.

```python
@before_agent(
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
)
def my_hook(state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """Function to decorate."""
```

### @after_agent / @aafter_agent

Creates middleware with an `after_agent` hook.

```python
@after_agent(
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
)
def my_hook(state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """Function to decorate."""
```

### @dynamic_prompt

Creates middleware that dynamically generates system prompts. Convenience decorator for `wrap_model_call` specialized to prompt generation.

```python
@dynamic_prompt
def my_prompt(request: ModelRequest) -> str | SystemMessage:
    """Generate system prompt based on request context."""
```

**Returns**: A system prompt string or `SystemMessage` instance.

**Example**:

```python
@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    msg_count = len(request.state["messages"])
    if msg_count > 10:
        return "You are in a long conversation. Be concise."
    return "You are a helpful assistant."
```

### @wrap_model_call / @awrap_model_call

Creates middleware that intercepts model calls via handler callback.

```python
@wrap_model_call(
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
)
def my_hook(
    request: ModelRequest[ContextT],
    handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT] | Awaitable[ModelResponse[ResponseT]]],
) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
    """Intercept model execution."""
```

**Signature**: Function receives `request` and `handler` callback. Handler executes the model and returns `ModelResponse[ResponseT]`.

**Returns**: `ModelResponse`, `AIMessage` (auto-wrapped), or `ExtendedModelResponse`.

**Example**:

```python
@wrap_model_call
def retry_on_error(request, handler):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return handler(request)
        except Exception:
            if attempt == max_retries - 1:
                raise
```

### @wrap_tool_call / @awrap_tool_call

Creates middleware that intercepts tool calls via handler callback.

```python
@wrap_tool_call(
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
)
def my_hook(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command[Any] | Awaitable[ToolMessage | Command[Any]]],
) -> ToolMessage | Command[Any]:
    """Intercept tool execution."""
```

**Signature**: Function receives `request` and `handler` callback. Handler executes the tool and returns `ToolMessage | Command[Any]`.

**Returns**: `ToolMessage` or LangGraph `Command`.

**Example**:

```python
@wrap_tool_call
def retry_on_error(request, handler):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = handler(request)
            if is_valid(result):
                return result
        except Exception:
            if attempt == max_retries - 1:
                raise
    return result
```

## Synchronous vs. Asynchronous Execution

Middleware can define either sync or async implementations of hooks. The agent automatically detects and invokes the appropriate version based on the execution context.

```python
# Sync version: used in sync contexts (stream(), invoke())
def wrap_model_call(self, request, handler) -> ModelCallResult:
    return handler(request)

# Async version: used in async contexts (astream(), ainvoke())
async def awrap_model_call(self, request, handler) -> ModelCallResult:
    return await handler(request)
```

**Important**: If only the sync version is defined and the agent runs asynchronously, calling that hook raises `NotImplementedError` with guidance to implement the async version or invoke the agent synchronously.

Decorators automatically detect function async status and generate the appropriate hook method.

## Control Flow

### JumpTo

Control flow signal returned in state updates from `before_model` or `after_model` hooks.

```python
JumpTo = Literal["tools", "model", "end"]
```

- `"tools"`: Jump to the tool execution node
- `"model"`: Jump back to the model invocation node
- `"end"`: Jump to agent completion

Must declare valid destinations via `can_jump_to` in hook decorator or `@hook_config`.

### @hook_config

Decorator to declare which jump destinations are valid for a `before_model` or `after_model` method.

```python
@hook_config(can_jump_to: list[JumpTo] | None = None)
def my_method(self, state: StateT, runtime: Runtime) -> dict[str, Any] | None:
    """Declare valid jump destinations."""
```

**Example**:

```python
class MyMiddleware(AgentMiddleware):
    @hook_config(can_jump_to=["end", "model"])
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if some_condition(state):
            return {"jump_to": "end"}
        return None
```

## Type Variables

The module defines several type variables for generic constraints:

```python
StateT = TypeVar("StateT", bound=AgentState[Any], default=AgentState[Any])
"""State type variable, bounded by AgentState, defaults to AgentState[Any]."""

StateT_co = TypeVar("StateT_co", bound=AgentState[Any], default=AgentState[Any], covariant=True)
"""Covariant state type variable for return positions."""

ResponseT = TypeVar("ResponseT", default=Any)
"""Response type variable for structured output, defaults to Any."""
```

## LangGraph Integration

Middleware hooks integrate with LangGraph types:

```python
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph.typing import ContextT
```

**Runtime context**: Passed to all hooks; provides access to app context via `runtime.context: ContextT`, and custom event streaming via `runtime.stream_writer()`.

**Command**: LangGraph control flow object returned from `wrap_model_call` (via `ExtendedModelResponse`) or `wrap_tool_call` for state updates or navigation. Currently supports state updates and jump operations; `goto`, `resume`, and `graph` are not yet supported for middleware commands.

**ToolCallWrapper and ToolCallRequest**: From LangGraph prebuilt tool node, used in tool call interception.
