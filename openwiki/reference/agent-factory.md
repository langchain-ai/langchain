---
type: Reference
title: Agent Factory - Function Reference
description: Complete reference documentation for create_agent function signature, parameters, return types, middleware composition rules, state schemas, and factory construction internals.
tags: [agent-factory, create_agent, middleware, state-machine, reference, api]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-07T08:15:07.286Z
sources:
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
generated: { by: "openwiki/0.5.0", at: "2026-11-07T08:15:07.286Z" }
---

## Function Signature

### create_agent

The `create_agent` function is the primary factory for constructing LangChain agents. It returns a compiled LangGraph `StateGraph` that orchestrates conversation flow between a language model, tool execution, and pluggable middleware layers.

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

@overload
def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | dict[str, Any] | None = None,
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

The function is overloaded to provide type inference based on the `response_format` argument. See [Type Overloads](#type-overloads) below for details.

## Parameters

### model: str | BaseChatModel

The language model to use for the agent.

**Type:**
- String identifier (e.g., `"openai:gpt-4o"`, `"anthropic:claude-sonnet-4-5"`, `"openai:gpt-4.1"`)
- `BaseChatModel` instance (direct model object)

**Behavior:**
- String models are resolved via `init_chat_model(model)` to instantiate the corresponding chat model.
- Direct model instances bypass resolution and are used as-is.
- Model profile data (if available) is used for auto-detection of structured output support.
- Fallback model name patterns are checked if profile data is unavailable (see `FALLBACK_MODELS_WITH_STRUCTURED_OUTPUT` in factory).

**Examples:**
```python
# String model
agent = create_agent(model="openai:gpt-4o", tools=[...])

# Direct instance
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_agent(model=model, tools=[...])
```

### tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None

Available tools for the agent to call.

**Type:**
- `None` or empty list: Agent has no tools; only model node is created.
- `BaseTool` instances: Fully-typed tools with metadata and execution logic.
- Callables (raw functions): Automatically wrapped into `BaseTool` by `ToolNode`.
- `dict` objects: Provider-native tools (e.g., OpenAI code interpreter). Not executed client-side; provider handles them.

**Behavior:**
- All `BaseTool` instances and callables are converted by `ToolNode` and registered in `tool_node.tools_by_name`.
- Dict tools remain as-is and are passed to model binding but not to `ToolNode`.
- If tools are provided, a `tools` graph node is added for execution.
- If tools is `None` or empty AND no middleware defines `wrap_tool_call`, no tools node is created.
- Middleware can register additional tools via the `tools` attribute (added to `ToolNode`).

**Examples:**
```python
from langchain_core.tools import tool

# BaseTool instances
@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for {query}"

agent = create_agent(model=model, tools=[search])

# Callables (auto-wrapped)
def calculate(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

agent = create_agent(model=model, tools=[calculate])

# Provider tools (e.g., OpenAI code interpreter)
agent = create_agent(
    model=model,
    tools=[{"type": "code_interpreter"}],  # OpenAI-native tool
)

# Mixed
agent = create_agent(
    model=model,
    tools=[search, calculate, {"type": "code_interpreter"}],
)
```

### system_prompt: str | SystemMessage | None

Optional system message prepended to every model call.

**Type:**
- `str`: Converted to `SystemMessage(content=system_prompt)`.
- `SystemMessage`: Used directly.
- `None`: No system message is added.

**Behavior:**
- System message is stored separately from the messages list in `ModelRequest.system_message`.
- During model execution, the system message is prepended to the messages list before invoking the model.
- Middleware can override via `request.override(system_message=...)`.

**Example:**
```python
agent = create_agent(
    model=model,
    tools=[search],
    system_prompt="You are a helpful assistant that searches the web.",
)

# Or with SystemMessage
from langchain_core.messages import SystemMessage

agent = create_agent(
    model=model,
    tools=[search],
    system_prompt=SystemMessage(content="You are a helpful assistant."),
)
```

### middleware: Sequence[AgentMiddleware[StateT_co, ContextT]]

Ordered sequence of middleware instances to compose into the agent.

**Type:** Sequence of `AgentMiddleware` subclass instances

**Behavior:**
- Middleware are composed in list order: **first item becomes the outermost layer**.
<!-- openwiki: broken internal link [#middleware-hooks] heading anchor "middleware-hooks" does not exist in /openwiki/reference/agent-factory.md. Fix the href or restore the target, then delete this comment. -->
- Each middleware can implement sync and/or async hooks (see [Middleware Hooks](#middleware-hooks)).
- Duplicate middleware instances are detected and raise `AssertionError`.
- Middleware state schemas are merged with the base `AgentState` (middleware first, base last).
- Middleware with `wrap_model_call` or `wrap_tool_call` are composed into middleware stacks (inner-first ordering for handler calls).

**Composition Example:**
```python
middleware = [MiddlewareA, MiddlewareB, MiddlewareC]
# Execution order for node hooks:
# before_model: A.before_model() → B.before_model() → C.before_model()
# Execution order for wrap_model_call:
# A.wrap_model_call(request, B.wrap_model_call(request, C.wrap_model_call(request, execute)))
```

**Example:**
```python
from langchain.agents.middleware import ModelRetryMiddleware, ToolErrorMiddleware

agent = create_agent(
    model=model,
    tools=[search],
    middleware=[
        ModelRetryMiddleware(max_retries=3),
        ToolErrorMiddleware(),
    ],
)
```

### response_format: ResponseFormat[ResponseT] | type[ResponseT] | dict[str, Any] | None

Structured output specification. Enables the agent to return typed data.

**Type:**
- `None`: No structured output (default); `ResponseT` is `Any`.
- `ResponseFormat` subclass instance: `ProviderStrategy`, `ToolStrategy`, or `AutoStrategy`.
- `type[ResponseT]`: Pydantic model class (e.g., `WeatherReport`).
- `dict[str, Any]`: Raw JSON schema dict (wrapped in `AutoStrategy`).

**Behavior:**
- `None`: No structured output is configured; `structured_response` field is omitted from output schema.
- Raw schema (dict or Pydantic class): Wrapped in `AutoStrategy` for auto-detection.
- `AutoStrategy`: Converts to `ProviderStrategy` if model supports it, otherwise `ToolStrategy`.
- `ToolStrategy`: Creates synthetic tools for each schema spec; model forced to call tool to complete turn.
- `ProviderStrategy`: Uses model's native structured output (e.g., OpenAI's `response_format` param).
- Structured output tools are added to the model's tool binding and are validated after model output.
- If a structured output tool is called, execution does not loop back to tools; it exits instead.

**Type Inference:**
- When `response_format` is provided, `ResponseT` is inferred for better type checking.
- Return type includes `structured_response: ResponseT | None` in output schema.

**Examples:**
```python
from pydantic import BaseModel

class WeatherReport(BaseModel):
    location: str
    temperature: int
    conditions: str

# Raw Pydantic class (auto-detected strategy)
agent = create_agent(
    model=model,
    response_format=WeatherReport,
)

# Explicit strategy
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model=model,
    response_format=ProviderStrategy(schema=WeatherReport),
)

# Raw schema dict
agent = create_agent(
    model=model,
    response_format={
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "temperature": {"type": "integer"},
        },
        "required": ["location", "temperature"],
    },
)
```

### state_schema: type[AgentState[ResponseT]] | None

Custom state base class extending `AgentState`. Allows adding custom state fields without middleware.

**Type:** `TypedDict` subclass extending `AgentState[ResponseT]`

**Behavior:**
- When provided, used as the base state schema instead of `AgentState`.
- Merged with middleware state schemas during graph construction.
- User's `state_schema` wins on field conflicts (merged last).
- Custom fields can be marked with `OmitFromSchema` to exclude from input/output schemas.

**Example:**
```python
from typing import Annotated, TypedDict, Required
from langchain.agents.middleware.types import AgentState

class CustomState(AgentState):
    user_id: Required[str]
    conversation_context: Required[dict[str, Any]]

agent = create_agent(
    model=model,
    tools=[search],
    state_schema=CustomState,
)
```

### context_schema: type[ContextT] | None

Optional schema for runtime context accessible to middleware.

**Type:** TypedDict or dataclass type

**Behavior:**
- Passed to `StateGraph(context_schema=...)`.
- Accessible via `runtime.context` in middleware hooks.
- Enables type-safe middleware access to execution-time context (e.g., user ID, request metadata).

**Example:**
```python
from typing import TypedDict

class RequestContext(TypedDict):
    user_id: str
    session_id: str

agent = create_agent(
    model=model,
    tools=[search],
    context_schema=RequestContext,
)

# Middleware can access via runtime.context
```

### checkpointer: Checkpointer | None

Optional checkpoint saver for persisting agent state across turns.

**Type:** `BaseCheckpointer` instance

**Behavior:**
- Used for thread-level persistence (e.g., saving chat history per conversation thread).
- Enables checkpointing and resuming agent execution.
- Passed to `graph.compile(checkpointer=...)`.

**Example:**
```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver(":memory:")

agent = create_agent(
    model=model,
    tools=[search],
    checkpointer=checkpointer,
)
```

### store: BaseStore | None

Optional store for cross-thread persistence.

**Type:** `BaseStore` instance

**Behavior:**
- Used for cross-thread data (e.g., user profiles, document vectors, shared knowledge).
- Accessible from middleware and tools via `runtime`.
- Passed to `graph.compile(store=...)`.

### interrupt_before: list[str] | None

List of node names to interrupt execution before.

**Type:** List of node names (e.g., `["model", "tools"]`)

**Behavior:**
- Execution pauses before entering these nodes, allowing external intervention (e.g., user approval).
- Useful for human-in-the-loop scenarios.

**Example:**
```python
agent = create_agent(
    model=model,
    tools=[delete_file],
    interrupt_before=["tools"],  # Pause before tool execution
)
```

### interrupt_after: list[str] | None

List of node names to interrupt execution after.

**Type:** List of node names

**Behavior:**
- Execution pauses after exiting these nodes.
- Useful for collecting feedback on outputs.

### debug: bool

Enable verbose logging for graph execution.

**Default:** `False`

**Behavior:**
- When `True`, prints detailed information about node execution, state updates, and transitions.
- Useful for debugging middleware behavior and understanding agent flow.

**Example:**
```python
agent = create_agent(
    model=model,
    tools=[search],
    debug=True,
)
```

### name: str | None

Optional name for the compiled graph.

**Behavior:**
- Used in LangSmith tracing metadata (`lc_agent_name`).
- Automatically used when adding the graph as a subgraph node in multi-agent systems.
- Included in graph compilation config.

**Example:**
```python
agent = create_agent(
    model=model,
    tools=[search],
    name="WebSearchAgent",
)
```

### cache: BaseCache[Any] | None

Optional cache instance for caching graph execution.

**Type:** `BaseCache` from LangGraph

**Behavior:**
- Passed to `graph.compile(cache=...)`.
- Enables memoization of node outputs to reduce redundant computation.

### transformers: Sequence[TransformerFactory] | None

Optional sequence of scope-aware stream transformer factories.

**Type:** Sequence of callables with signature `(scope: str) -> StreamTransformer`

**Behavior:**
- Each factory is invoked per scope to receive a fresh instance.
- Factories are merged with middleware-declared transformers and registered defaults (`ToolCallTransformer`, `SubagentTransformer`).
- Final order: `ToolCallTransformer` → middleware transformers → user transformers.
- Used for custom event filtering, stream adaptation, or scope-aware processing.

**Example:**
```python
def custom_transformer_factory(scope):
    return MyCustomTransformer(scope=scope)

agent = create_agent(
    model=model,
    tools=[search],
    transformers=[custom_transformer_factory],
)
```

## Return Type

### CompiledStateGraph[AgentState[ResponseT], ContextT, InputAgentState, OutputAgentState[ResponseT]]

A compiled LangGraph state graph ready for execution.

**Generic Parameters:**
- `AgentState[ResponseT]`: Internal state schema type-parametrized by structured response type.
- `ContextT`: Runtime context type.
- `InputAgentState`: Public input schema (accepts message dicts and `AnyMessage` objects).
- `OutputAgentState[ResponseT]`: Public output schema (returns `AnyMessage` objects and optional structured response).

**Key Methods:**
- `invoke(input: InputAgentState, config: RunnableConfig | None = None) -> OutputAgentState`: Synchronous execution.
- `ainvoke(input: InputAgentState, config: RunnableConfig | None = None) -> OutputAgentState`: Asynchronous execution.
- `stream(input: InputAgentState, config: RunnableConfig | None = None, stream_mode: str = "updates") -> Iterator[...]`: Synchronous streaming.
- `astream(input: InputAgentState, config: RunnableConfig | None = None, stream_mode: str = "updates") -> AsyncIterator[...]`: Asynchronous streaming.

**Example:**
```python
agent = create_agent(model=model, tools=[search])

# Synchronous invocation
output = agent.invoke(
    {"messages": [{"role": "user", "content": "Search for Python"}]},
    config={"configurable": {"thread_id": "user123"}},
)

# Asynchronous invocation
output = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Search for Python"}]},
)

# Streaming
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Search for Python"}]},
    stream_mode="updates",
):
    print(chunk)
```

## Type Overloads

The `create_agent` function uses three overloads for type inference:

### Overload 1: No response_format

When `response_format=None` (default):
```python
def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: None = None,
    state_schema: None = None,
    # ... other parameters ...
) -> CompiledStateGraph[AgentState[Any], ContextT, InputAgentState, OutputAgentState[Any]]:
```

**Inference:**
- `ResponseT = Any`
- Output schema does not include `structured_response` field.

### Overload 2: Raw dict response_format

When `response_format` is a plain `dict[str, Any]`:
```python
def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: dict[str, Any],
    state_schema: type[AgentState[dict[str, Any]]] | None = None,
    # ... other parameters ...
) -> CompiledStateGraph[
    AgentState[dict[str, Any]], ContextT, InputAgentState, OutputAgentState[dict[str, Any]]
]:
```

**Inference:**
- `ResponseT = dict[str, Any]`
- Structured response is an untyped dict.

### Overload 3: Schema-typed response_format

When `response_format` is a Pydantic model or `ResponseFormat[ResponseT]`:
```python
def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | None = None,
    state_schema: type[AgentState[ResponseT]] | None = None,
    # ... other parameters ...
) -> CompiledStateGraph[
    AgentState[ResponseT], ContextT, InputAgentState, OutputAgentState[ResponseT]
]:
```

**Inference:**
- `ResponseT` is inferred from the Pydantic model or `ResponseFormat[ResponseT]`.
- Structured response is properly typed in output schema.

## State Schemas

### AgentState[ResponseT]

The internal state used throughout the graph.

```python
class AgentState(TypedDict, Generic[ResponseT]):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]
```

**Fields:**
- `messages` (Required): Conversation history. Uses `add_messages` reducer (merges/replaces by ID).
- `jump_to` (NotRequired, Ephemeral, Private): Control flow override for middleware (e.g., `"end"` to skip to exit).
- `structured_response` (NotRequired): Parsed structured output (if `response_format` was set). Omitted from input schema.

### InputAgentState

The input schema for `invoke`/`stream` calls.

```python
class InputAgentState(TypedDict):
    messages: Required[Annotated[list[AnyMessage | dict[str, Any]], add_messages]]
```

**Behavior:**
- Accepts plain dicts with `{"role": "user", "content": "..."}` format.
- Accepts `AnyMessage` objects.
- Merges with existing messages via `add_messages` reducer.

### OutputAgentState[ResponseT]

The output schema returned by `invoke`/`stream`.

```python
class OutputAgentState(TypedDict, Generic[ResponseT]):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    structured_response: NotRequired[ResponseT]
```

**Behavior:**
- Returns fully-typed `AnyMessage` objects (not dicts).
- Includes `structured_response` only if `response_format` was provided.

## Middleware Hooks and Composition

### Sync and Async Hook Pairs

Each middleware hook has a corresponding async variant:

| Sync | Async | Description |
|------|-------|-------------|
| `before_agent` | `abefore_agent` | Runs once at start before any model calls |
| `before_model` | `abefore_model` | Runs before each model call |
| `wrap_model_call` | `awrap_model_call` | Wraps model invocation (retry, caching, etc.) |
| `after_model` | `aafter_model` | Runs after each model call |
| `after_agent` | `aafter_agent` | Runs once at the end after all iterations |
| `wrap_tool_call` | `awrap_tool_call` | Wraps tool execution (validation, retry, etc.) |

**Sync/Async Rules:**
- If only async is defined and sync is invoked, `NotImplementedError` is raised.
- If only sync is defined and async is invoked, `NotImplementedError` is raised.
- Middleware should implement both for full compatibility.
- The factory selects the appropriate variant at graph invocation time.

### Node Hooks Ordering

Node hooks (`before_agent`, `before_model`, `after_model`, `after_agent`) are chained sequentially:

```
middleware = [A, B, C]

before_agent hook execution: A.before_agent() → B.before_agent() → C.before_agent()
before_model hook execution: A.before_model() → B.before_model() → C.before_model()
after_model hook execution: C.after_model() → B.after_model() → A.after_model()
after_agent hook execution: C.after_agent() → B.after_agent() → A.after_agent()
```

**Note:** `after_*` hooks run in reverse order (innermost first).

### wrap_model_call Composition

`wrap_model_call` handlers are composed with the first middleware as the outermost layer:

```
middleware = [A, B, C]

A.wrap_model_call(request, B.wrap_model_call(request, C.wrap_model_call(request, execute)))
                    ↑                                    ↑
                  outermost                         innermost
```

**Behavior:**
- Each layer can call `handler(request)` to invoke the next layer.
- Commands are accumulated inner-first (C's commands before B's before A's).
- If a layer calls handler multiple times (retry), inner layers see fresh requests each time.

### wrap_tool_call Composition

Similar to `wrap_model_call`:

```
middleware = [A, B, C]

A.wrap_tool_call(request, B.wrap_tool_call(request, C.wrap_tool_call(request, execute)))
```

**Behavior:**
- Each layer wraps tool execution.
- Can modify the tool call, retry, or short-circuit.
- Results are `ToolMessage` or `Command` objects.

### Request/Response Immutability

Middleware should use immutable patterns for requests:

```python
def wrap_model_call(self, request, handler):
    # ❌ Don't (deprecated):
    request.tools = new_tools  # Raises deprecation warning

    # ✅ Do (immutable override):
    new_request = request.override(tools=new_tools)
    return handler(new_request)
```

**Available overrides for ModelRequest.override():**
- `model`: `BaseChatModel`
- `system_message`: `SystemMessage | None`
- `messages`: `list[AnyMessage]`
- `tools`: `list[BaseTool | dict[str, Any]]`
- `response_format`: `ResponseFormat[Any] | None`
- `tool_choice`: Tool choice configuration
- `model_settings`: `dict[str, Any]`
- `state`: `AgentState[Any]`

### Control Flow with jump_to

Middleware node hooks can override default routing via the `jump_to` state field:

```python
def before_model(self, state, runtime):
    if should_skip_model_call():
        return {"jump_to": "end"}  # Skip to exit
    return None
```

**Valid destinations:** `"model"`, `"tools"`, `"end"`

**Requirements:**
- Hook method must declare `@before_model(can_jump_to=["end"])` or similar to enable conditional routing.
- `jump_to` is ephemeral—cleared after routing decision.
- Only allowed in node hooks (`before_*`, `after_*`), not in `wrap_*` hooks.

## Model Binding and Structured Output

### Strategy Auto-Detection

When `response_format` is provided as a raw schema (Pydantic class or dict), the factory auto-detects the best strategy:

```python
response_format = WeatherReport  # Pydantic class

if _supports_provider_strategy(model, tools=...):
    # Model supports provider-native structured output
    effective_response_format = ProviderStrategy(schema=WeatherReport)
else:
    # Fall back to tool-based strategy
    effective_response_format = ToolStrategy(schema=WeatherReport)
```

**Detection Logic:**
1. Check model's `profile` data (if available).
2. Check fallback model name patterns (regex matching against `FALLBACK_MODELS_WITH_STRUCTURED_OUTPUT`).
3. Account for model limitations (e.g., Gemini < 3-series doesn't support structured output with tools).

### ProviderStrategy Binding

When provider strategy is selected:

```python
# Model is bound with provider-specific kwargs
kwargs = effective_response_format.to_model_kwargs()
bind_kwargs = {**kwargs, **request.model_settings}

# For OpenAI-compatible models, redundantly set strict=True
if _is_openai_compatible_model(request.model):
    bind_kwargs["strict"] = True

bound_model = request.model.bind_tools(final_tools, **bind_kwargs)
```

**Behavior:**
- Model is asked to return JSON matching the schema directly.
- No synthetic tool is created; structured output is extracted from model response.
- Provider handles validation; parsing errors are caught and can trigger retry (if configured).

### ToolStrategy Binding

When tool strategy is selected:

```python
# Create synthetic tools from schema specs
for response_schema in tool_strategy_for_setup.schema_specs:
    structured_tool_info = OutputToolBinding.from_schema_spec(response_schema)
    structured_output_tools[structured_tool_info.tool.name] = structured_tool_info

# Force tool use
tool_choice = "any" if structured_output_tools else request.tool_choice
bound_model = request.model.bind_tools(final_tools, tool_choice=tool_choice, **request.model_settings)
```

**Behavior:**
- Synthetic tools are added to the model's tool binding.
- Model is forced to call the tool (via `tool_choice="any"`).
- Tool arguments are parsed and extracted as structured response.
- Calling a structured output tool triggers exit (no further tool execution).

### Structured Output Validation

After model output, structured tools are validated:

```python
structured_tool_calls = [tc for tc in output.tool_calls if tc["name"] in structured_output_tools]

if len(structured_tool_calls) > 1:
    # Multiple structured outputs (union type)
    exception = MultipleStructuredOutputsError(tool_names, output)
    should_retry, error_message = _handle_structured_output_error(exception, effective_response_format)
    if not should_retry:
        raise exception

elif structured_tool_calls:
    # Single structured output
    tool_call = structured_tool_calls[0]
    structured_tool_binding = structured_output_tools[tool_call["name"]]
    try:
        structured_response = structured_tool_binding.parse(tool_call["args"])
    except Exception as exc:
        # Handle parsing error (retry or raise)
        should_retry, error_message = _handle_structured_output_error(exc, effective_response_format)
        if not should_retry:
            raise StructuredOutputValidationError(tool_call["name"], exc, output)
```

**Errors Handled:**
- `MultipleStructuredOutputsError`: Multiple structured output tools were called (unions).
- `StructuredOutputValidationError`: Parsing or validation failed.

**Error Recovery:**
- If `response_format.handle_errors` is configured, retry logic applies.
- Default template: `"Error: {error}\n Please fix your mistakes."`

## Tool Execution

### ToolNode Creation

A `ToolNode` is created if:
1. Tools are provided (either regular tools or middleware tools), OR
2. Middleware defines `wrap_tool_call` or `awrap_tool_call` (for dynamic tools)

```python
tool_node = (
    ToolNode(
        tools=available_tools,
        wrap_tool_call=wrap_tool_call_wrapper,
        awrap_tool_call=awrap_tool_call_wrapper,
    )
    if available_tools or wrap_tool_call_wrapper or awrap_tool_call_wrapper
    else None
)
```

### Tool Execution Flow

1. Model returns `AIMessage` with `tool_calls`.
2. Conditional routing checks if any tool calls are pending.
3. `ToolNode` batches calls and executes them (parallel by default, can be sequential).
4. Execution results are wrapped in `ToolMessage`s.
5. Loop back to model unless:
   - A structured output tool was called
   - All executed tools have `return_direct=True`
   - No pending tool calls remain

### Dynamic Tool Validation

Middleware can add tools dynamically via `request.override(tools=[...])` in `wrap_model_call`:

```python
def wrap_model_call(self, request, handler):
    new_tools = request.tools + [dynamic_tool]
    new_request = request.override(tools=new_tools)
    return handler(new_request)
```

**Validation Rules:**
- If `wrap_tool_call` is NOT defined: Unknown tool names raise `ValueError` with guidance.
- If `wrap_tool_call` IS defined: Dynamic tool execution is delegated to middleware; no validation error.

**Error Message Template:**
```
DYNAMIC_TOOL_ERROR_TEMPLATE = """
Middleware added tools that the agent doesn't know how to execute.

Unknown tools: {unknown_tool_names}
Registered tools: {available_tool_names}

This happens when middleware modifies `request.tools` in `wrap_model_call` to include
tools that weren't passed to `create_agent()`.

How to fix this:

Option 1: Register tools at agent creation (recommended for most cases)
    Pass the tools to `create_agent(tools=[...])` or set them on `middleware.tools`.
    This makes tools available for every agent invocation.

Option 2: Handle dynamic tools in middleware (for tools created at runtime)
    Implement `wrap_tool_call` to execute tools that are added dynamically:

    class MyMiddleware(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            if request.tool_call["name"] == "dynamic_tool":
                # Execute the dynamic tool yourself or override with tool instance
                return handler(request.override(tool=my_dynamic_tool))
            return handler(request)
```

## Graph Construction

### Node Creation

The factory creates nodes in the following order:

1. **model** node: Model invocation (sync/async via `RunnableCallable`).
2. **tools** node (conditional): Tool execution (only if tools are available).
3. **Middleware nodes** (conditional):
   - `{middleware.name}.before_agent` (one per middleware with `before_agent`)
   - `{middleware.name}.before_model` (one per middleware with `before_model`)
   - `{middleware.name}.after_model` (one per middleware with `after_model`)
   - `{middleware.name}.after_agent` (one per middleware with `after_agent`)

### Edge Construction

Edges are built to support the control flow:

1. **START → Entry Node**: First middleware's `before_agent` or first middleware's `before_model` or `model`.
2. **Entry → Loop Entry**: After entry hooks, go to loop start (before_model chain or model).
3. **Loop Entry → Model**: Always go to model node.
4. **Model → After Model Middleware**: If after_model hooks exist, chain them; otherwise skip.
5. **After Model → Tools / Exit**: Conditional edge based on tool calls and exit conditions.
6. **Tools → Loop Entry / Exit**: Conditional edge after tool execution.
7. **Exit → END**: Final exit after after_agent hooks.

**Special Nodes:**
- **entry_node**: Runs `before_agent` hooks once at start.
- **loop_entry_node**: Start of the model-tool iteration loop. Points to `before_model` or `model`.
- **loop_exit_node**: Exit point after model execution. Points to `after_model` or conditional edge.
- **exit_node**: Final cleanup. Runs `after_agent` hooks once before END.

### Conditional Edges

The factory uses conditional edges to decide routing:

```python
# After model: decide if tools should be called
graph.add_conditional_edges(
    loop_exit_node,
    _make_model_to_tools_edge(...),
    ["tools", exit_node, loop_entry_node],  # Possible destinations
)

# After tools: decide if model should be called again
graph.add_conditional_edges(
    "tools",
    _make_tools_to_model_edge(...),
    ["model", exit_node],  # Possible destinations
)
```

**Routing Logic:**
- `_make_model_to_tools_edge`: Routes based on tool calls, structured output status, and exit conditions.
- `_make_tools_to_model_edge`: Routes based on `return_direct` and structured output tool execution.

## Handler Composition

### Model Call Handler Chain

Multiple `wrap_model_call` middleware are composed into a single middleware stack:

```python
# Sync handlers
sync_handlers = [
    traceable(name=f"{m.name}.wrap_model_call", **_wrap_trace_kwargs(m))(m.wrap_model_call)
    for m in middleware_w_wrap_model_call
]
wrap_model_call_handler = _chain_model_call_handlers(sync_handlers)

# Async handlers
async_handlers = [
    traceable(name=f"{m.name}.awrap_model_call", **_wrap_trace_kwargs(m))(m.awrap_model_call)
    for m in middleware_w_awrap_model_call
]
awrap_model_call_handler = _chain_async_model_call_handlers(async_handlers)
```

**Composition Rules:**
- Each handler is wrapped with `traceable` for LangSmith tracing.
- Commands from each layer are accumulated (inner-first, then outer).
- Inner layers always see `ModelResponse` from the handler (not `ExtendedModelResponse`).
- Outer layers can receive `ExtendedModelResponse` with accumulated commands.

### Tool Call Handler Chain

Similarly for `wrap_tool_call`:

```python
wrappers = [
    traceable(name=f"{m.name}.wrap_tool_call", **_wrap_trace_kwargs(m))(m.wrap_tool_call)
    for m in middleware_w_wrap_tool_call
]
wrap_tool_call_wrapper = _chain_tool_call_wrappers(wrappers)
```

## Tracing and Debugging

### Trace Policy

Middleware can define a `trace_policy` to control what is traced:

```python
from langgraph.types import TracePolicy, omit_payload

class MyMiddleware(AgentMiddleware):
    trace_policy = TracePolicy(process_inputs=omit_payload)
```

**Features:**
- `process_inputs`: Function to transform traced inputs (e.g., redact sensitive data).
- `process_outputs`: Function to transform traced outputs.
- Messages are still captured on the inner model-call span even if outer hooks are redacted.

### Debug Mode

Enable verbose logging with `debug=True`:

```python
agent = create_agent(
    model=model,
    tools=[search],
    debug=True,  # Prints node execution, state updates, transitions
)
```

## Error Handling

### Invalid Tool Calls

If a model returns malformed tool calls, they are patched before graph execution:

```python
def _patch_invalid_tool_calls(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """Add error ToolMessage for unmatched invalid tool calls."""
    # For each invalid tool call without a matching ToolMessage response,
    # create an error ToolMessage to inform the model.
```

### Structured Output Error Handling

Parsing errors are handled based on `response_format.handle_errors`:

```python
# Config options:
# - False: Raise exception immediately (no retry)
# - True: Retry with default error template
# - str: Retry with custom error message
# - Exception type: Retry only if exception matches type
# - tuple of types: Retry if exception matches any type
# - callable: Custom error handler function
```

## Graph Compilation

The factory compiles the graph with:

```python
return graph.compile(
    checkpointer=checkpointer,
    store=store,
    interrupt_before=interrupt_before,
    interrupt_after=interrupt_after,
    debug=debug,
    name=name,
    cache=cache,
    transformers=_dedupe_transformers([
        ToolCallTransformer,
        SubagentTransformer,
        *middleware_transformers,
        *(transformers or ()),
    ]),
).with_config({
    "recursion_limit": 9_999,
    "metadata": {
        "ls_integration": "langchain_create_agent",
        "lc_agent_name": name,  # If provided
    }
})
```

**Recursion Limit:** Set to 9,999 to allow deep agentic loops.

**Transformers:** De-duplicated by identity to prevent registering the same transformer multiple times.

**Metadata:** Used for LangSmith integration and tracing.
