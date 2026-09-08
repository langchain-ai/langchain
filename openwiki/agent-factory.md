---
type: "Reference"
title: "Create a basic agent"
openwiki_generated: true
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:18:34.589Z
sources:
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
  - id: openwiki-source-07e634f5cd5f00c636010306
    resource: repo://libs/langchain_v1/langchain/agents/middleware/__init__.py
  - id: openwiki-source-4ed5b553d7dea01d659161d1
    resource: repo://libs/langchain_v1/langchain/agents/middleware/_trace_policy.py
  - id: openwiki-source-03e8ca0eebe37feda8566793
    resource: repo://libs/langchain_v1/langchain/agents/middleware/types.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:18:34.589Z" }
---


## Overview

The **Agent Factory** is the foundational entry point for building LangChain agents. The `create_agent` function constructs a compiled LangGraph state machine that orchestrates conversation flow between a language model, tool execution, and pluggable middleware layers. It handles tool binding, structured output, state schema resolution, and middleware composition automatically, allowing developers to focus on business logic while the factory manages the complex graph construction and execution model.

## Quick Start

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

def check_weather(location: str) -> str:
    """Return the weather forecast for the specified location."""
    return f"It's sunny in {location}"

# Create a basic agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=[check_weather],
    system_prompt="You are a helpful weather assistant."
)

# Stream responses
inputs = {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
for chunk in agent.stream(inputs, stream_mode="updates"):
    print(chunk)
```

## Agent Architecture

The agent factory constructs a **state machine graph** with the following structure:

<!-- openwiki: mermaid parse failed and this diagram was converted to a text fence so it does not break rendering. Fix the diagram source and restore the mermaid fence. Parser error: Heuristic: an unescaped angle bracket inside a label breaks rendering; rephrase the label. -->
```text
graph TD
    START["START"] --> ENTRY["Entry Node<br/>(before_agent)"]
    ENTRY --> LOOP_ENTRY["Loop Entry<br/>(before_model | model)"]
    LOOP_ENTRY --> MODEL["Model Node<br/>(LLM Call)"]
    MODEL --> AFTER_MODEL["After Model<br/>(middleware)"]
    AFTER_MODEL --> ROUTER{Has Tool Calls?}
    ROUTER -->|Yes| TOOLS["Tools Node<br/>(Execute Tools)"]
    ROUTER -->|No| EXIT["Exit Node<br/>(after_agent)"]
    TOOLS --> TOOLS_ROUTER{Tool Direct Return?}
    TOOLS_ROUTER -->|No| LOOP_ENTRY
    TOOLS_ROUTER -->|Yes| EXIT
    EXIT --> END["END"]
```

**Key Nodes:**

- **Entry Node**: Runs `before_agent` hooks once at the start of the conversation.
- **Loop Entry**: Begins each iteration of the model → tool loop. Runs `before_model` middleware.
- **Model Node**: Calls the language model with messages and system prompt. Handles structured output parsing.
- **After Model**: Runs `after_model` hooks after model output (runs each loop iteration).
- **Tools Node**: Executes tools returned by the model. Skipped if no tools are defined.
- **Exit Node**: Runs `after_agent` hooks once at the end of the conversation.

## Core Concepts

### AgentState

The agent maintains a typed state dictionary that flows through the graph:

```python
class AgentState(TypedDict):
    messages: list[AnyMessage]              # Conversation history
    jump_to: JumpTo | None                  # Optional control flow override
    structured_response: ResponseT | None   # Parsed structured output (if enabled)
```

**Reducers and Aggregation:**
- `messages` uses `add_messages` reducer: new messages are merged with existing ones, with duplicates by `id` being replaced.
- `jump_to` is ephemeral: set by middleware to override default routing (e.g., "model", "tools", "end").
- `structured_response` is cleared each iteration unless explicitly set, preventing stale values after checkpointing.

### Input and Output Schemas

The factory derives input and output schemas from the base `AgentState` and any middleware-provided schemas:

```python
class InputAgentState(TypedDict):
    messages: list[AnyMessage | dict[str, Any]]  # User can pass plain dicts

class OutputAgentState(TypedDict):
    messages: list[AnyMessage]
    structured_response: ResponseT | None  # Only if response_format is set
```

Middleware can extend state by declaring a `state_schema` attribute (a `TypedDict`), which is merged during graph construction.

### ModelRequest and ModelResponse

All middleware hooks operate on structured request/response objects:

**ModelRequest** encapsulates everything needed for a model call:
- `model`: The `BaseChatModel` instance
- `messages`: Current conversation (excluding system message)
- `system_message`: Optional system prompt
- `tools`: Available tools to bind
- `response_format`: Structured output spec (if enabled)
- `state`: Current agent state
- `runtime`: LangGraph `Runtime` for accessing context
- `tool_choice`: Override tool selection behavior
- `model_settings`: Extra kwargs to pass to `model.bind()`

Middleware can call `request.override(**changes)` to create a new request immutably, enabling request transformation before the model is invoked.

**ModelResponse** carries the result:
- `result`: List of messages (usually one `AIMessage`, sometimes with `ToolMessage` for structured output)
- `structured_response`: Parsed structured output (if `response_format` was set and parsing succeeded)

### Model Binding and Structured Output

The factory handles three structured-output strategies:

1. **ProviderStrategy**: Uses the model's native structured output (e.g., OpenAI's `response_format` param). Auto-detected for models with profile data indicating support.
2. **ToolStrategy**: Uses a special tool call to capture structured output. Tools are registered upfront; the model is forced to call the structured output tool to complete the turn.
3. **AutoStrategy**: Raw Pydantic schema that the factory auto-detects — converts to `ProviderStrategy` if the model supports it, otherwise `ToolStrategy`.

When `response_format` is set, the factory:
- Creates `OutputToolBinding` instances wrapping the schema(s)
- Adds them as synthetic tools to the model binding
- After model output, parses tool calls matching the schema and extracts the structured response
- Prevents further tool execution if a structured output tool was called (exit condition)

## Entry Point: create_agent

The `create_agent(model, tools, ...)` function is the primary factory. Signature highlights:

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

**Arguments:**

- **model**: Model string (e.g., `"openai:gpt-4o"`) or `BaseChatModel` instance. String models are resolved via `init_chat_model`.
- **tools**: List of `BaseTool` instances, raw callables, or dict-based provider tools. `None` or empty creates a model-only agent.
- **system_prompt**: String or `SystemMessage` prepended to every model call.
- **middleware**: Ordered sequence of `AgentMiddleware` instances. Composing happens in list order (first = outermost).
- **response_format**: Structured output spec. Can be a Pydantic model, `ResponseFormat` subclass, or raw `dict` schema.
- **state_schema**: Custom state base class extending `AgentState`. Merged with middleware schemas; user's schema wins on conflicts.
- **checkpointer**: Thread-level persistence (e.g., chat memory across turns).
- **store**: Cross-thread persistence (e.g., user profiles, document stores).
- **interrupt_before/after**: Node names to suspend execution for user intervention.
- **debug**: Enable verbose logging.
- **name**: Graph name; used in LangSmith tracing and subgraph imports.
- **cache**: Execution cache (LangGraph feature).
- **transformers**: Additional stream transformer factories (e.g., for custom event filtering).

## Middleware Composition

Middleware extend agent behavior without modifying the core logic. They are composed into stacks at each intercept point.

### Middleware Hooks

Each `AgentMiddleware` can implement up to six hook methods:

```python
class MyMiddleware(AgentMiddleware):
    # Sync hooks (default)
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Runs once at start before any model calls."""
        return {"key": "value"}  # Optional state updates
    
    def before_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Runs before each model call."""
        return None
    
    def wrap_model_call(
        self, request: ModelRequest, handler: Callable
    ) -> ModelResponse | AIMessage | ExtendedModelResponse:
        """Wraps the model invocation itself (retry, fallback, caching, etc.)."""
        return handler(request)  # Call inner layer
    
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Runs after each model call."""
        return None
    
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Runs once at the end after all iterations."""
        return None
    
    def wrap_tool_call(
        self, request: ToolCallRequest, execute: Callable
    ) -> ToolMessage | Command:
        """Wraps tool execution (validation, retry, auth, etc.)."""
        return execute(request)  # Call inner layer
```

**Async Versions**: Prefix with `a` (e.g., `abefore_agent`, `awrap_model_call`). If only async is defined, sync invocation raises; if only sync is defined, async falls back.

### Middleware Composition Rules

- **Hook Chaining**: Each hook type (e.g., `before_model`) from all middleware is chained sequentially.
- **Order**: First middleware in the list becomes the outermost layer.
  - Example: `middleware=[A, B, C]` → `A.before_model → B.before_model → C.before_model`
  - For `wrap_*` hooks (outermost matters for retry/caching): `A.wrap_model_call(request, B.wrap_model_call(request, C.wrap_model_call(request, execute)))`
- **Sync/Async**: Sync and async paths are kept separate. Each hook can choose to implement sync, async, or both. The factory selects the appropriate variant at runtime.
- **Commands**: Middleware can return `Command` objects from `wrap_model_call` to update state (e.g., add synthetic tool messages). Commands accumulate inner-first and are applied after the model response.

### Request/Response Immutability

Middleware should use immutable patterns:

```python
def wrap_model_call(self, request, handler):
    # DON'T: request.tools = new_tools  (deprecated, will warn)
    # DO:
    new_request = request.override(tools=new_tools)
    return handler(new_request)
```

### Jump To Control Flow

Middleware node hooks can override routing via the `jump_to` state field:

```python
def before_model(self, state, runtime):
    if should_skip_model_call():
        return {"jump_to": "end"}  # Skip to end
    return None
```

Valid destinations: `"model"`, `"tools"`, `"end"`. The hook method must declare `@before_model(can_jump_to=["end"])` to enable conditional routing.

## Core Middleware

The factory ships with a comprehensive middleware library:

### Retry and Error Handling

- **ModelRetryMiddleware**: Automatically retry failed model calls with exponential backoff. Configure max retries, exception types, and backoff factor.
- **ToolRetryMiddleware**: Retry failed tool executions. Supports custom failure policies (propagate, suppress, replace with error message).
- **ToolErrorMiddleware**: Convert selected tool exceptions to error `ToolMessage`s returned to the model (graceful error handling).

### Model Variants and Fallback

- **ModelFallbackMiddleware**: Fallback to alternate models on failure. Useful for resilience (e.g., try GPT-4o, fall back to Claude).

### Tool-Related

- **ToolCallLimitMiddleware**: Enforce maximum tool calls per turn or per agent run.
- **LLMToolSelectorMiddleware**: Use an LLM to pre-filter available tools based on the user query (reduce token cost and model confusion).
- **LLMToolEmulator**: Emulate tool calls without executing them (e.g., for testing or policy enforcement).
- **ProviderToolSearchMiddleware**: Automatically search for and register provider-native tools.

### Execution and Sandbox

- **ShellToolMiddleware**: Execute shell commands safely. Supports multiple execution policies: `HostExecutionPolicy` (local shell), `DockerExecutionPolicy` (containerized), `CodexSandboxExecutionPolicy` (remote sandbox).

### Search and File Access

- **FilesystemFileSearchMiddleware**: Search for files on the filesystem and return matches to the model.

### Conversational Quality

- **HumanInTheLoopMiddleware**: Pause execution to collect human feedback or approval before critical actions.
- **SummarizationMiddleware**: Automatically summarize long conversation histories to manage context length.

### Data Protection

- **PIIMiddleware**: Detect and redact personally identifiable information (PII). Configurable redaction rules; supports email, phone, SSN, API keys, and custom patterns.

### Advanced

- **ModelCallLimitMiddleware**: Limit total model invocations to prevent runaway loops.
- **ContextEditingMiddleware**: Edit or clear tool usage records in state (for context management).
- **TodoListMiddleware**: Maintain a persistent todo list across the conversation (custom state extension).

## Tool Handling

### Tool Registration

Tools are registered at agent creation. Supported formats:

1. **BaseTool instances** (preferred): Full control over execution, caching, and metadata.
   ```python
   from langchain_core.tools import tool
   
   @tool
   def search(query: str) -> str:
       """Search the web."""
       return ...
   ```

2. **Raw callables**: Automatically wrapped into `BaseTool` instances.
   ```python
   def search(query: str) -> str:
       """Search the web."""
       return ...
   ```

3. **Dict-based provider tools**: Native tools from the model's provider (e.g., OpenAI's code interpreter). Not executed client-side; the provider handles them.

### Tool Execution Flow

1. Model returns `AIMessage` with `tool_calls` list.
2. Conditional routing checks for pending tool calls (not yet executed).
3. **ToolNode** batches pending calls and executes them in parallel (or sequentially, depending on config).
4. Execution results are wrapped in `ToolMessage`s and added to state.
5. Loop back to model unless:
   - A tool with `return_direct=True` was executed
   - A structured output tool was executed
   - No pending tool calls remain

### Dynamic Tools

Middleware can add tools dynamically via `request.override(tools=[...])` in `wrap_model_call`. However, client-side execution requires either:
1. Tools registered upfront at agent creation, OR
2. Middleware implementing `wrap_tool_call` to execute dynamic tools

If a tool is in the model's binding but not in the `ToolNode`, the factory raises `DYNAMIC_TOOL_ERROR_TEMPLATE` with guidance.

## Structured Output Integration

Structured output allows agents to return typed data alongside messages. The factory integrates three strategies:

### ProviderStrategy (Preferred)

Uses the model's native structured output (e.g., OpenAI's `response_format`):

```python
from pydantic import BaseModel

class WeatherReport(BaseModel):
    location: str
    temperature: int
    conditions: str

agent = create_agent(
    model="openai:gpt-4o",
    tools=[get_weather_tool],
    response_format=WeatherReport,
)
```

The model is asked to return JSON matching the schema directly. Requires model support (auto-detected via profile or fallback patterns).

### ToolStrategy

Uses a special tool call to capture structure:

```python
agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[get_weather_tool],
    response_format=WeatherReport,
)
```

The factory creates a synthetic tool named after the schema (e.g., `WeatherReport`) and forces the model to call it with the desired data. The tool's arguments are parsed as the structured response.

### AutoStrategy

Automatically detects the best strategy:

```python
agent = create_agent(
    model=my_model,
    response_format=WeatherReport,
    # or response_format={"type": "object", ...}
)
```

If `response_format` is a raw Pydantic class or dict, the factory wraps it in `AutoStrategy`, which:
1. Checks the model's profile for native structured output support
2. If supported, converts to `ProviderStrategy`
3. Otherwise, uses `ToolStrategy`

This conversion happens at model binding time, so middleware can override it by setting a different `ResponseFormat` in `wrap_model_call`.

## State Schema Resolution

The factory merges state schemas in this order:

1. Middleware `state_schema`s (in registration order)
2. User-provided `state_schema` (if any)

The user's schema wins on field conflicts. This allows:
- Middleware to extend state without forcing the user to know about it
- User to override middleware state field definitions (e.g., replace a field's reducer)

Example:

```python
class CustomState(AgentState):
    user_id: str  # Add custom field

agent = create_agent(
    model, tools,
    state_schema=CustomState,
    middleware=[SomeMiddleware()],  # SomeMiddleware also extends state
)
```

The final graph uses a merged schema with all fields.

## Graph Compilation and Execution

Once the factory constructs the graph, it compiles it with LangGraph's `StateGraph.compile()`. This:

- **Validates** node and edge definitions
- **Freezes** the schema and topology
- **Prepares** for execution (checkpointing, interrupts, etc.)
- **Returns** a `CompiledStateGraph` object

### Execution Modes

```python
# Synchronous streaming
for chunk in agent.stream({"messages": [...]}, stream_mode="updates"):
    print(chunk)

# Asynchronous streaming
async for chunk in agent.astream({"messages": [...]}):
    print(chunk)

# Blocking invocation
result = agent.invoke({"messages": [...]})
```

### Checkpointing and Persistence

Checkpointers persist state at each node boundary, enabling:
- **Chat memory**: Resume a conversation from any point
- **Human-in-the-loop**: Interrupt, inspect, and resume
- **Debugging**: Replay execution with modified state

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string(":memory:")
agent = create_agent(
    model, tools,
    checkpointer=checkpointer,
    interrupt_before=["model"],  # Pause before model calls
)

# Invoke with a thread ID to save state
config = {"configurable": {"thread_id": "user_123"}}
result = agent.invoke({"messages": [...]}, config=config)

# Resume later
result = agent.invoke({"messages": [...]}, config=config)  # Resumes from checkpoint
```

## Tracing and Observability

The factory integrates with LangSmith for tracing:

- Each middleware hook is traced as a separate span
- Model calls are traced with inputs/outputs
- Tool executions are recorded
- Trace policies can filter sensitive data

### Custom Trace Policies

Middleware can declare a `trace_policy` to shape what is recorded:

```python
from langgraph.types import TracePolicy, omit_payload

class MyMiddleware(AgentMiddleware):
    trace_policy = TracePolicy(process_inputs=omit_payload)
    
    def wrap_model_call(self, request, handler):
        # This hook's span will not include request payloads
        ...
```

A process-wide default can be set:

```python
from langchain.agents.middleware import configure_trace_policy

configure_trace_policy(TracePolicy(process_inputs=omit_payload))
```

This applies to all agents created after the call, even those already instantiated.

## Error Handling

### Structured Output Errors

If structured output parsing fails:
- `StructuredOutputValidationError`: Raised if the parsed JSON doesn't match the schema
- `MultipleStructuredOutputsError`: Raised if the model tried to return multiple structured outputs

The `response_format`'s `handle_errors` parameter controls retry behavior:
```python
response_format = ToolStrategy(
    schema=MySchema,
    handle_errors=True,  # Retry with error message
    # or handle_errors="Custom error message"
    # or handle_errors=(ValueError, TypeError)  # Retry on these exceptions only
)
```

When retry is enabled, the factory adds an error `ToolMessage` to the state and loops back to the model.

### Dynamic Tool Errors

If middleware adds tools to `request.tools` that aren't in the client-side `ToolNode`:
- Factory raises `ValueError` with `DYNAMIC_TOOL_ERROR_TEMPLATE`
- Message includes registered tools and guidance on fixing it
- Mitigation: Either register tools upfront, or implement `wrap_tool_call` to execute dynamic tools

## Extension Points

Developers can customize agents via:

1. **Custom middleware**: Subclass `AgentMiddleware` and implement desired hooks
2. **State extensions**: Declare `state_schema` to add custom fields
3. **Custom nodes**: Add nodes to the graph before compiling (advanced)
4. **Transformers**: Register stream transformers for event filtering (advanced)

Example custom middleware:

```python
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        print(f"Model call #{len(state['messages']) // 2}")
        return None
    
    def wrap_model_call(self, request, handler):
        print(f"  Tools: {[t.name for t in request.tools]}")
        response = handler(request)
        print(f"  Output: {response.result[0].content[:100]}...")
        return response

agent = create_agent(
    model, tools,
    middleware=[LoggingMiddleware()],
)
```

## Configuration Reference

### Key Parameters Summary

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `model` | `str \| BaseChatModel` | required | Language model for the agent |
| `tools` | `Sequence[...]` | `None` | Available tools |
| `system_prompt` | `str \| SystemMessage` | `None` | System context for model |
| `middleware` | `Sequence[AgentMiddleware]` | `()` | Behavior customization |
| `response_format` | `ResponseFormat \| type \| dict` | `None` | Structured output spec |
| `state_schema` | `type[AgentState]` | `None` | Custom state fields |
| `checkpointer` | `Checkpointer` | `None` | State persistence |
| `interrupt_before` | `list[str]` | `None` | Pause before these nodes |
| `interrupt_after` | `list[str]` | `None` | Pause after these nodes |
| `debug` | `bool` | `False` | Verbose logging |
| `name` | `str` | `None` | Graph identifier |

## See Also

- **[Middleware](/openwiki/middleware.md)**: Detailed middleware API and patterns
- **[Structured Output](/openwiki/structured-output.md)**: Deep dive on response schemas
- **[Tools](/openwiki/tools.md)**: Tool definition and integration
- **[Chat Models](/openwiki/chat-models.md)**: Model initialization and binding
- **[Agent Execution](/openwiki/agent-execution.md)**: Runtime behavior and streaming
