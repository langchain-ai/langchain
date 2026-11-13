---
type: "Workflow"
title: "Agent Execution: Agentic Loops and State Management"
description: "Detailed walkthrough of agent lifecycle, execution loops, state management, tool calling, error handling, and observability in LangChain agents."
tags: ["agents", "agentic-loop", "middleware", "tool-calling", "state-management", "observability"]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-10T08:17:41.265Z
sources:
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
  - id: openwiki-source-03e8ca0eebe37feda8566793
    resource: repo://libs/langchain_v1/langchain/agents/middleware/types.py
generated: { by: "openwiki/0.5.0", at: "2026-11-06T08:17:28.425Z" }
---

# Agent Execution: Agentic Loops and State Management

Agent execution in LangChain is built on a composable architecture where language models repeatedly plan, invoke tools, and interpret results in a structured loop. The [`create_agent`][langchain.agents.create_agent] factory function constructs this agentic loop by orchestrating:

1. **Model invocation** with bound tools and system prompt
2. **Middleware composition** (ordered request/response interception)
3. **Tool execution and routing** (looping, termination, or state jumps)
4. **Observability** (tracing, callbacks, state transitions)

This document explains the agent lifecycle, loop control, state management, middleware patterns, error handling, and real-world patterns for production agents.

## Agent Lifecycle: From Initialization to Termination

### Initialization Phase

When `create_agent()` is called, the factory:

1. **Resolves the model**: Converts string identifiers (e.g., `"openai:gpt-4o"`) to `BaseChatModel` instances via `init_chat_model()`.
2. **Normalizes tools**: Converts callables to `BaseTool` instances, preserves `BaseTool` and dict-based tools.
3. **Sets up structured output**: Creates pseudo-tools if `response_format` is provided (for `ToolStrategy`).
4. **Composes middleware**: Chains `wrap_model_call` and `wrap_tool_call` handlers, merges state schemas, registers node hooks.
5. **Constructs the graph**: Builds a LangGraph `StateGraph` with model, tools, and middleware nodes.
6. **Compiles**: Returns a `CompiledStateGraph` ready for execution.

### Agent State and State Schema

Every agent operates over an `AgentState` that persists across the execution loop:

```python
class AgentState(TypedDict, Generic[ResponseT]):
    """State schema for the agent."""
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue]]
    structured_response: NotRequired[Annotated[ResponseT, OmitFromOutput]]
```

**State fields:**
- **`messages`**: Conversation history, merged via `add_messages` reducer (new messages append, not replace).
- **`jump_to`**: Ephemeral routing hint set by middleware to override default flow ("tools", "model", or "end").
- **`structured_response`**: Parsed output when `response_format` is configured (omitted from input schema).

Middleware can extend `AgentState` via `state_schema` parameter. When multiple middleware extend state, schemas are merged in order with user-provided `state_schema` winning conflicts.

## The Agentic Loop

The core loop iterates through three main phases until a stopping condition is met:

### Phase 1: Model Invocation

The model node calls the language model with the current message history and bound tools. Returns an `AIMessage` with optional `tool_calls`.

**Flow:**
1. `before_model` middleware hooks execute (state updates applied).
2. Model is invoked with system prompt, messages, and bound tools.
3. `after_model` middleware hooks execute (state updates applied).
4. Response is handled (structured output parsing, invalid tool repair).

### Phase 2: Conditional Routing

After the model produces a response, the agent decides what to do next:

- **If `AIMessage.tool_calls` is present**: Route to tools node.
- **If `state["jump_to"] == "model"`**: Loop back to `before_model`.
- **If `state["jump_to"] == "end"`**: Skip to termination.
- **If no tool calls and no structured output**: Terminate.

### Phase 3: Tool Execution

If routing to tools, the agent:

1. Matches each tool call to a registered tool.
2. Executes the tool (wrapped by middleware `wrap_tool_call` handlers).
3. Returns `ToolMessage` with result.
4. Determines routing: loop back to model, or exit if `return_direct=True` or structured output was produced.

### Loop Termination

The agent stops when:

- The model produces an `AIMessage` with no tool calls.
- A tool has `return_direct=True` and produces output.
- A structured output tool completes (automatic termination with `ToolStrategy`).
- `state["jump_to"] == "end"` is set by middleware.

**Termination phase:**
1. `after_agent` middleware hooks execute once (cleanup, logging, state finalization).
2. Graph returns final state (messages + optional structured response).

### Control Flow Diagram

```mermaid
graph TD
    START["START"] --> BEFORE_AGENT["before_agent hooks<br/>(run once)"]
    BEFORE_AGENT --> BEFORE_MODEL["before_model hooks<br/>(before each model call)"]
    BEFORE_MODEL --> MODEL["Model Node<br/>(invoke LLM)"]
    MODEL --> AFTER_MODEL["after_model hooks<br/>(after each model call)"]
    
    AFTER_MODEL --> ROUTE{jump_to set?}
    ROUTE -->|jump_to=='end'| AFTER_AGENT["after_agent hooks<br/>(run once)"]
    ROUTE -->|jump_to=='model'| BEFORE_MODEL
    ROUTE -->|no jump_to| ROUTE2{Tool calls?}
    
    ROUTE2 -->|No| ROUTE3{Structured<br/>output?}
    ROUTE3 -->|Yes| TOOLS["Tools Node<br/>(structured tools)"]
    ROUTE3 -->|No| AFTER_AGENT
    
    ROUTE2 -->|Yes| TOOLS
    TOOLS --> ROUTE4{return_direct<br/>or structured?}
    ROUTE4 -->|Yes| AFTER_AGENT
    ROUTE4 -->|No| BEFORE_MODEL
    
    AFTER_AGENT --> END["END"]
```

## Creating an Agent with `create_agent()`

### Minimal Example

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for {query}"

agent = create_agent(
    model="openai:gpt-4o",
    tools=[search],
    system_prompt="You are a helpful assistant"
)

result = agent.invoke({"messages": [{"role": "user", "content": "Search for AI news"}]})
print(result["messages"][-1].content)
```

### Factory Signature

```python
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
) -> CompiledStateGraph[AgentState[ResponseT], ContextT, InputAgentState, OutputAgentState[ResponseT]]
```

### Key Parameters

| Parameter | Purpose | Notes |
|-----------|---------|-------|
| `model` | Language model | String (e.g., `"openai:gpt-4o"`) or `BaseChatModel` instance |
| `tools` | Tool functions/schemas | `BaseTool`, callables, or dict schemas; can be empty |
| `system_prompt` | System instructions | String or `SystemMessage`; prefixed to every model call |
| `middleware` | Pre/post hooks and wrappers | Ordered list; first = outermost layer |
| `response_format` | Structured output schema | `ToolStrategy`, `ProviderStrategy`, or Pydantic class |
| `state_schema` | Custom state fields | Extends `AgentState` with application state |
| `checkpointer` | Conversation memory | Saves state per thread; enables resumption |
| `store` | Cross-thread persistence | Retrieve context across conversations |
| `interrupt_before/after` | Human-in-the-loop | Pause before/after named nodes |
| `debug` | Verbose logging | Prints execution trace |
| `name` | Graph name | Used for subagent identification in multi-agent systems |
| `cache` | Graph-level caching | Cache entire graph outputs |
| `transformers` | Stream output formatters | Custom `StreamTransformer` factories |

## Tool Binding and Execution

### Tool Types

Tools are bound to the model during agent creation. The factory accepts three formats:

1. **BaseTool instances** (preferred):
   ```python
   from langchain_core.tools import tool
   
   @tool
   def search(query: str) -> str:
       """Search for information."""
       return f"Results for {query}"
   
   agent = create_agent(model="openai:gpt-4o", tools=[search])
   ```

2. **Python callables** (auto-wrapped):
   ```python
   def search(query: str) -> str:
       """Search for information."""
       return f"Results for {query}"
   
   agent = create_agent(model="openai:gpt-4o", tools=[search])
   ```

3. **Dict schemas** (provider built-ins):
   ```python
   tools = [
       {
           "type": "builtin_text_search",  # Provider-specific
           "name": "search",
           "description": "Search for information"
       }
   ]
   agent = create_agent(model="anthropic:claude-opus", tools=tools)
   ```

### Tool Node and Execution Flow

When tools are provided, `create_agent` constructs a `ToolNode` that:

1. **Receives**: Model's `AIMessage` with `tool_calls`.
2. **Matches**: Each call to a registered tool by name.
3. **Executes**: Tool (sync or async, error handling included).
4. **Wraps**: Result in `ToolMessage` with tool call ID and name.
5. **Applies middleware**: `wrap_tool_call` handlers intercept execution.

The ToolNode validates that:
- All tool names referenced by the model are registered.
- Tools return proper result strings/structured data.
- Exceptions are caught and returned as error `ToolMessage` objects.

If a tool call fails during execution, the framework automatically generates an error `ToolMessage` and loops back to the model for recovery.

### Model Binding and Structured Output

Before calling the model, `create_agent` binds tools via `model.bind_tools()`:

```python
bound_model = model.bind_tools(
    final_tools,
    tool_choice=tool_choice,  # "any", "required", specific name, or None
    **model_settings
)
```

**Structured output** (when `response_format` is configured) adds special behavior:

- **ProviderStrategy**: Uses the model's native structured output API (e.g., OpenAI's response_format).
- **ToolStrategy**: Creates pseudo-tools for each schema; model calls the tool to return structured data.
- **AutoStrategy**: Auto-detects the best strategy based on model capabilities.

When ToolStrategy is used, the agent automatically:
1. Adds structured tools to the model's tool list.
2. Executes the tool call to parse the schema.
3. Stores the parsed result in `state["structured_response"]`.
4. Routes to termination (not looping back for more tools).

## Middleware: Hooks and Wrappers

Middleware instances allow customization at every stage of agent execution. They inherit from `AgentMiddleware` and implement any combination of hooks and wrappers.

### Hook-Based Middleware (Node Hooks)

Node hooks are called at specific points in the execution flow, returning state updates:

```python
class MyMiddleware(AgentMiddleware):
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Called once before agent execution starts."""
        # Inject initial state, validate inputs, etc.
        return {"custom_field": "value"}
    
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Called once after agent execution completes."""
        # Log final state, cleanup, etc.
        return None
    
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Called before each model invocation (runs in loop)."""
        # Modify messages, set tool constraints, etc.
        return None
    
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Called after each model invocation (runs in loop)."""
        # Validate model output, log decisions, etc.
        return None
```

**Routing via `jump_to`:**
Hooks can set `state["jump_to"]` to alter the execution flow:
- `"model"`: Jump to model node (skip tools, loop back).
- `"tools"`: Jump to tools node (execute tools).
- `"end"`: Skip to agent termination.

### Wrapper-Based Middleware (Handler Wraps)

Wrappers intercept and control core operations by wrapping handler callbacks. Each wrapper receives a `handler` callable that executes the next layer.

#### `wrap_model_call` – Model Retry and Caching

```python
from langchain.agents import ModelRequest, ModelResponse, AgentMiddleware

class RetryMiddleware(AgentMiddleware):
    def wrap_model_call(self, request: ModelRequest, handler):
        """Retry model call up to 3 times on error."""
        for attempt in range(3):
            try:
                response = handler(request)  # Call the model
                return response
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"Attempt {attempt + 1} failed: {e}")
```

`wrap_model_call` can:
- **Retry** on transient errors (exponential backoff).
- **Cache** responses by request hash.
- **Fallback** to a different model.
- **Modify request** (tools, system message, model settings).
- **Rewrite response** (transform AI message content).
- **Short-circuit** by not calling the handler (return cached result).

**Return types:**
- `ModelResponse`: Full response with messages and optional structured output.
- `AIMessage`: Simplified response (auto-converted to `ModelResponse`).
- `ExtendedModelResponse`: Response + `Command` for additional state updates.

#### `wrap_tool_call` – Tool Validation and Retry

```python
class ToolRetryMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        """Retry tool execution if result is invalid."""
        for attempt in range(2):
            result = handler(request)  # Execute the tool
            if self._is_valid(result):
                return result
        return result
    
    def _is_valid(self, result):
        # Custom validation logic
        return "error" not in result.content.lower()
```

`wrap_tool_call` can:
- **Validate** tool arguments before execution.
- **Retry** on errors or invalid outputs.
- **Monitor** tool usage and cost.
- **Modify calls** (transform arguments, inject context).
- **Mock** tools for testing.

**Return types:**
- `ToolMessage`: Tool execution result.
- `Command`: Route to a different node or update state.

### Middleware Composition and Ordering

Middleware is applied in **first-in, outermost** order:

```python
middleware = [CacheMiddleware(), RetryMiddleware(), LoggingMiddleware()]
agent = create_agent(model="openai:gpt-4o", tools=[...], middleware=middleware)
```

**Execution order for `wrap_model_call`:**
- Request: `CacheMiddleware` → `RetryMiddleware` → `LoggingMiddleware` → actual model
- Response: actual model → `LoggingMiddleware` → `RetryMiddleware` → `CacheMiddleware`

**State schema composition:**
Middleware state schemas are merged in order: `middleware[0].state_schema`, `middleware[1].state_schema`, ..., then `agent.state_schema` (if provided). The agent's explicit `state_schema` always wins field conflicts, allowing users to override middleware annotations.

## Agent Execution Flow

### Synchronous vs. Asynchronous

`create_agent` returns a compiled graph that supports both sync and async execution:

```python
# Synchronous
result = agent.invoke({"messages": [...]})

# Asynchronous
result = await agent.ainvoke({"messages": [...]})

# Streaming (sync)
for chunk in agent.stream({"messages": [...]}, stream_mode="updates"):
    print(chunk)

# Streaming (async)
async for chunk in agent.astream({"messages": [...]}):
    print(chunk)
```

The graph automatically chooses sync or async based on the execution context. If a middleware implements only async (`awrap_model_call`) and is invoked synchronously, it raises `NotImplementedError`. Provide both implementations for maximum compatibility.

### Detailed Execution Steps

1. **Initialization** (`before_agent`):
   - Middleware `before_agent` hooks run once, allowing setup and state injection.
   - State updates are merged using reducers (e.g., `add_messages` for messages).

2. **Loop iteration**:
   - **before_model**: Hooks run, allowing message filtering, tool narrowing, etc.
   - **model**: Model is called with current messages + system prompt + bound tools.
   - **after_model**: Hooks run for logging, validation, or routing overrides.

3. **Routing decision**:
   - If `state["jump_to"]` is set: Use it to route to target node.
   - Else if `AIMessage` has `tool_calls`: Route to tools node.
   - Else if no tool calls and no structured output: Route to termination.

4. **Tool execution** (if routed to tools):
   - Each tool call is matched to a registered tool.
   - `wrap_tool_call` middleware intercepts execution.
   - Tool result is wrapped in a `ToolMessage`.
   - Routing: If any tool has `return_direct=True` or structured output was produced: skip tools loop, go to termination.
   - Otherwise: Loop back to `before_model`.

5. **Termination** (`after_agent`):
   - Middleware `after_agent` hooks run once for cleanup, logging, or final state updates.
   - Graph returns the final state (messages + optional structured response).

## Loop Control: Max Iterations and Stopping Criteria

### Max Iterations

The agent loop has a default recursion limit of 9,999 iterations to prevent infinite loops. You can configure this per invocation:

```python
result = agent.invoke(
    {"messages": [...]},
    config={"recursion_limit": 100}  # Custom limit
)
```

### Stopping Criteria

The agent stops under these conditions:

1. **No tool calls**: Model returns `AIMessage` with empty `tool_calls` list.
2. **Structured output complete**: A `ToolStrategy` structured tool was executed.
3. **return_direct flag**: A tool with `return_direct=True` completes.
4. **jump_to == "end"**: Middleware explicitly routes to termination.
5. **Recursion limit exceeded**: Graph hits configured recursion limit.

## Observability: Tracing and Callbacks

### LangSmith Integration

All agent execution is automatically traced to LangSmith (if configured). Traces capture:

- **Model node invocations** (with tool bindings and system prompt).
- **Middleware hooks** (before/after execution points).
- **wrap_model_call and wrap_tool_call** interceptions (request/response details).
- **Tool executions** (with arguments, results, and errors).
- **State transitions** (messages, structured response, routing decisions).

Traces are scoped hierarchically: outer middleware wraps inner, showing composition visually.

### Trace Policy and Privacy

Middleware can configure trace privacy via `trace_policy`:

```python
from langgraph.types import TracePolicy, omit_payload

class MyMiddleware(AgentMiddleware):
    trace_policy = TracePolicy(process_inputs=omit_payload)  # Hide sensitive inputs
```

This allows middleware to drop message payloads while keeping timing and span structure.

### Debug Mode

Enable verbose logging to stdout:

```python
agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    debug=True
)

result = agent.invoke({"messages": [...]})
# Prints: node transitions, state updates, execution trace
```

Debug output shows:
- Each node invocation.
- State updates at each step.
- Tool calls and results.
- Routing decisions.

### State Inspection

Access the runtime context in middleware hooks:

```python
class ObservableMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # runtime.config contains execution configuration
        # runtime.step_number shows iteration count
        print(f"Step {runtime.step_number}: {len(state['messages'])} messages")
        return None
```

## Common Middleware Patterns

### Pattern 1: Tool Selection and Routing

Narrow the tools available to the model based on context:

```python
from langchain.agents import AgentMiddleware, ModelRequest

class ContextualToolSelector(AgentMiddleware):
    def before_model(self, state, runtime):
        # Narrow tools based on conversation context
        user_level = state.get("user_level", "basic")
        if user_level == "admin":
            # Admin has access to all tools
            pass
        else:
            # Regular users see subset
            pass
        return None
    
    def wrap_model_call(self, request: ModelRequest, handler):
        # Filter tools dynamically
        advanced_tools = [t for t in request.tools if "admin" not in t.name]
        request = request.override(tools=advanced_tools)
        return handler(request)
```

### Pattern 2: Retry and Error Recovery

Implement exponential backoff or fallback strategies:

```python
import asyncio
from langchain.agents import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

class SmartRetryMiddleware(AgentMiddleware):
    async def awrap_model_call(self, request: ModelRequest, handler):
        backoff = 1
        for attempt in range(3):
            try:
                return await handler(request)
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    # Fallback: return safe response
                    return ModelResponse(
                        result=[AIMessage(content="I encountered an error. Please try again.")]
                    )
```

### Pattern 3: Output Formatting and Validation

Post-process model responses to ensure compliance:

```python
from langchain.agents import AgentMiddleware, ModelResponse
from langchain_core.messages import AIMessage

class OutputFormatterMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        response = handler(request)
        # Ensure response is ModelResponse
        if isinstance(response, AIMessage):
            response = ModelResponse(result=[response])
        
        # Post-process AI message
        ai_msg = response.result[0]
        formatted_content = f"Response: {ai_msg.content}"
        return ModelResponse(
            result=[AIMessage(content=formatted_content, tool_calls=ai_msg.tool_calls)],
            structured_response=response.structured_response
        )
```

### Pattern 4: Tool Error Handling and Fallback

Handle tool execution failures gracefully:

```python
from langchain.agents import AgentMiddleware
from langchain_core.messages import ToolMessage

class ToolErrorHandlerMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        try:
            return handler(request)
        except Exception as e:
            # Return error message instead of crashing
            return ToolMessage(
                content=f"Tool error: {str(e)}. Please retry or use a different tool.",
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
                status="error"
            )
```

### Pattern 5: Dynamic Tool Registration

Register tools at runtime based on context:

```python
from langchain.agents import AgentMiddleware, ModelRequest

class DynamicToolMiddleware(AgentMiddleware):
    tools = []  # Will be populated at runtime
    
    def before_model(self, state, runtime):
        # Update available tools based on user context
        user_id = state.get("user_id")
        if user_id:
            self.tools = self._load_user_tools(user_id)
        return None
    
    def wrap_model_call(self, request: ModelRequest, handler):
        # Add dynamic tools to request
        all_tools = list(request.tools) + self.tools
        request = request.override(tools=all_tools)
        return handler(request)
```

### Pattern 6: Structured Output with Fallback

Handle both provider-native and tool-based structured output:

```python
from langchain.agents import AgentMiddleware, ModelRequest
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy

class StructuredOutputMiddleware(AgentMiddleware):
    def wrap_model_call(self, request: ModelRequest, handler):
        # Try provider strategy first
        if request.response_format:
            try:
                return handler(request)
            except Exception:
                # Fall back to tool strategy
                request = request.override(
                    response_format=ToolStrategy(schema=request.response_format.schema)
                )
                return handler(request)
        return handler(request)
```

## Error Handling and Resilience

### Invalid Tool Calls

When a model generates a tool call with:
- **Malformed arguments** (JSON parse error, missing required fields)
- **Unknown tool name**

The agent automatically patches these by inserting error `ToolMessage` objects before retrying the model. Middleware can intercept these via `wrap_model_call` or `after_model` hooks.

### Middleware-Induced Tool Mismatches

If middleware adds tools via `wrap_model_call` but doesn't handle their execution:

```
ValueError: DYNAMIC_TOOL_ERROR_TEMPLATE
Unknown tools: [dynamic_tool]
```

**Solutions:**
1. Register tools upfront via `create_agent(tools=[...])`.
2. Implement `wrap_tool_call` in the middleware to handle dynamic tools.
3. Use dict-based tools if the provider natively supports them.

### Async-Sync Context Mismatch

If middleware implements only async (`awrap_model_call`) but the agent is invoked synchronously:

```
NotImplementedError: Asynchronous implementation of awrap_model_call is not available.
```

**Solutions:**
1. Implement both `wrap_model_call` and `awrap_model_call`.
2. Use decorators: `@wrap_model_call` for sync, then adapt for async.
3. Invoke the agent asynchronously: `agent.ainvoke()` or `agent.astream()`.

### Tool Execution Errors

Tool execution errors are caught and returned as `ToolMessage` with `status="error"`. The agent loops back to the model with the error message, allowing it to recover or try a different tool.

## Practical Example: Multi-Tool Planner Agent

```python
from langchain.agents import create_agent, AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

# Define tools
@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

@tool
def calculate(expr: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expr))

# Middleware for planning
class PlannerMiddleware(AgentMiddleware):
    def before_agent(self, state, runtime):
        # Inject a planning step at the start
        plan_request = HumanMessage(
            content="First, make a plan for how to solve this problem. Then execute it."
        )
        return {"messages": [plan_request]}

# Create agent
agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[search, calculate],
    system_prompt="You are a helpful assistant with access to search and math tools.",
    middleware=[PlannerMiddleware()],
    debug=True
)

# Run
result = agent.invoke({
    "messages": [{"role": "user", "content": "What is the population of France times 2?"}]
})

print(result["messages"][-1].content)
```

## Performance and Scaling

### Checkpointing for Resumption

Store agent state per conversation thread to enable pausing and resuming:

```python
from langgraph.checkpoint.memory import MemorySaver

agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    checkpointer=MemorySaver()
)

# First invocation
thread = {"configurable": {"thread_id": "user_123"}}
result = agent.invoke({"messages": [...]}, thread)

# Resume later
result2 = agent.invoke({"messages": [...]}, thread)  # Resumes from saved state
```

### Cross-Thread Persistence

Store data accessible across multiple conversations:

```python
from langgraph.store.memory import InMemoryStore

agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    store=InMemoryStore()
)

# Middleware can query the store
class ContextMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        # Load user context from store
        user_id = state.get("user_id")
        if user_id and runtime.store:
            context = runtime.store.get("user_context", user_id)
            # Use context to shape model behavior
        return None
```

### Caching and Streaming

Enable graph-level caching and configure stream transformers:

```python
from langgraph.cache.base import BaseCache

agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    cache=my_cache,  # Cache entire graph outputs
    transformers=[my_transformer]  # Custom stream formatters
)

# Stream with updates
for chunk in agent.stream(
    {"messages": [...]},
    stream_mode="updates",
    config={"recursion_limit": 100}
):
    print(chunk)
```

## Testing and Debugging

### Debug Mode

Enable verbose logging:

```python
agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    debug=True
)

result = agent.invoke({"messages": [...]}, debug=True)
# Prints node transitions, state updates, and execution trace
```

### Mocking Tools for Unit Tests

```python
from langchain_core.tools import tool

@tool
def mock_search(query: str) -> str:
    """Mock search tool."""
    if "AI" in query:
        return "Artificial Intelligence is..."
    return "No results"

agent = create_agent(
    model="openai:gpt-4o",
    tools=[mock_search]
)

# Test agent
result = agent.invoke({"messages": [{"role": "user", "content": "Tell me about AI"}]})
assert "Artificial Intelligence" in result["messages"][-1].content
```

### Testing Middleware

```python
from langchain.agents import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

def test_retry_middleware():
    middleware = SmartRetryMiddleware()
    request = ModelRequest(model=model, messages=[], tools=[])
    
    call_count = 0
    def failing_handler(req):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Transient error")
        return ModelResponse(result=[AIMessage(content="Success")])
    
    result = middleware.wrap_model_call(request, failing_handler)
    assert call_count == 2  # Retried once
    assert "Success" in result.result[0].content
```

## Integration with LangGraph

Agents created by `create_agent()` are LangGraph `StateGraph` instances that can:

1. **Be subgraphs**: Added to larger agent networks for multi-agent systems.
2. **Use interrupts**: Pause execution for human review or intervention.
3. **Stream outputs**: Real-time token-by-token responses.
4. **Support nested graphs**: Hierarchical agentic structures.

The `name` parameter helps identify subagents in trace logs and error messages.

## Related Concepts

- **Chat Models**: Language models that accept messages and return responses (see `/openwiki/concepts/chat-models.md`)
- **Tools**: Schema-driven functions that agents can invoke (see `/openwiki/concepts/tools.md`)
- **Middleware Framework**: Base classes and decorators for custom extensions (see `/openwiki/workflows/middleware.md`)
- **Streaming**: Real-time output via `stream()` and `astream()`
- **Structured Output**: Type-safe results from language models
- **Composability**: Building complex workflows from agents and tools (see `/openwiki/workflows/composability.md`)
