---
type: Workflow
title: "Agent Composability: Building Complex Agents"
description: "Patterns for composing agents through middleware layers, subagent delegation, tool selection strategies, and context management across hierarchical agent workflows."
tags: [agents, middleware, composition, subagents, hierarchical, extensibility, tool-management, state-merging]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-07T08:15:07.286Z
sources:
  - id: openwiki-source-65071982a626569c8820a34b
    resource: repo://libs/core/langchain_core/runnables/__init__.py
  - id: openwiki-source-a1981e868973f6fd7f71e12e
    resource: repo://libs/core/langchain_core/runnables/base.py
  - id: openwiki-source-48e94bbe49ab4f33ba87e9cb
    resource: repo://libs/core/langchain_core/runnables/branch.py
  - id: openwiki-source-079792f059657900794e2955
    resource: repo://libs/core/langchain_core/runnables/config.py
  - id: openwiki-source-f9f4c1dc4f9cdf80d824ce15
    resource: repo://libs/core/langchain_core/runnables/fallbacks.py
generated: { by: "openwiki/0.5.0", at: "2026-11-07T08:15:07.286Z" }
---

# Agent Composability: Building Complex Agents

Agent composability in LangChain enables building sophisticated multi-agent systems through two fundamental mechanisms: **middleware composition** for layering cross-cutting concerns onto a single agent, and **subagent delegation** for spawning nested agents from tool calls. Together, these patterns enable composable, reusable agent behaviors without monolithic implementations.

## Middleware Composition: Layered Customization

Middleware provides composable request/response interception at multiple execution phases. Multiple middleware instances compose into a single middleware stack where the first instance wraps all others, creating predictable layering.

### Middleware Registration and Composition Order

Middleware is registered as a sequence on `create_agent()`:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware

graph = create_agent(
    model="openai:gpt-5.5",
    tools=[...],
    middleware=[
        LoggingMiddleware(),      # Outermost: sees all requests first
        ValidationMiddleware(),    # Middle: validates after logging
        CacheMiddleware(),         # Innermost: caches before execution
    ]
)
```

**Composition semantics**: Request flows from outermost to innermost (logging → validation → cache → execution), response flows back (execution → cache → validation → logging). Each middleware receives a `handler` callback to invoke the next layer.

### Hooking Points and Execution Phases

Middleware can intercept at six lifecycle phases and two execution wrappers:

**Lifecycle hooks** (state transitions):
- `before_agent` / `abefore_agent`: Runs once at agent start, before any model call
- `before_model` / `abefore_model`: Before each model invocation
- `after_model` / `aafter_model`: After model returns (before tool routing)
- `after_agent` / `aafter_agent`: After agent terminates

**Execution wrappers** (function interception):
- `wrap_model_call` / `awrap_model_call`: Wraps LLM invocation; can retry, cache, modify request/response
- `wrap_tool_call` / `awrap_tool_call`: Wraps tool execution; can validate, audit, route, or skip

Each hook method has sync and async variants. When using `invoke()` or `stream()`, sync variants are called; `ainvoke()` or `astream()` call async variants. At least one variant must be implemented; if only async is defined, sync invocations raise `NotImplementedError`.

### Handler Composition Pattern

Middleware handlers compose via closures, chaining inner calls with accumulated state:

```python
def _chain_model_call_handlers(
    handlers: Sequence[_ModelCallHandler],
) -> _ComposedModelCallHandler | None:
    """Compose handlers where first = outermost layer."""
    
    if len(handlers) == 1:
        return handlers[0]
    
    def compose_two(outer, inner):
        """Compose two handlers where outer wraps inner."""
        def composed(request, handler):
            accumulated_commands = []
            
            def inner_handler(req):
                # Capture commands from inner before normalizing
                inner_result = inner(req, handler)
                if isinstance(inner_result, _ComposedExtendedModelResponse):
                    accumulated_commands.extend(inner_result.commands)
                    return inner_result.model_response
                return _normalize_to_model_response(inner_result)
            
            outer_result = outer(request, inner_handler)
            # Merge accumulated commands from inner with outer result
            return _to_composed_result(outer_result, accumulated_commands)
        
        return composed
    
    # Compose right-to-left: outermost(inner(innermost))
    composed = compose_two(handlers[-2], handlers[-1])
    for h in reversed(handlers[:-2]):
        composed = compose_two(h, composed)
    
    return composed
```

**Key properties**:
- **Command accumulation**: Commands from all middleware layers (inner-first) are collected into a list
- **Normalization**: Handler results (`ModelResponse`, `AIMessage`, `ExtendedModelResponse`) are normalized consistently
- **Clear/retry safety**: Commands are cleared on each retry within middleware to avoid stale accumulation

### State Schema Merging

Middleware can extend agent state via `state_schema` parameter. When multiple middleware declare schemas, they are merged in declaration order with user-supplied `state_schema` winning conflicts:

```python
from typing_extensions import TypedDict
from langchain.agents.middleware import AgentState, OmitFromInput, PrivateStateAttr

class CacheMiddlewareState(TypedDict):
    cache_hits: int  # Visible in input/output
    cache_metadata: Annotated[dict, PrivateStateAttr]  # Internal only

class MyMiddleware(AgentMiddleware):
    state_schema = CacheMiddlewareState
```

Schema resolution follows this order:
1. Middleware state schemas (in registration order)
2. User-supplied `state_schema` parameter (if provided)
3. Base `AgentState` (default)

Later schemas override earlier ones on field conflicts. Use `OmitFromSchema`, `OmitFromInput`, and `OmitFromOutput` to exclude fields from specific schemas.

## Subagent Delegation: Hierarchical Composition

Subagent delegation enables agents to spawn nested agents from within tool calls. A tool in the parent agent can invoke another `create_agent()` graph as a nested execution, with automatic lifecycle tracking and causation edges.

### Subagent Creation and Binding

Subagents are named agents spawned via tool calls in the parent:

```python
from langchain.agents import create_agent

# Parent agent with a tool that delegates to a subagent
def delegate_to_research_team(query: str) -> str:
    """Delegate a research task to the research team subagent."""
    research_agent = create_agent(
        name="research_team",
        model="openai:gpt-5.5",
        tools=[web_search, summarize],
        system_prompt="You are a research specialist..."
    )
    
    result = research_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content

# Parent agent with delegation tool
parent = create_agent(
    model="openai:gpt-5.5",
    tools=[delegate_to_research_team],
    system_prompt="You are a manager coordinating specialized teams..."
)
```

**Subagent boundaries are detected via**:
- `lc_agent_name`: A field in run config set by `create_agent(name=...)` to identify the agent
- `lc_agent_name` differs from parent's: A nested run with a different `lc_agent_name` is a subagent boundary
- **Same-named nested agents** (e.g., recursive calls) are still surfaced; unnamed agents (`lc_agent_name=None`) are excluded

### SubagentRunStream and Causation Tracking

When a subagent is detected, the `SubagentTransformer` (registered by default) surfaces a typed handle on `run.subagents`:

```python
class SubagentRunStream(SubgraphRunStream):
    """Typed sync handle for nested named-agent execution.
    
    Surfaces on `run.subagents` when nested run's `lc_agent_name` differs
    from parent's.
    """
    
    @property
    def name(self) -> str | None:
        """Subagent name (the nested run's `lc_agent_name`)."""
        return self.graph_name
    
    @property
    def cause(self) -> LifecycleCause | None:
        """Causation edge — the tool call that triggered this subagent.
        
        Returns `{"type": "toolCall", "tool_call_id": ...}` dict when the
        originating tool call could be joined, else `None`.
        """
        return self._cause
```

**Causation tracking**: The transformer recovers the originating tool call from the parent task's pending tool calls and exposes it via `cause`. This creates a verifiable edge from parent tool invocation to child execution.

**Async variant**: `AsyncSubagentRunStream` provides the same interface for async execution paths.

### Subagent Lifecycle and State Isolation

Subagents execute in their own state namespace and have independent lifecycles:

```python
# Parent invocation
for event in parent.stream({"messages": [...]}, stream_mode="updates"):
    if "research_team" in event:  # Subagent updates
        for subagent_event in event["research_team"]["subagents"]:
            print(f"Subagent: {subagent_event.name}")
            print(f"Caused by: {subagent_event.cause}")
```

**Lifecycle semantics**:
- Subagent is created on first tool call to its delegating function
- Subagent state (messages, state fields) is isolated from parent
- Parent's model sees only the tool result returned by the subagent
- Subagent completion is tracked and can be awaited independently
- Multiple subagents run sequentially within the same tool call (no auto-parallelism across subagents)

## Tool Selection and Dynamic Tool Management

Middleware can dynamically modify tools at request time, enabling tool selection, authorization, and feature composition.

### Tool Override and Dynamic Addition

The `wrap_model_call` middleware receives and can override `request.tools`:

```python
class ToolSelectionMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        """Select tools based on context."""
        # Filter tools by category
        if request.state.get("context") == "research":
            request = request.override(
                tools=[t for t in request.tools if t.name.startswith("web_")]
            )
        
        # Add runtime tools for this specific context
        if request.state.get("needs_calculator"):
            request = request.override(
                tools=request.tools + [self.calculator_tool]
            )
        
        return handler(request)
```

**Tool resolution**:
- **Built-in tools** (dict format): Defined by model provider (e.g., Claude's native tools)
- **Client-side tools** (BaseTool/callable): Registered upfront or added by middleware
- **Validation**: Unknown client-side tools raise `ValueError` unless `wrap_tool_call` is defined (dynamic tool handling)

### Structured Output Tools

When `response_format` is provided with `ToolStrategy`, pseudo-tools are created to represent structured outputs:

```python
from pydantic import BaseModel
from langchain.agents.structured_output import ToolStrategy

class ArticleOutput(BaseModel):
    title: str
    summary: str

graph = create_agent(
    model="openai:gpt-5.5",
    response_format=ToolStrategy(schema=ArticleOutput),
    # Pseudo-tool "__structured_output_ArticleOutput__" is auto-created
)
```

These tools are not available for manual tool calls—they represent the final structured response and trigger automatic termination on completion.

## Context Management and Configuration Threading

Middleware and subagents operate within a shared context that threads configuration and runtime state down the execution stack.

### RunnableConfig Threading

`RunnableConfig` (tags, metadata, callbacks, max_concurrency, configurable fields) is propagated through composed chains automatically:

```python
result = agent.invoke(
    {"messages": [...]},
    config={
        "tags": ["production", "critical"],
        "metadata": {"user_id": "123", "session": "abc"},
        "callbacks": [custom_callback],
        "max_concurrency": 4,
    }
)
```

**Threading semantics**:
- Configuration is merged at composition boundaries
- Child values override parent values
- List fields (tags, metadata) accumulate rather than replace
- Context variables propagate config down the call stack automatically

### Runtime Context and Subagent Coordination

Agents execute within a `Runtime` context that provides access to checkpointer, store, and other infrastructure:

```python
class MyMiddleware(AgentMiddleware):
    async def abefore_model(self, state, runtime):
        """Access shared infrastructure via runtime."""
        # runtime.checkpointer: For persisting state
        # runtime.store: For cross-agent data sharing
        # runtime.created_at: Execution start time
        return None
```

**Subagent coordination** through shared store:
```python
def delegate_with_shared_context(query: str) -> str:
    """Subagent can access parent's store data."""
    research_agent = create_agent(
        name="research",
        model="openai:gpt-5.5",
        tools=[web_search],
        store=shared_store,  # Same store as parent
    )
    return research_agent.invoke({"messages": [...]})["messages"][-1].content
```

## Extension Points and Best Practices

### Writing Custom Middleware

Middleware should follow these principles:

1. **Single responsibility**: Each middleware handles one concern (validation, caching, logging)
2. **Composability**: Never assume it's the only middleware; use the `handler` callback
3. **Immutability**: Use `request.override()` instead of direct assignment
4. **Command clarity**: Accumulate commands in `ExtendedModelResponse` for state updates
5. **Async support**: Implement both sync and async variants or use bridging decorators

```python
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

class CustomMiddleware(AgentMiddleware):
    """Example middleware with both phases."""
    
    def before_model(self, state, runtime):
        """Prepare before model call."""
        return {"prepared": True}
    
    def wrap_model_call(self, request, handler):
        """Wrap and potentially modify model call."""
        # Validate request
        if not request.messages:
            raise ValueError("No messages")
        
        # Call next layer
        response = handler(request)
        
        # Post-process response
        return response
```

### Subagent Best Practices

1. **Name subagents**: Always use `create_agent(name=...)` to enable detection
2. **Isolate concerns**: Each subagent should have a focused responsibility
3. **Share infrastructure**: Use `store` for cross-agent knowledge
4. **Handle failures**: Subagent errors surface as tool errors in parent
5. **Monitor causation**: Use `cause` edge to trace parent-child relationships

### Testing and Debugging

```python
from langchain.agents import create_agent

# Enable debug output
agent = create_agent(
    model="openai:gpt-5.5",
    tools=[...],
    debug=True,  # Verbose logging of execution
)

# Trace middleware execution
for event in agent.stream(
    {"messages": [...]},
    stream_mode="updates",
):
    print(f"Event: {event}")
```

## Composition Patterns and Real-World Examples

### Multi-Agent Hierarchies

Parent agents orchestrate multiple specialized subagents:

```python
# Specialized subagents
def research_task(query: str) -> str:
    return create_agent(name="researcher", ...)({"messages": [...]})

def analysis_task(data: str) -> str:
    return create_agent(name="analyst", ...)({"messages": [...]})

# Coordinator agent
coordinator = create_agent(
    model="openai:gpt-5.5",
    tools=[research_task, analysis_task],
    system_prompt="You coordinate research and analysis tasks..."
)
```

### Middleware Stacks for Cross-Cutting Concerns

```python
# Security stack
security_stack = [
    RateLimitMiddleware(max_calls_per_minute=10),
    ToolAuthorizationMiddleware(permissions_db),
    AuditLoggingMiddleware(),
]

# Resilience stack
resilience_stack = [
    RetryMiddleware(max_retries=3),
    TimeoutMiddleware(timeout_seconds=30),
    FallbackMiddleware(fallback_model="openai:gpt-4o-mini"),
]

# Combine both
agent = create_agent(
    model="openai:gpt-5.5",
    tools=[...],
    middleware=security_stack + resilience_stack,
)
```

### Dynamic Tool Selection by Domain

```python
class DomainToolSelectionMiddleware(AgentMiddleware):
    """Select tools based on detected domain."""
    
    def wrap_model_call(self, request, handler):
        # Analyze messages to detect domain
        domain = detect_domain(request.messages)
        
        # Filter tools to domain-specific subset
        domain_tools = {
            "finance": [calculate_roi, analyze_portfolio],
            "research": [web_search, cite_paper],
            "general": request.tools,
        }
        
        selected = domain_tools.get(domain, request.tools)
        request = request.override(tools=selected)
        
        return handler(request)
```

## Summary

Agent composability through middleware and subagent delegation enables:

- **Reusable behaviors** via middleware stacks (logging, caching, validation, retry)
- **Hierarchical execution** via subagent delegation with causation tracking
- **Flexible tool management** through dynamic selection and authorization
- **Clean separation of concerns** without monolithic agent implementations
- **Predictable composition** via closure-based handler chaining and automatic state merging

These patterns form the foundation for building production agents that are maintainable, testable, and extensible across diverse use cases.
