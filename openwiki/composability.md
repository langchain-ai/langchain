---
type: "Concept"
title: "Agent Composability: Building Complex Agents"
description: "Patterns for composing agents through middleware layers, subagent delegation, tool selection strategies, and context management across hierarchical agent workflows."
tags: ["composability", "agents", "middleware", "subagents", "tool-delegation", "context-management"]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-20T08:25:57.412Z
sources:
  - id: openwiki-source-2bb5c49861e72a9099b51970
    resource: repo://libs/langchain_v1/langchain/agents/_subagent_transformer.py
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
  - id: openwiki-source-737abffff349404aff61cdc1
    resource: repo://libs/langchain_v1/langchain/agents/middleware/provider_tool_search.py
  - id: openwiki-source-a5eb2874945fdc55df0c578f
    resource: repo://libs/langchain_v1/langchain/agents/middleware/tool_selection.py
  - id: openwiki-source-03e8ca0eebe37feda8566793
    resource: repo://libs/langchain_v1/langchain/agents/middleware/types.py
generated: { by: "openwiki/0.5.0", at: "2026-10-20T08:25:57.412Z" }
---

## Overview

Agent composability in LangChain enables building complex, hierarchical agent systems through three primary mechanisms: **middleware composition** (layered request/response interception), **subagent delegation** (tool-based agent dispatch), and **tool selection strategies** (filtering and routing). These patterns allow multiple agents to coordinate, share context, and delegate specialized tasks while maintaining observability and fault isolation.

The composability model is fundamentally built on **middleware layers** that wrap and transform agent execution at multiple hooks (before/after agent, before/after model, around tool execution), combined with **parallel/sequential tool execution** via LangGraph's `Send` command and **state schema composition** that merges input from multiple agents into unified workflows.

## Middleware Composition: Layering Request/Response Handlers

Middleware forms the backbone of agent composability. Each middleware instance registers hooks that intercept execution at well-defined points, enabling cross-cutting concerns without modifying core agent logic.

### Middleware Stack Architecture

When multiple middleware are registered, they compose into a stack where **first in the list becomes the outermost layer**:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

class LoggingMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        print(f"Calling model with {len(request.tools)} tools")
        response = handler(request)
        print(f"Model returned {len(response.result)} messages")
        return response

class RetryMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        for attempt in range(3):
            try:
                return handler(request)
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"Retry attempt {attempt + 1}")

# Middleware order: LoggingMiddleware wraps RetryMiddleware wraps the model
agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[
        LoggingMiddleware(),
        RetryMiddleware(),
    ]
)
```

**Control flow:** LoggingMiddleware.wrap_model_call → RetryMiddleware.wrap_model_call → actual model execution

Middleware receive a `handler` callback (the next layer) and must invoke it to continue execution. Each middleware can:
- **Inspect** the request or response without modification
- **Rewrite** the request (e.g., modify tools, system message, state) before calling handler
- **Short-circuit** execution (skip handler entirely to return early)
- **Retry** by calling handler multiple times
- **Return commands** from `wrap_model_call` or `wrap_tool_call` to control graph flow

### Lifecycle Hooks

Each agent execution triggers multiple hooks in sequence:

1. **`before_agent(state, runtime)`** – Runs once at start, before the first model call
2. **`before_model(state, runtime)`** – Runs before each model invocation; can update state or jump to different nodes
3. **`wrap_model_call(request, handler)`** – Intercepts the actual model call; middleware chains are composed here
4. **`after_model(state, runtime)`** – Runs after model response; can filter tool calls or update state
5. **`wrap_tool_call(request, handler)`** – Intercepts each tool execution
6. **`after_agent(state, runtime)`** – Runs once at end, after loop terminates

### State Flow Through Middleware

All hooks receive `state: AgentState[ResponseT]` and `runtime: Runtime[ContextT]`. The `AgentState` is a TypedDict containing:

```python
class AgentState(TypedDict):
    messages: Required[list[AnyMessage]]      # Conversation history
    jump_to: NotRequired[JumpTo | None]       # "model" | "tools" | "end" (ephemeral)
    structured_response: NotRequired[Any]     # Parsed structured output (if configured)
```

Hooks return `dict[str, Any] | None` for state updates. Updates are merged into the state using LangGraph reducers (e.g., `add_messages` for the `messages` field). The `jump_to` field allows middleware to override default routing—set to `"tools"` to skip the model, `"model"` to loop back, or `"end"` to exit early.

### Model Call Interception

The `wrap_model_call` hook provides deepest control over model behavior. The middleware receives a `ModelRequest` containing:

- **`model`**: The LLM instance
- **`messages`**: List of conversation messages (excluding system prompt)
- **`system_message`**: SystemMessage instance (or None)
- **`tools`**: List of available tools
- **`tool_choice`**: Tool selection preference (e.g., `required`, `any`, specific tool name)
- **`response_format`**: Structured output specification (if enabled)
- **`state`**: Current agent state dictionary
- **`runtime`**: Runtime context with access to checkpointer, store, etc.

Middleware can call `request.override(**updates)` to create a modified request:

```python
class ToolFilteringMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        # Keep only tools whose names start with 'safe_'
        filtered_tools = [
            t for t in request.tools
            if t.name.startswith('safe_')
        ]
        modified_request = request.override(tools=filtered_tools)
        return handler(modified_request)
```

The handler returns a `ModelResponse` (or `AIMessage` or `ExtendedModelResponse` for command support), which middleware can inspect, modify, or replace before returning to the next layer.

## Parallel and Sequential Tool Execution: Using `Send`

The agent's tool dispatch layer supports both parallel and sequential execution patterns through LangGraph's `Send` command. When a model requests multiple tools, the graph can dispatch them in parallel, each receiving its own isolated tool call.

### Parallel Tool Dispatch via `Send`

The default tool routing dispatches each tool call independently:

```python
# In the model → tools edge decision logic:
if pending_tool_calls:
    # Each tool call gets a separate Send to the tools node
    return [Send("tools", [tool_call]) for tool_call in pending_tool_calls]
```

Each `Send` targets the tools node with a single `[tool_call]` in a list. The tools node executes it concurrently with other Sends. This enables true parallel execution:

- Tool 1 and Tool 2 run concurrently if the model called both
- Results arrive and are added to state in any order
- The graph reconvenes after all Send tasks complete
- The model is invoked again to process all results together

### Implementation Details

The tools node receives each tool call via `Send` and:
1. Looks up the tool by name
2. Invokes `wrap_tool_call` middleware for interception
3. Executes the tool with arguments
4. Returns a `ToolMessage` with the result

Middleware can override tool execution via `wrap_tool_call`:

```python
class ToolValidationMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request, execute):
        # Validate arguments before execution
        if not self.is_valid(request.tool_call):
            return ToolMessage(
                content="Invalid arguments",
                tool_call_id=request.tool_call["id"],
                status="error",
            )
        return execute(request)
```

### Sequential Tool Handling

For tools that must run in a specific order, use middleware to redirect:

```python
class SequentialToolMiddleware(AgentMiddleware):
    def after_model(self, state, runtime):
        # Inspect tool calls and reorder them
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage):
            # Reorder tool calls as needed
            reordered = self.reorder_tool_calls(last_message.tool_calls)
            # Update the message to reflect new order
            last_message.tool_calls[:] = reordered
        return None
```

## Subagent Composition: Tool-Based Agent Delegation

Subagents enable hierarchical task delegation. A parent agent invokes a tool that internally creates and runs a child agent, receiving the result back as a tool message.

### SubagentTransformer: Detecting Subagent Boundaries

The `SubagentTransformer` automatically detects when a nested agent run boundary occurs and surfaces it via `run.subagents`. A nested run is considered a subagent when:

1. It is created via `create_agent(name=...)` with an explicit name
2. Its `lc_agent_name` differs from its parent's name
3. It was triggered by a tool call (joined via `cause` metadata)

```python
from langchain.agents import SubagentTransformer

# The transformer is automatically added by create_agent's default transformer list
transformer = SubagentTransformer(scope=())

# When consuming a run stream with this transformer:
# run.subagents yields SubagentRunStream or AsyncSubagentRunStream handles
```

Each subagent handle provides:
- **`name`**: The subagent's configured name
- **`cause`**: Metadata about the triggering tool call (LifecycleCause dict with `type: "toolCall"` and `tool_call_id`)
- **`status`**: Final status ("success", "failed", or running)
- **`error`**: Optional error message if failed

### Creating Delegating Tools

A tool can create and invoke a subagent:

```python
from langchain.agents import create_agent

def weather_lookup_tool(location: str, *, runtime) -> str:
    """Look up weather by delegating to a weather specialist agent."""
    # Create a child agent with specialized tools
    weather_agent = create_agent(
        model="openai:gpt-4o",
        name="weather_specialist",  # Important: sets lc_agent_name
        tools=[get_forecast, get_historical_data],
        system_prompt="You are a weather expert.",
    )
    
    # Invoke the subagent
    result = weather_agent.invoke({
        "messages": [{"role": "user", "content": f"Weather for {location}?"}]
    })
    
    # Extract the final response
    last_message = result["messages"][-1]
    return last_message.content if hasattr(last_message, 'content') else str(result)
```

The `runtime` parameter is injected by the middleware framework and provides access to:
- **`state`**: Current agent state (via CONFIG_KEY_READ at execution time)
- **`checkpointer`**: Checkpoint store for resuming interrupted runs
- **`store`**: LangGraph persistent memory store
- **`context`**: User-provided runtime context (type `ContextT`)

### State Merging Across Subagents

When a subagent completes, its state is not automatically merged with the parent. The delegating tool must extract relevant outputs:

```python
def research_agent_tool(topic: str, *, runtime) -> str:
    """Delegate research to a specialized agent, return its findings."""
    research_agent = create_agent(
        model="openai:gpt-4o",
        name="researcher",
        tools=[search_web, read_article],
    )
    
    result = research_agent.invoke({
        "messages": [{"role": "user", "content": f"Research {topic}"}]
    })
    
    # Return structured findings for the parent agent to use
    findings = {
        "topic": topic,
        "summary": result["messages"][-1].content,
        "sources": extract_sources(result["messages"]),
    }
    return json.dumps(findings)
```

The parent agent receives this result as a `ToolMessage` and can use it for further reasoning.

## Tool Selection Strategies

When an agent has many tools, filtering to only relevant ones improves efficiency and focus. LangChain provides multiple strategies:

### 1. LLM-Based Tool Selection

The `LLMToolSelectorMiddleware` uses a small language model to select the most relevant tools before calling the main model:

```python
from langchain.agents.middleware import LLMToolSelectorMiddleware

selector = LLMToolSelectorMiddleware(
    model="openai:gpt-5.5-mini",  # Fast, cheaper model for selection
    max_tools=5,                    # Limit to 5 most relevant tools
)

agent = create_agent(
    model="openai:gpt-4o",
    tools=[tool1, tool2, tool3, ...tool50],  # Many tools
    middleware=[selector],
)
```

How it works:
1. Before the main model call, the selector model receives the user's query and full tool list
2. The selector model returns a structured list of tool names ranked by relevance
3. Only the selected tools are passed to the main model
4. The main model works with a reduced set and focuses better

Configuration options:
- **`max_tools`**: Maximum number of tools to select (default: 10)
- **`model`**: Which model to use for selection (default: inferred from main model)
- **`on_parsing_failure`**: Fallback if the selector fails (`"error"`, `"none"`, `"all"`, or a list of tool names)

### 2. Provider-Native Tool Search

The `ProviderToolSearchMiddleware` defers tool schema loading to the provider (Anthropic, OpenAI). The provider includes a special "tool search" tool that lets the model retrieve full tool schemas only when needed:

```python
from langchain.agents.middleware import ProviderToolSearchMiddleware

agent = create_agent(
    model="anthropic:claude-opus-4-8",
    tools=[tool1, tool2, ...tool100],  # Many tools
    middleware=[
        ProviderToolSearchMiddleware(searchable_tools=[
            "lookup_order",
            "check_inventory",
            "get_shipping_info",
        ])
    ],
)
```

Benefits:
- Initial request payload is small (only tool names, not schemas)
- Model can use the `tool_search_tool` to search for relevant tools dynamically
- Provider retrieves matching tool schemas on-demand
- Scales well with hundreds of tools

Only Anthropic and OpenAI support provider-native tool search. The middleware validates this at runtime.

### 3. Exact Tool Matching

For deterministic tool selection, middleware can implement exact matching on known conditions:

```python
class ContextualToolMiddleware(AgentMiddleware):
    """Route to different tools based on context."""
    
    def wrap_model_call(self, request, handler):
        # Check if this is a file operation
        if any("file" in msg.content.lower() for msg in request.messages):
            # Keep only file tools
            file_tools = [t for t in request.tools if "file" in t.name.lower()]
            request = request.override(tools=file_tools)
        
        return handler(request)
```

This approach is fast, deterministic, and suitable for small tool sets.

## Context Passing: Parent to Subagent

When invoking subagents, the parent can pass context that shapes the subagent's behavior:

### Context Via System Prompt

```python
def delegating_tool(query: str, *, runtime) -> str:
    # Parent context: user's stated preferences
    user_preferences = runtime.state.get("user_preferences", {})
    
    subagent = create_agent(
        model="openai:gpt-4o",
        name="child_agent",
        tools=[...],
        system_prompt=(
            f"You are helping a user with preferences: {user_preferences}. "
            f"Original request: {query}"
        ),
    )
    
    result = subagent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content
```

### Context Via State

For more structured context, pass data in the initial messages:

```python
def delegating_tool(task: str, *, runtime) -> str:
    # Extract context from parent state
    conversation_history = runtime.state.get("messages", [])
    
    subagent = create_agent(
        model="openai:gpt-4o",
        name="specialized_agent",
        tools=[...],
    )
    
    # Seed the subagent with parent's conversation context
    result = subagent.invoke({
        "messages": [
            *conversation_history[-3:],  # Last 3 parent messages for context
            {"role": "user", "content": task},
        ]
    })
    
    return extract_result(result)
```

### Context Via Runtime Store

For stateful coordination across many subagents, use the LangGraph store:

```python
def delegating_tool(operation: str, *, runtime) -> str:
    # Read shared state from the store
    session_id = runtime.context.get("session_id")
    user_data = runtime.store.get(f"session:{session_id}", ["user_profile"])[0]
    
    subagent = create_agent(
        model="openai:gpt-4o",
        name="worker_agent",
        tools=[...],
    )
    
    # Subagent can also write to the store
    result = subagent.invoke({
        "messages": [{"role": "user", "content": operation}]
    })
    
    # Store the result for other subagents or parent to read
    runtime.store.put(f"session:{session_id}", [f"result:{operation}"], json.dumps(result))
    
    return "Operation completed"
```

## State Schema Composition

When middleware adds custom state fields or subagents return structured state, those schemas must be merged into the agent's overall state.

### Declaring Custom State in Middleware

Middleware can extend `AgentState` with additional fields:

```python
from typing_extensions import TypedDict
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    OmitFromInput,
)

class CustomState(AgentState):
    """Extended state with middleware-specific fields."""
    
    analysis: dict  # Added by analysis middleware, omitted from input
    retry_count: int  # Counts retries, omitted from both input and output


class AnalysisMiddleware(AgentMiddleware):
    state_schema = CustomState
    
    def after_model(self, state: CustomState, runtime):
        # Populate analysis field
        return {
            "analysis": {
                "tool_count": len(state.get("messages", [])),
                "complexity": self.rate_complexity(state),
            }
        }
```

When `create_agent` is called with this middleware, it automatically merges `CustomState` into the final state schema.

### Schema Merging Rules

When multiple middleware declare state extensions:
1. Schemas are merged left-to-right (first middleware's fields override later ones for conflicts)
2. Fields marked with `OmitFromSchema`, `OmitFromInput`, `OmitFromOutput` are excluded from the respective schemas
3. Reducer annotations (e.g., `add_messages`) are preserved
4. The final schema is published as the agent's `input_schema` and `output_schema`

## Failure Handling in Subagent Scenarios

Subagent failures need careful handling to avoid cascading errors.

### Tool-Level Failure Handling

The delegating tool can catch and report subagent failures:

```python
def resilient_delegation_tool(query: str, *, runtime) -> str:
    """Delegate work with graceful fallback."""
    try:
        subagent = create_agent(
            model="openai:gpt-4o",
            name="worker",
            tools=[...],
        )
        result = subagent.invoke({"messages": [{"role": "user", "content": query}]})
        return result["messages"][-1].content
    except Exception as e:
        # Return a fallback message instead of raising
        logger.error(f"Subagent failed: {e}")
        return f"I couldn't complete that task. Please try again or rephrase your request."
```

### Middleware-Level Failure Handling

Use `wrap_tool_call` to intercept and handle failures:

```python
class ToolErrorRecoveryMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request, execute):
        try:
            return execute(request)
        except ValueError as e:
            # Return an error message instead of propagating
            return ToolMessage(
                content=f"Tool execution failed: {str(e)}. Please try with different parameters.",
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
                status="error",
            )
```

### Timeout and Resource Limits

For long-running subagents, use timeouts:

```python
import signal
import contextlib

def delegating_tool_with_timeout(query: str, *, runtime, timeout_seconds=30) -> str:
    """Delegate with timeout."""
    subagent = create_agent(
        model="openai:gpt-4o",
        name="worker",
        tools=[...],
    )
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Subagent exceeded time limit")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        result = subagent.invoke({"messages": [{"role": "user", "content": query}]})
        return result["messages"][-1].content
    except TimeoutError:
        return "The operation took too long. Try a simpler request."
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
```

## End-to-End Example: Multi-Tier Agent Hierarchy

Here's a complete example showing a 3-tier agent system:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolSelectorMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# Tier 3: Specialist agents for specific domains

@tool
def get_weather(location: str) -> str:
    """Get current weather."""
    return f"Weather in {location}: Sunny, 72°F"

weather_agent = create_agent(
    model="openai:gpt-4o-mini",
    name="weather_specialist",
    tools=[get_weather],
    system_prompt="You are a weather expert.",
)

@tool
def search_flights(destination: str) -> str:
    """Search available flights."""
    return f"Flights to {destination}: AA100, UA201, DL303"

travel_agent = create_agent(
    model="openai:gpt-4o-mini",
    name="travel_specialist",
    tools=[search_flights],
    system_prompt="You are a travel booking specialist.",
)

# Tier 2: Coordinator delegates to specialists

def weather_tool(location: str, *, runtime) -> str:
    """Get weather by delegating to weather specialist."""
    result = weather_agent.invoke({
        "messages": [{"role": "user", "content": f"Weather for {location}"}]
    })
    return result["messages"][-1].content

def travel_tool(destination: str, *, runtime) -> str:
    """Find flights by delegating to travel specialist."""
    result = travel_agent.invoke({
        "messages": [{"role": "user", "content": f"Find flights to {destination}"}]
    })
    return result["messages"][-1].content

coordinator_agent = create_agent(
    model="openai:gpt-4o",
    name="trip_coordinator",
    tools=[weather_tool, travel_tool],
    system_prompt="You help users plan trips by checking weather and finding flights.",
    middleware=[LLMToolSelectorMiddleware(max_tools=2)],
)

# Tier 1: Main entry point

def orchestrator_tool(planning_request: str, *, runtime) -> str:
    """Plan a trip by delegating to coordinator."""
    result = coordinator_agent.invoke({
        "messages": [{"role": "user", "content": planning_request}]
    })
    return result["messages"][-1].content

main_agent = create_agent(
    model="openai:gpt-4o",
    name="travel_assistant",
    tools=[orchestrator_tool],
    system_prompt="You are a helpful travel assistant.",
)

# Use the system
output = main_agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Plan a trip to Paris next week"
    }]
})

print(output["messages"][-1].content)
```

**Execution flow:**
1. `main_agent` receives the trip planning request
2. It invokes `orchestrator_tool`, which delegates to `coordinator_agent`
3. `coordinator_agent` with `LLMToolSelectorMiddleware` filters to relevant tools
4. Based on the query, it calls `weather_tool` and `travel_tool` (possibly in parallel via Send)
5. `weather_tool` invokes `weather_agent` (tier 3)
6. `travel_tool` invokes `travel_agent` (tier 3)
7. Results flow back up through the hierarchy
8. `main_agent` receives the final coordinated response

This hierarchy demonstrates:
- **Middleware composition** (selector filters tools at tier 2)
- **Subagent delegation** (tools create and invoke nested agents)
- **Parallel execution** (weather and travel agents run concurrently if both are needed)
- **State isolation** (each agent has its own state; results are manually extracted)
- **Observability** (each tier is traced independently via `SubagentTransformer`)

## Summary

Agent composability provides powerful patterns for building sophisticated multi-agent systems:

| Pattern | Use Case | Example |
|---------|----------|---------|
| **Middleware layers** | Cross-cutting concerns (retry, logging, validation) | Add `RetryMiddleware` to auto-retry failed model calls |
| **Subagent delegation** | Specialize work to child agents with focused tools | Weather expert agent invoked by travel coordinator |
| **Tool selection** | Filter tools in large sets for efficiency | `LLMToolSelectorMiddleware` for 100+ tools |
| **Parallel tool execution** | Run independent tools concurrently | Model calls 3 tools → 3 Sends dispatched in parallel |
| **Context passing** | Share parent state with subagents | System prompt includes user preferences from parent |
| **State composition** | Extend state with middleware-specific fields | Custom state schema with analysis results |

By combining these patterns, you can build agents that coordinate across multiple specialized domains, recover gracefully from failures, optimize token usage, and remain fully observable.
