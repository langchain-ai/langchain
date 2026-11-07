---
type: Reference Documentation
title: Agent Execution - Reference Guide
description: Technical reference for agent execution architecture, including state schemas, control flow logic, middleware composition, and the conditional edge system that governs tool dispatch and loop termination.
tags: [agent-execution, reference, control-flow, state-machine, middleware, langchain, langgraph]
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

This reference documents the technical architecture of agent execution in LangChain. Agents are orchestrated by a LangGraph `StateGraph` that repeats a cycle of model invocation, tool execution, and loop-control decisions until a termination condition is met. The system is extensible via middleware hooks that run at specific lifecycle points, enabling caching, retry logic, dynamic tool injection, and other cross-cutting concerns.

For a detailed walkthrough of the execution flow, see the [related guide](/openwiki/agent-execution.md).

## State Architecture

### AgentState TypedDict

The agent's internal state is a `TypedDict` with the following schema:

```python
class AgentState(TypedDict, Generic[ResponseT]):
    """State schema for the agent."""
    
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]
```

**Field Descriptions:**

| Field | Type | Purpose |
|-------|------|---------|
| `messages` | `Required[list[AnyMessage]]` with `add_messages` reducer | Accumulates all messages (user, AI, tool) in the conversation. The reducer ensures messages are appended rather than replaced. |
| `jump_to` | `NotRequired[JumpTo \| None]` | Ephemeral middleware control field. Set by `before_model()` or `after_model()` hooks to override loop control: `'model'`, `'tools'`, or `'end'`. Not persisted across graph invocations. |
| `structured_response` | `NotRequired[ResponseT]` | When `response_format` is configured on the agent, holds the parsed structured output from the last model invocation. Cleared explicitly at iteration start if no new structured response is produced. |

### Input and Output Schemas

**InputAgentState** (what the user provides):
```python
class InputAgentState(TypedDict):
    messages: Required[Annotated[list[AnyMessage | dict[str, Any]], add_messages]]
```

**OutputAgentState** (what is returned):
```python
class OutputAgentState(TypedDict, Generic[ResponseT]):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    structured_response: NotRequired[ResponseT]
```

The input schema accepts both `AnyMessage` objects and plain dicts (for backward compatibility). The output schema returns fully resolved `AnyMessage` objects and includes the optional `structured_response`.

## Message Types and Encoding

Messages in the agent system are `langchain_core.messages` objects:

- **`UserMessage`** – User input or query. Often passed in via `InputAgentState.messages`.
- **`SystemMessage`** – System prompt or instructions, prepended to the message list at model call time.
- **`AIMessage`** – Output from the language model. May contain a `tool_calls` list (dicts with `id`, `name`, and `args`).
- **`ToolMessage`** – Result of tool execution. Carries `tool_call_id` (links to the originating `AIMessage.tool_calls[i]["id"]`), `name`, and `content`.

## Model Request and Response Objects

### ModelRequest

Before invoking the model, the agent constructs a `ModelRequest` dataclass that encapsulates all inputs and is passed to middleware:

```python
@dataclass(init=False)
class ModelRequest(Generic[ContextT]):
    model: BaseChatModel
    messages: list[AnyMessage]  # excluding system message
    system_message: SystemMessage | None
    tool_choice: Any | None
    tools: list[BaseTool | dict[str, Any]]
    response_format: ResponseFormat[Any] | None
    state: AgentState[Any]
    runtime: Runtime[ContextT]
    model_settings: dict[str, Any] = field(default_factory=dict)
```

**Key Attributes:**

- `messages` – Conversation history, excluding the system message (which is added at call time).
- `system_message` – System prompt instance.
- `tools` – List of tools available to the model.
- `response_format` – Optional structured output configuration (e.g., `ToolStrategy`, `ProviderStrategy`, or raw schema).
- `state` – The current agent state (for middleware inspection or modification).
- `runtime` – LangGraph runtime context (provides thread info, store, etc.).
- `model_settings` – Additional kwargs passed to `model.bind_tools()` or `model.bind()`.

**Override Pattern:**

`ModelRequest` is immutable. Middleware can create a modified request via the `.override()` method:

```python
new_request = request.override(
    model=different_model,
    system_message=SystemMessage(content="New instructions"),
    tools=[...]
)
```

### ModelResponse

The core execution returns a `ModelResponse`:

```python
@dataclass
class ModelResponse(Generic[ResponseT]):
    result: list[BaseMessage]
    structured_response: ResponseT | None = None
```

- `result` – Usually a single `AIMessage`, but may include additional `ToolMessage` objects if structured output tools were invoked.
- `structured_response` – Parsed schema when `response_format` is configured.

### ExtendedModelResponse

Middleware's `wrap_model_call` can return an `ExtendedModelResponse` to attach an optional `Command` for state updates:

```python
@dataclass
class ExtendedModelResponse(Generic[ResponseT]):
    model_response: ModelResponse[ResponseT]
    command: Command[Any] | None = None
```

Commands are applied via the graph's reducers, so messages in commands are **added alongside** (not replacing) the model response messages. Commands from multiple middleware layers are accumulated (inner-first ordering) without merging.

## Middleware Architecture

### Hook Lifecycle

Middleware runs at five lifecycle points during agent execution:

1. **`before_agent(state, runtime) -> dict | None`**
   - Runs once at the very start, before model initialization.
   - Useful for setup, initial state configuration, or load-time validation.
   - Both sync (`before_agent`) and async (`abefore_agent`) variants are supported.

2. **`before_model(state, runtime) -> dict | None`**
   - Runs before each model invocation, including after tool execution in loop iterations.
   - Middleware can modify state or use `@hook_config(can_jump_to=[...])` to jump to `'tools'`, `'model'`, or `'end'`.
   - Returns `dict` of state updates or `None`.

3. **`wrap_model_call(request, handler) -> ModelResponse | AIMessage | ExtendedModelResponse`**
   - Intercepts the actual model call (synchronous version; `awrap_model_call` for async).
   - Receives a `handler` callback. Middleware can:
     - Call `handler(request)` to execute the model normally.
     - Call it multiple times for retry logic.
     - Skip calling it for short-circuit caching.
     - Modify the request before calling (e.g., narrow tools, change model).
   - Returns normalized response type; `AIMessage` is converted to `ModelResponse` automatically.

4. **`after_model(state, runtime) -> dict | None`**
   - Runs after the model returns and messages are added to state.
   - Middleware can inject synthetic `ToolMessage` objects, modify state, or jump.
   - Runs before conditional edge decisions (so injected messages influence loop control).

5. **`after_agent(state, runtime) -> dict | None`**
   - Runs once at the very end, after the loop exits.
   - Useful for final cleanup, summarization, or post-processing before returning to the user.
   - Both sync and async variants supported.

### Middleware Composition Rules

**Sequential Hooks (`before_agent`, `before_model`, `after_model`, `after_agent`):**

Multiple middleware instances are chained in registration order. State updates flow forward through the chain. Each middleware receives the state as updated by earlier middleware in the sequence.

**Wrapped Model Call (`wrap_model_call`):**

Handlers compose as nested functions, with the **first middleware in the list becoming the outermost layer**:

```
First (outer)
  → Second
    → Third (inner)
      → _execute_model_sync()
```

Commands from each layer are accumulated (inner-first ordering) without merging. Outermost middleware can see all inner commands.

**Tool Call Wrappers (`wrap_tool_call`):**

Similar to `wrap_model_call`: first middleware is outermost. Middleware can retry, short-circuit, or modify tool execution.

### Middleware Tools and Jump Control

**Tool Registration:**

Each middleware instance can declare additional tools via the `tools` class attribute:

```python
class MyMiddleware(AgentMiddleware):
    tools = [my_tool_1, my_tool_2]
```

These tools are merged into the `ToolNode` and are available for the model to call. If middleware adds tools dynamically in `wrap_model_call`, it should implement `wrap_tool_call` to handle their execution.

**Jump Control:**

Middleware's `before_model()` or `after_model()` methods can use the `@hook_config(can_jump_to=[...])` decorator to declare which destinations they may jump to. For example:

```python
from langchain.agents.middleware.types import hook_config

@hook_config(can_jump_to=['tools', 'model', 'end'])
def before_model(self, state, runtime):
    if should_skip:
        return {"jump_to": "end"}  # Jump directly to exit
    return None
```

Valid jump targets are:
- `'model'` – Jump to the model node (or the next middleware in `before_model` chain).
- `'tools'` – Jump to the tool execution node (only if tools exist).
- `'end'` – Exit the loop and end the agent.

If a middleware method sets `state['jump_to']`, the graph will honor it, overriding the default loop logic.

## Graph Structure and Nodes

The agent is compiled as a `StateGraph` with the following nodes:

| Node | Purpose | Created When |
|------|---------|--------------|
| `START` | Entry point | Always |
| Middleware nodes (prefixed with middleware name) | `before_agent`, `before_model`, `after_model`, `after_agent` hooks | Middleware implements the hook |
| `model` | Model invocation with optional middleware wrapping | Always |
| `tools` | Tool execution | Tools are provided to `create_agent()` |
| `END` | Exit point | Always |

### Edge Configuration

**Entry:** `START` → (first before_agent or before_model, or model)

**Loop Entry:** After tool execution or synthetic message injection, graph returns to `before_model` (if exists) or `model`.

**Loop Exit:** When termination condition is met (no more tool calls, structured output ready, return_direct tool executed, explicit jump to end).

**Middleware Edges:**

- `before_agent` middleware are chained sequentially, each defaulting to the next. The last connects to `before_model` or `model`.
- `before_model` middleware are chained, each defaulting to the next. The last connects to `model`.
- After `model` executes, if `after_model` middleware exist, it routes to the first one (in reverse order of registration). They chain backward to the conditional edges.
- `after_agent` middleware run once at the very end, chained in reverse order, before `END`.

## Control Flow and Conditional Edges

Loop control is governed by conditional edges that inspect the last `AIMessage`, tool execution results, and the `jump_to` state field.

### Model-to-Tools Decision (`_make_model_to_tools_edge`)

After the model node (and optional `after_model` middleware), the graph decides whether to dispatch tools or exit:

1. **Explicit Jump:** If `state['jump_to']` is set by middleware, use that destination.

2. **No AIMessage:** If the message list is empty or corrupted (no last AIMessage), exit to `end_destination`.

3. **No Tool Calls:** If the last `AIMessage.tool_calls` is empty, the model chose not to use tools. Exit the loop.

4. **Pending Tool Calls:** Filter out:
   - Tool calls that have already been executed (matched by `tool_call_id` in existing `ToolMessage` objects).
   - Structured output tool calls (whose names are in `structured_output_tools`).
   
   If pending calls remain, dispatch them as a list of `Send` commands to the `tools` node.

5. **All Calls Already Executed or Are Structured:** Jump back to the model node to process injected `ToolMessage` results.

6. **Structured Response Ready:** If `state['structured_response']` is now populated (a structured output tool was executed), exit.

### Tools-to-Model Decision (`_make_tools_to_model_edge`)

After tool execution, the graph decides whether to loop back to the model or exit:

1. **No AIMessage:** If no AIMessage exists, return to `model_destination` (to regenerate).

2. **return_direct Tools:** If all executed tools have `return_direct=True`, exit to `end_destination`.

3. **Structured Output Tool Executed:** If a `ToolMessage` is for a structured output tool and has no error status, exit.

4. **Default:** Continue the loop, returning to `model_destination` (model processes results and decides next step).

### Model-to-Model Decision (Structured Output Only)

When the agent has structured output tools but no regular tools, a `model_to_model` edge handles retries:

1. **Explicit Jump:** If `state['jump_to']` is set, use that.

2. **Structured Response Ready:** If `state['structured_response']` is populated, exit.

3. **Default:** Continue to `model_destination` (retry structured output generation).

## Structured Output Processing

When `response_format` is configured on the agent, the system handles structured output in two strategies:

### Tool Strategy (`ToolStrategy`)

Uses a dedicated tool to capture structured output. The model is forced to call the tool via `tool_choice="any"`.

**Processing:**
1. Model is invoked with structured output tools bound (along with regular tools).
2. If the model calls a structured output tool, the tool call arguments are parsed against the schema.
3. A `ToolMessage` is created with the tool call result (or error if validation fails).
4. `state['structured_response']` is set to the parsed value.
5. Loop exits (see `tools_to_model` exit condition #3).

**Error Handling:**

If parsing fails and `ToolStrategy.handle_errors` is configured:
- `True` or unspecified: Insert a `ToolMessage` with a standardized error prompt; retry.
- Custom string: Insert custom error message; retry.
- `False`: Raise the validation error immediately; do not retry.
- Exception class or tuple: Raise if exception matches; retry otherwise.

### Provider Strategy (`ProviderStrategy`)

Uses the model's native structured output support (e.g., OpenAI's `response_format`).

**Processing:**
1. Model is invoked with `response_format` kwargs from `ProviderStrategy.to_model_kwargs()`.
2. If successful, model output is parsed to extract the structured response.
3. `state['structured_response']` is set to the parsed value.
4. No tool calls are made; loop exits normally.

**Auto-Detection:**

When `response_format` is a raw schema (dict or Pydantic class), it's wrapped in `AutoStrategy`. At model invocation time:
- If the model supports provider strategy (determined by model name or profile), use `ProviderStrategy`.
- Otherwise, use `ToolStrategy`.

## Termination Conditions

An agent loop exits when any of the following occurs:

1. **Model declines tools:** Last `AIMessage.tool_calls` is empty.
2. **No pending tool calls:** All tool calls in the last `AIMessage` have been executed or are structured output tools.
3. **Structured response ready:** `state['structured_response']` is populated.
4. **return_direct tool executed:** A tool with `return_direct=True` was called and completed.
5. **Middleware jump to 'end':** A middleware hook set `state['jump_to'] = 'end'`.
6. **No AIMessage:** The message list is corrupted or empty.

After any exit condition, middleware's `after_agent()` hooks run (if defined), then the graph terminates and returns to the user.

## Error Handling

### Model Invocation Errors

Exceptions during model invocation are propagated by default. Middleware's `wrap_model_call` can catch and handle them:

```python
def wrap_model_call(self, request, handler):
    try:
        return handler(request)
    except RateLimitError:
        # Retry or fallback
        return ModelResponse(result=[AIMessage(content="Service overloaded")])
```

### Invalid Tool Call Messages

If a model generates a tool call with malformed arguments that cannot be parsed, the agent creates a `ToolMessage` with status `"error"` instead of failing. This allows the loop to continue and the model to correct the issue.

### Dynamic Tool Validation

If middleware modifies `request.tools` in `wrap_model_call` to add tools that weren't declared upfront, the agent validates this only if no `wrap_tool_call` middleware is registered. If validation fails, a `ValueError` is raised with the list of unknown tools.

### Structured Output Errors

When parsing structured output fails (either tool or provider strategy), the error is either retried (with error feedback) or raised, depending on `ToolStrategy.handle_errors` configuration.

## Recursion and Limits

The compiled graph sets `recursion_limit` to `9_999` to allow for long-running agent loops. This can be overridden at invocation time via `RunnableConfig`.

## Important Implementation Details

### Message Reducer and Accumulation

The `messages` field uses the `add_messages` reducer from LangGraph. This ensures that:
- New messages are appended, not replacing the entire list.
- Middleware and model responses can contribute messages independently.
- Commands from `wrap_model_call` handlers are merged correctly.

### Repaired Message Tracking

When invalid tool calls are detected, they are patched with error `ToolMessage` objects. The repaired message list is passed through the graph, ensuring consistency.

### Tool Parallel Execution

Pending tool calls are dispatched as separate `Send` commands to the `tools` node, allowing LangGraph to execute them in parallel or sequentially depending on runtime configuration.

### Jump Field Reset

The `jump_to` field is ephemeral and automatically cleared each iteration (or between invocations) because it's marked with `EphemeralValue` in the state schema.

## References

- **Seed Source:** `repo://libs/langchain_v1/langchain/agents/factory.py` – Contains `create_agent()`, graph construction, conditional edge logic, and middleware composition.
- **Middleware Types:** `repo://libs/langchain_v1/langchain/agents/middleware/types.py` – Defines `AgentMiddleware`, `AgentState`, `ModelRequest`, `ModelResponse`, and all hook signatures.
- **Related Documentation:** See [Agent Execution Flow and Loop Control](/openwiki/agent-execution.md) for a detailed walkthrough with sequence diagrams.
