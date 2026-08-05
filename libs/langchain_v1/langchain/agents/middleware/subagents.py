"""Middleware for providing subagents to an agent via a `task` tool."""

import contextlib
import dataclasses
import json
from collections.abc import Awaitable, Callable, Generator, Sequence
from typing import Annotated, Any, TypedDict, cast, get_args, get_origin, get_type_hints

from typing_extensions import NotRequired

from langchain.agents import create_agent
from langchain.agents.middleware._utils import append_to_system_message
from langchain.agents.middleware.shell_tool import DEFAULT_TOOL_DESCRIPTION
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    ResponseT,
    StateT,
)
from langchain.agents.structured_output import ResponseFormat
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from langsmith.run_helpers import get_tracing_context, tracing_context
from pydantic import BaseModel, Field

from langchain.tools.tool_node import ToolRuntime

class SubAgent(TypedDict):
    """Specification for a raw subagent created by the `task` tool.

    Required fields:
        name: Unique identifier used as the task tool's `subagent_type`.
        model: Model used to create the subagent. This may be a model instance or a
            provider-qualified model name such as `'openai:gpt-5.5'`.

    Optional fields:
        description: Concise, action-oriented explanation of the subagent's role.
            It is shown to the main agent when it decides whether to delegate. If
            omitted, the subagent name is shown without a description.
        system_prompt: Instructions for the subagent, including tool-use guidance
            and expected response format. If omitted, no system prompt is provided.
        tools: Tools available to the subagent. If omitted, the subagent is created
            without tools.
        middleware: Additional middleware used when creating the subagent.
        response_format: Schema or structured-output strategy for the subagent.
            Structured responses are serialized as JSON and returned to the main
            agent instead of the subagent's final AI-message text.
    """

    name: str
    """Unique identifier for the subagent."""

    description: NotRequired[str]
    """What this subagent does, shown to the main agent when delegating."""

    system_prompt: NotRequired[str]
    """Instructions for the subagent."""

    tools: NotRequired[Sequence[BaseTool | Callable | dict[str, Any]]]
    """Tools the subagent can use. Defaults to an empty sequence."""

    model: str | BaseChatModel
    """Override the main agent's model.

    Use `'provider:model-name'` format.
    """

    middleware: NotRequired[list[AgentMiddleware]]
    """Additional middleware for custom behavior."""

    response_format: NotRequired[ResponseFormat[Any] | type | dict[str, Any]]
    """Structured output response format for the subagent.

    When specified, the subagent will produce a `structured_response` conforming
    to the given schema. The structured response is JSON-serialized and returned
    as the `ToolMessage` content to the parent agent, replacing the default
    last-message extraction.

    Accepted formats (from `langchain.agents.structured_output`):

    - `ToolStrategy(schema)`: Use tool calling to extract structured output from the model.
    - `ProviderStrategy(schema)`: Use the model provider's native structured output mode.
    - `AutoStrategy(schema)`: Automatically select the best strategy.
    - A bare Python `type`: A Pydantic `BaseModel` subclass, `dataclass`,
        or `TypedDict` class.

        Equivalent to `AutoStrategy(schema)`.
    - `dict[str, Any]`: A JSON schema dictionary
        (e.g., `{"type": "object", "properties": {...}, "required": [...]}`).

    Example:
        ```python
        from pydantic import BaseModel

        class Findings(BaseModel):
            findings: str
            confidence: float

        analyzer: SubAgent = {
            "name": "analyzer",
            "description": "Analyzes data and returns structured findings",
            "system_prompt": "Analyze the data and return your findings.",
            "model": "openai:gpt-5.5",
            "tools": [],
            "response_format": Findings,
        }
        ```
    """


class CompiledSubAgent(TypedDict):
    """A pre-compiled agent spec.

    !!! note

        The `runnable`'s state schema must include a 'messages' key.

        This is required for the subagent to communicate results back to
        the main agent.

    !!! note

        `CompiledSubAgent` runnables are used as provided. They do not
        inherit `create_deep_agent(state_schema=...)`; if the runnable
        needs custom state fields, compile it with a compatible state
        schema yourself.

    When the subagent completes, the parent reads the returned state:
    if `structured_response` is non-`None`, it is JSON-serialized and used as
    the `ToolMessage` content; otherwise, the last non-empty `AIMessage`
    text is used.

    Examples:
        Using `create_agent` with `response_format`:

        ```python
        from pydantic import BaseModel
        from langchain.agents import create_agent


        class Findings(BaseModel):
            summary: str
            confidence: float


        researcher: CompiledSubAgent = {
            "name": "researcher",
            "description": "Researches a topic and returns findings.",
            "runnable": create_agent(
                "openai:gpt-5.5",
                tools=[],  # your tools here
                response_format=Findings,
            ),
        }
        ```

        Custom `langgraph` graph (write `structured_response` directly):

        ```python
        def node(state):
            return {
                "messages": [...],
                "structured_response": Findings(summary="...", confidence=0.9),
            }
        ```
    """

    name: str
    """Unique identifier for the subagent."""

    description: str
    """What this subagent does.

    The main agent uses this to decide when to delegate.
    """

    runnable: Runnable
    """A custom agent implementation.

    Create a custom agent using either:

    1. LangChain's [`create_agent()`](https://docs.langchain.com/oss/python/langchain/quickstart)
    2. A custom graph using [`langgraph`](https://docs.langchain.com/oss/python/langgraph/quickstart)

    If you're creating a custom graph, make sure the state schema includes
    a 'messages' key. This is required for the subagent to communicate
    results back to the main agent.
    """


_EXCLUDED_STATE_KEYS = {
    "messages",
    "todos",
    "structured_response",
}
"""State keys that are excluded when passing state to subagents and when
returning updates from subagents.

When returning updates:

1. The messages key is handled explicitly to ensure only the final message
    is included
2. The todos and `structured_response` keys are excluded as they do not have
    a defined reducer and no clear meaning for returning them from a subagent
    to the main agent.
3. Agent-private fields on middleware state schemas are excluded from both
    subagent output and subagent inputs.
"""


class TaskToolSchema(BaseModel):
    """Input schema for the `task` tool."""

    description: str = Field(
        description=(
            "A detailed description of the task for the subagent to perform autonomously. "
            "Include all necessary context and specify the expected output format."
        )
    )

    subagent_type: str = Field(description=("The type of subagent to use. Must be one of the available agent types listed in the tool description."))


TASK_TOOL_DESCRIPTION = """Launch an ephemeral subagent to handle complex, multi-step independent tasks with isolated context windows.

Available agent types and the tools they have access to:
{available_agents}

When using the Task tool, you must specify a subagent_type parameter to select which agent type to use.

## Usage notes:
1. Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses
2. When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
3. Each agent invocation is stateless. You will not be able to send additional messages to the agent, nor will the agent be able to communicate with you outside of its final report. Therefore, your prompt should contain a highly detailed task description for the agent to perform autonomously and you should specify exactly what information the agent should return back to you in its final and only message to you.
4. The agent's outputs should generally be trusted
5. Clearly tell the agent whether you expect it to create content, perform analysis, or just do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent
6. If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement.
7. When only the general-purpose agent is provided, you should use it for all tasks. It is great for isolating context and token usage, and completing specific, complex tasks, as it has all the same capabilities as the main agent.

### Example usage of the general-purpose agent:

<example_agent_descriptions>
"general-purpose": use this agent for general purpose tasks, it has access to all tools as the main agent.
</example_agent_descriptions>

<example>
User: "I want to conduct research on the accomplishments of Lebron James, Michael Jordan, and Kobe Bryant, and then compare them."
Assistant: *Uses the task tool in parallel to conduct isolated research on each of the three players*
Assistant: *Synthesizes the results of the three isolated research tasks and responds to the User*
<commentary>
Research is a complex, multi-step task in it of itself.
The research of each individual player is not dependent on the research of the other players.
The assistant uses the task tool to break down the complex objective into three isolated tasks.
Each research task only needs to worry about context and tokens about one player, then returns synthesized information about each player as the Tool Result.
This means each research task can dive deep and spend tokens and context deeply researching each player, but the final result is synthesized information, and saves us tokens in the long run when comparing the players to each other.
</commentary>
</example>

<example>
User: "Analyze a single large code repository for security vulnerabilities and generate a report."
Assistant: *Launches a single `task` subagent for the repository analysis*
Assistant: *Receives report and integrates results into final summary*
<commentary>
Subagent is used to isolate a large, context-heavy task, even though there is only one. This prevents the main thread from being overloaded with details.
If the user then asks followup questions, we have a concise report to reference instead of the entire history of analysis and tool calls, which is good and saves us time and money.
</commentary>
</example>

<example>
User: "Schedule two meetings for me and prepare agendas for each."
Assistant: *Calls the task tool in parallel to launch two `task` subagents (one per meeting) to prepare agendas*
Assistant: *Returns final schedules and agendas*
<commentary>
Tasks are simple individually, but subagents help silo agenda preparation.
Each subagent only needs to worry about the agenda for one meeting.
</commentary>
</example>

<example>
User: "I want to order a pizza from Dominos, order a burger from McDonald's, and order a salad from Subway."
Assistant: *Calls tools directly in parallel to order a pizza from Dominos, a burger from McDonald's, and a salad from Subway*
<commentary>
The assistant did not use the task tool because the objective is super simple and clear and only requires a few trivial tool calls.
It is better to just complete the task directly and NOT use the `task` tool.
</commentary>
</example>

### Example usage with custom agents:

<example_agent_descriptions>
"content-reviewer": use this agent after you are done creating significant content or documents
"greeting-responder": use this agent when to respond to user greetings with a friendly joke
"research-analyst": use this agent to conduct thorough research on complex topics
</example_agent_descriptions>

<example>
user: "Please write a function that checks if a number is prime"
assistant: Sure let me write a function that checks if a number is prime
assistant: First let me use the Write tool to write a function that checks if a number is prime
assistant: I'm going to use the Write tool to write the following code:
<code>
function isPrime(n) {{
  if (n <= 1) return false
  for (let i = 2; i * i <= n; i++) {{
    if (n % i === 0) return false
  }}
  return true
}}
</code>
<commentary>
Since significant content was created and the task was completed, now use the content-reviewer agent to review the work
</commentary>
assistant: Now let me use the content-reviewer agent to review the code
assistant: Uses the Task tool to launch with the content-reviewer agent
</example>

<example>
user: "Can you help me research the environmental impact of different renewable energy sources and create a comprehensive report?"
<commentary>
This is a complex research task that would benefit from using the research-analyst agent to conduct thorough analysis
</commentary>
assistant: I'll help you research the environmental impact of renewable energy sources. Let me use the research-analyst agent to conduct comprehensive research on this topic.
assistant: Uses the Task tool to launch with the research-analyst agent, providing detailed instructions about what research to conduct and what format the report should take
</example>

<example>
user: "Hello"
<commentary>
Since the user is greeting, use the greeting-responder agent to respond with a friendly joke
</commentary>
assistant: "I'm going to use the Task tool to launch with the greeting-responder agent"
</example>"""  # noqa: E501

TASK_SYSTEM_PROMPT = """## `task` (subagent spawner)

You have access to a `task` tool to launch short-lived subagents that handle isolated tasks. These agents are ephemeral — they live only for the duration of the task and return a single result.

When to use the task tool:

- When a task is complex and multi-step, and can be fully delegated in isolation
- When a task is independent of other tasks and can run in parallel
- When a task requires focused reasoning or heavy token/context usage that would bloat the orchestrator thread
- When sandboxing improves reliability (e.g. code execution, structured searches, data formatting)
- When you only care about the output of the subagent, and not the intermediate steps (ex. performing a lot of research and then returned a synthesized report, performing a series of computations or lookups to achieve a concise, relevant answer.)

Subagent lifecycle:

1. **Spawn** → Provide clear role, instructions, and expected output
2. **Run** → The subagent completes the task autonomously
3. **Return** → The subagent provides a single structured result
4. **Reconcile** → Incorporate or synthesize the result into the main thread

When NOT to use the task tool:

- If you need to see the intermediate reasoning or steps after the subagent has completed (the task tool hides them)
- If the task is trivial (a few tool calls or simple lookup)
- If delegating does not reduce token usage, complexity, or context switching
- If splitting would add latency without benefit

## Important Task Tool Usage Notes to Remember

- Whenever possible, parallelize the work that you do. This is true for both tool_calls, and for tasks. Whenever you have independent steps to complete - make tool_calls, or kick off tasks (subagents) in parallel to accomplish them faster. This saves time for the user, which is incredibly important.
- Remember to use the `task` tool to silo independent tasks within a multi-part objective.
- You should use the `task` tool whenever you have a complex task that will take multiple steps, and is independent from other tasks that the agent needs to complete. These agents are highly competent and efficient.

Available subagent types:

{available_agents}"""  # noqa: E501

SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY = "__deepagents_subagent_response_format"
"""Configurable key used by task-tool callers to request dynamic response format."""

def _get_subagent_response_format_config(
    runtime: ToolRuntime,
) -> ResponseFormat[Any] | type | dict[str, Any] | None:
    """Return the response format carried in this task tool call's config."""
    config = runtime.config
    configurable = config.get("configurable") if isinstance(config, dict) else None
    if not isinstance(configurable, dict):
        return None
    value = configurable.get(SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY)
    if value is None:
        return None
    return value



@contextlib.contextmanager
def _subagent_tracing_context() -> Generator[None, None, None]:
    """Context manager that tags subagent runs with `ls_agent_type="subagent"`.

    Sets `ls_agent_type` on the langsmith tracing context `metadata`, which is
    propagated to LangSmith runs. This mirrors
    langchain's `ls_agent_type="root"` tagging behavior.

    Forwards all other current tracing-context fields (parent, client, tags,
    etc.) unchanged so this wrapper does not clobber the enclosing context.
    """
    current = get_tracing_context()

    merged_metadata = {**(current.get("metadata") or {}), "ls_agent_type": "subagent"}
    # Pass every field from the current tracing context through to
    # `tracing_context` so we don't accidentally clobber fields that may be
    # added to langsmith in the future. The only change is `metadata`.

    kwargs: dict[str, Any] = {**current, "metadata": merged_metadata}

    with tracing_context(**kwargs):
        yield


def _has_marker(annotation: object, marker: object) -> bool:
    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        return any(meta is marker for meta in args[1:])
    if origin is not None:
        return any(_has_marker(arg, marker) for arg in get_args(annotation))
    return False

def _private_state_field_names(state_schema: type[object]) -> frozenset[str]:
    """Return fields annotated with `PrivateStateAttr` across state schemas."""
    names: set[str] = set()
    with contextlib.suppress(Exception):
        hints = get_type_hints(state_schema, include_extras=True)
        for name, annotation in hints.items():
            if _has_marker(annotation, PrivateStateAttr):
                names.add(name)
    return frozenset(names)

def _compile_sub_agent_spec(
    spec: SubAgent,
    *,
    state_schema: type | None = None,
) -> Runnable:
    """Create a runnable agent from a raw `SubAgent` spec.

    This is the shared entrypoint for the `create_agent` path used by
    raw subagent specs. Pre-compiled `CompiledSubAgent` runnables are already
    created by the caller and are handled separately by `SubAgentMiddleware`.

    Args:
        spec: Subagent spec to compile. Must specify `name` and `model`.
        state_schema: Optional state schema for the raw subagent agent.

    Returns:
        Runnable agent ready for task-tool invocation.

    Raises:
        ValueError: If `spec` is missing `name` or `model`.
    """
    if "name" not in spec:
        msg = "SubAgent must specify 'name'"
        raise ValueError(msg)
    if "model" not in spec:
        msg = f"SubAgent '{spec['name']}' must specify 'model'"
        raise ValueError(msg)

    create_agent_kwargs: dict[str, Any] = {
        "model": spec["model"],
        "system_prompt": spec.get("system_prompt"),
        "tools": list(spec.get("tools", [])),
        "middleware": list(spec.get("middleware", [])),
        "name": spec["name"],
        "response_format": spec.get("response_format", None),
    }
    if state_schema is not None:
        create_agent_kwargs["state_schema"] = state_schema
    return create_agent(**create_agent_kwargs).with_config(
        {
            "metadata": {"lc_agent_name": spec["name"]},
            "run_name": spec["name"],
        }
    )

def _build_task_tool(
    subagents: Sequence[SubAgent | CompiledSubAgent],
    *,
    description: str | None = TASK_TOOL_DESCRIPTION,
    excluded_state_keys: frozenset[str] | None = None,
    state_schema: type | None = None,
) -> BaseTool:
    """Create a task tool from subagent specs.

    Args:
        subagents: List of raw or compiled subagent specs.
        description: Custom description for the task tool. If `None`,
            uses default template. Supports `{available_agents}` placeholder.
        excluded_state_keys: State keys marked with `PrivateStateAttr` that
            should be stripped from input state before invoking subagents.
        state_schema: State schema passed when compiling raw subagent specs.

    Returns:
        A StructuredTool that can invoke subagents by type.
    """
    if not subagents:
        msg = "At least one subagent must be specified"
        raise ValueError(msg)

    subagents_by_name: dict[str, SubAgent | CompiledSubAgent] = {}
    for spec in subagents:
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            msg = "Every subagent must specify a non-empty string 'name'"
            raise ValueError(msg)
        if name in subagents_by_name:
            msg = f"Duplicate subagent name: {name!r}"
            raise ValueError(msg)
        if "runnable" in spec:
            if spec["runnable"] is None:
                msg = f"CompiledSubAgent {name!r} must specify 'runnable'"
                raise ValueError(msg)
        elif "model" not in spec:
            msg = f"SubAgent {name!r} must specify 'model'"
            raise ValueError(msg)
        subagents_by_name[name] = spec

    subagent_runnables: dict[str, Runnable] = {
        name: (
            cast("CompiledSubAgent", spec)["runnable"]
            if "runnable" in spec
            else _compile_sub_agent_spec(
                cast("SubAgent", spec),
                state_schema=state_schema,
            )
        )
        for name, spec in subagents_by_name.items()
    }

    def _get_subagent_spec(subagent_type: str) -> SubAgent | CompiledSubAgent:
        """Validates and returns a subagent spec from the provided input"""
        try:
            return subagents_by_name[subagent_type]
        except KeyError:
            allowed_types = ", ".join(f"`{name}`" for name in subagents_by_name)
            msg = (
                f"Cannot use subagent type `{subagent_type}` because it does "
                f"not exist. The only allowed types are {allowed_types}"
            )
            raise ValueError(msg) from None

    def _get_subagent_runnable(
        spec: SubAgent | CompiledSubAgent,
        runtime: ToolRuntime,
    ) -> Runnable:
        """Return the baseline runnable or compile a dynamic-format raw spec."""
        response_format = _get_subagent_response_format_config(runtime)

        if "runnable" in spec:
            if response_format is not None:
                msg = (
                    f'response_format cannot be used with compiled subagent "{spec["name"]}"; '
                    "dynamic schemas require a raw SubAgent spec."
                )
                raise ValueError(msg)
            return subagent_runnables[spec["name"]].with_config(
                {
                    "metadata": {"lc_agent_name": spec["name"]},
                    "run_name": spec["name"],
                }
            )

        if response_format is None:
            return subagent_runnables[spec["name"]]

        # When a custom response format is provided,
        dynamic_spec = cast(
            "SubAgent",
            {**spec, "response_format": response_format},
        )
        return _compile_sub_agent_spec(dynamic_spec, state_schema=state_schema)

    def _filter_subagent_state(result: dict) -> dict:
        """Filter out excluded keys from input given to the subagent"""
        return {k: v for k, v in result.items() if k not in _EXCLUDED_STATE_KEYS and k not in excluded_state_keys}

    def _extract_subagent_response(result: dict) -> str:
        """Extract the response returned by a completed subagent run.

        Structured responses take precedence and are serialized as JSON. Otherwise,
        the text of the last non-empty `AIMessage` is returned.

        Args:
            result: Final state returned by the subagent runnable. The state must
                contain a `messages` sequence and may contain a
                `structured_response`.

        Returns:
            The serialized structured response or the final AI message text.

        Raises:
            ValueError: If `result` has no `messages` key or contains neither a
                structured response nor an AI message with text.
        """
        if "messages" not in result:
            error_msg = (
                "CompiledSubAgent must return a dict containing a 'messages' key. "
                "Custom StateGraphs used with CompiledSubAgent should include 'messages' "
                "in their state schema to communicate results back to the main agent."
            )
            raise ValueError(error_msg)

        # If the subagent has a structured response output, serialize it and return it
        structured = result.get("structured_response")
        if structured is not None:
            if hasattr(structured, "model_dump_json"):
                return structured.model_dump_json()
            elif dataclasses.is_dataclass(structured) and not isinstance(structured, type):
                return json.dumps(dataclasses.asdict(structured))
            else:
                return json.dumps(structured)

        # Walk back to the last AIMessage with non-empty text.
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                text = msg.text.rstrip() if msg.text else ""
                if text:
                    return text

        error_msg = (
            "SubAgent didn't return a usable response. Subagent runs must return either "
            "contain a message list with vali text content in the last AIMessage, or a "
            "structured response dict"
        )
        raise ValueError(error_msg)

    def _format_subagent_update(result: Any, tool_call_id: str) -> Command:
        return Command(
            update={
                **_filter_subagent_state(result),
                "messages": [
                    ToolMessage(
                        content=_extract_subagent_response(result),
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    def task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        spec = _get_subagent_spec(subagent_type)
        subagent = _get_subagent_runnable(spec, runtime)
        subagent_input = _filter_subagent_state(runtime.state)
        subagent_input["messages"] = [HumanMessage(content=description)]
        # The parent's callbacks, tags and configurable reach the subagent
        # automatically: langgraph's `ensure_config` seeds each run from the
        # ambient parent config and (as of langgraph#7926) merges it per-key, so
        # the subagent's bound config still wins collisions (e.g. `lc_agent_name`,
        # `recursion_limit`) and parent metadata propagates (deepagents#3634).
        # Forwarding those keys explicitly would double-count under the merge
        # (e.g. duplicate `tags`), so we only stamp the subagent tracing tag.
        subagent_config: RunnableConfig = {"configurable": {"ls_agent_type": "subagent"}}
        with _subagent_tracing_context():
            result = subagent.invoke(subagent_input, subagent_config)
        if not runtime.tool_call_id:
            return _extract_subagent_response(result)
        return _format_subagent_update(result, runtime.tool_call_id)

    async def atask(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime
    ) -> str | Command:
        spec = _get_subagent_spec(subagent_type)
        subagent = _get_subagent_runnable(spec, runtime)
        subagent_input = _filter_subagent_state(runtime.state)
        subagent_input["messages"] = [HumanMessage(content=description)]
        # The parent's callbacks, tags and configurable reach the subagent
        # automatically: langgraph's `ensure_config` seeds each run from the
        # ambient parent config and (as of langgraph#7926) merges it per-key, so
        # the subagent's bound config still wins collisions (e.g. `lc_agent_name`,
        # `recursion_limit`) and parent metadata propagates (deepagents#3634).
        # Forwarding those keys explicitly would double-count under the merge
        # (e.g. duplicate `tags`), so we only stamp the subagent tracing tag.
        subagent_config: RunnableConfig = {"configurable": {"ls_agent_type": "subagent"}}
        with _subagent_tracing_context():
            result = await subagent.ainvoke(subagent_input, subagent_config)
        if not runtime.tool_call_id:
            return _extract_subagent_response(result)
        return _format_subagent_update(result, runtime.tool_call_id)

    if description is not None:
        subagent_description_str = "\n".join(
            f"- {s['name']}: {s.get('description', '')}" for s in subagents
        )
        description = description.format(available_agents=subagent_description_str)

    return StructuredTool.from_function(
        name="task",
        func=task,
        coroutine=atask,
        description=description,
        infer_schema=False,
        args_schema=TaskToolSchema
    )

class SubAgentMiddleware(AgentMiddleware[StateT, ContextT, ResponseT]):
    """Middleware for providing subagents to an agent via a `task` tool.

    This middleware adds a `task` tool to the agent that can be used
    to invoke subagents.

    Subagents are useful for handling complex tasks that require multiple steps,
    or tasks that require a lot of context to resolve.

    A chief benefit of subagents is that they can handle multi-step tasks,
    and then return a clean, concise response to the main agent.

    Subagents are also great for different domains of expertise that require
    a narrower subset of tools and focus.

    Args:
        subagents: Subagents available to the `task` tool. Raw `SubAgent` specs must
            define `name` and `model`; `description`, `system_prompt`, and `tools`
            are optional. Precompiled specs must define `name`, `description`, and
            `runnable`.
            Names must be unique because the task's `subagent_type` selects a spec by
            name.
        system_prompt: Instructions appended to the main agent's system prompt.
            The `{available_agents}` placeholder is replaced with the configured
            subagent names and descriptions.
        tool_description: Description for the `task` tool. The
            `{available_agents}` placeholder is replaced with the configured
            subagent names and descriptions.
        state_schema: Main-agent state schema passed to raw subagent compilation.
            Fields annotated with `PrivateStateAttr` are excluded from state passed to
            subagents and from state updates returned to the main agent. Compiled
            subagents retain the schema of their caller-provided runnables.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware.subagents import SubAgentMiddleware

        agent = create_agent(
            "openai:gpt-5.5",
            middleware=[
                SubAgentMiddleware(
                    subagents=[
                        {
                            "name": "researcher",
                            "description": "Research a topic and summarize findings.",
                            "system_prompt": "You are a research specialist.",
                            "model": "openai:gpt-5.5",
                            "tools": [],
                        }
                    ],
                )
            ],
        )
        ```
    """
    def __init__(
        self,
        *,
        subagents: Sequence[SubAgent | CompiledSubAgent],
        system_prompt: str | None = TASK_SYSTEM_PROMPT,
        tool_description: str | None = DEFAULT_TOOL_DESCRIPTION,
        state_schema: type | None = None,
    ) -> None:
        super().__init__()

        subagent_description_str = "\n".join(
            f"- {s.get('name', '')}: {s.get('description', '')}" for s in subagents
        )
        self._system_prompt = (
            system_prompt.format(available_agents=subagent_description_str)
            if system_prompt is not None
            else None
        )

        self.tools = [
            _build_task_tool(
                subagents,
                description=tool_description,
                excluded_state_keys=(
                    _private_state_field_names(state_schema)
                    if state_schema is not None
                    else frozenset()
                ),
                state_schema=state_schema,
            )
        ]

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Update the system message to include instructions on using subagents."""
        if self._system_prompt is not None:
            new_system_message = append_to_system_message(request.system_message, self._system_prompt)
            return handler(request.override(system_message=new_system_message))
        return handler(request)


    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Update the system message to include instructions on using subagents."""
        if self._system_prompt is not None:
            new_system_message = append_to_system_message(request.system_message, self._system_prompt)
            return await handler(request.override(system_message=new_system_message))
        return await handler(request)
