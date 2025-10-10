"""Middleware for providing subagents to an agent via a `task` tool."""

from collections.abc import Callable, Sequence
from typing import Annotated, Any, NotRequired, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool
from langgraph.types import Command

from langchain.agents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware.planning import PlanningMiddleware
from langchain.agents.middleware.prompt_caching import AnthropicPromptCachingMiddleware
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain.tools import InjectedState, InjectedToolCallId


class DefinedSubAgent(TypedDict):
    """A subagent constructed with user-defined parameters."""

    name: str
    """The name of the subagent."""

    description: str
    """The description of the subagent."""

    prompt: str
    """The system prompt to use for the subagent."""

    tools: NotRequired[list[BaseTool]]
    """The tools to use for the subagent."""

    model: NotRequired[str | BaseChatModel]
    """The model for the subagent."""

    middleware: NotRequired[list[AgentMiddleware]]
    """The middleware to use for the subagent."""


class CustomSubAgent(TypedDict):
    """A Runnable passed in as a subagent."""

    name: str
    """The name of the subagent."""

    description: str
    """The description of the subagent."""

    runnable: Runnable
    """The Runnable to use for the subagent."""


DEFAULT_SUBAGENT_PROMPT = """In order to complete the objective that the user asks of you, you have access to a number of standard tools.
"""  # noqa: E501

TASK_TOOL_DESCRIPTION = """Launch an ephemeral subagent to handle complex, multi-step independent tasks with isolated context windows.

Available agent types and the tools they have access to:
- general-purpose: General-purpose agent for researching complex questions, searching for files and content, and executing multi-step tasks. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries use this agent to perform the search for you. This agent has access to all tools as the main agent.
{other_agents}

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
It is better to just complete the task directly and NOT use the `task`tool.
</commentary>
</example>

### Example usage with custom agents:

<example_agent_descriptions>
"content-reviewer": use this agent after you are done creating significant content or documents
"greeting-responder": use this agent when to respond to user greetings with a friendly joke
"research-analyst": use this agent to conduct thorough research on complex topics
</example_agent_description>

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


def _get_subagents(
    default_subagent_model: str | BaseChatModel,
    default_subagent_tools: Sequence[BaseTool | Callable | dict[str, Any]],
    subagents: list[DefinedSubAgent | CustomSubAgent],
) -> tuple[dict[str, Any], list[str]]:
    from langchain.agents.factory import create_agent

    default_subagent_middleware = [
        PlanningMiddleware(),
        FilesystemMiddleware(),
        SummarizationMiddleware(
            model=default_subagent_model,
            max_tokens_before_summary=120000,
            messages_to_keep=20,
        ),
        AnthropicPromptCachingMiddleware(ttl="5m", unsupported_model_behavior="ignore"),
    ]

    # Create the general-purpose subagent
    general_purpose_subagent = create_agent(
        model=default_subagent_model,
        system_prompt=DEFAULT_SUBAGENT_PROMPT,
        tools=default_subagent_tools,
        middleware=default_subagent_middleware,
    )
    agents: dict[str, Any] = {"general-purpose": general_purpose_subagent}
    subagent_descriptions = []
    for _agent in subagents:
        subagent_descriptions.append(f"- {_agent['name']}: {_agent['description']}")
        if "runnable" in _agent:
            # Type narrowing: _agent is CustomSubAgent here
            custom_agent = cast("CustomSubAgent", _agent)
            agents[custom_agent["name"]] = custom_agent["runnable"]
            continue
        _tools = _agent.get("tools", list(default_subagent_tools))

        subagent_model = _agent.get("model", default_subagent_model)

        if "middleware" in _agent:
            _middleware = [*default_subagent_middleware, *_agent["middleware"]]
        else:
            _middleware = default_subagent_middleware

        agents[_agent["name"]] = create_agent(
            subagent_model,
            system_prompt=_agent["prompt"],
            tools=_tools,
            middleware=_middleware,
            checkpointer=False,
        )
    return agents, subagent_descriptions


def _create_task_tool(
    default_subagent_model: str | BaseChatModel,
    default_subagent_tools: Sequence[BaseTool | Callable | dict[str, Any]],
    subagents: list[DefinedSubAgent | CustomSubAgent],
    *,
    is_async: bool = False,
) -> BaseTool:
    subagent_graphs, subagent_descriptions = _get_subagents(
        default_subagent_model, default_subagent_tools, subagents
    )
    subagent_description_str = "\n".join(subagent_descriptions)

    def _return_command_with_state_update(result: dict, tool_call_id: str) -> Command:
        state_update = {k: v for k, v in result.items() if k not in ["todos", "messages"]}
        return Command(
            update={
                **state_update,
                "messages": [
                    ToolMessage(result["messages"][-1].content, tool_call_id=tool_call_id)
                ],
            }
        )

    task_tool_description = TASK_TOOL_DESCRIPTION.format(other_agents=subagent_description_str)
    if is_async:

        @tool(description=task_tool_description)
        async def task(
            description: str,
            subagent_type: str,
            state: Annotated[dict, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> str | Command:
            if subagent_type not in subagent_graphs:
                msg = (
                    f"Error: invoked agent of type {subagent_type}, "
                    f"the only allowed types are {[f'`{k}`' for k in subagent_graphs]}"
                )
                raise ValueError(msg)
            subagent = subagent_graphs[subagent_type]
            state["messages"] = [HumanMessage(content=description)]
            if "todos" in state:
                del state["todos"]
            result = await subagent.ainvoke(state)
            return _return_command_with_state_update(result, tool_call_id)
    else:

        @tool(description=task_tool_description)
        def task(
            description: str,
            subagent_type: str,
            state: Annotated[dict, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> str | Command:
            if subagent_type not in subagent_graphs:
                msg = (
                    f"Error: invoked agent of type {subagent_type}, "
                    f"the only allowed types are {[f'`{k}`' for k in subagent_graphs]}"
                )
                raise ValueError(msg)
            subagent = subagent_graphs[subagent_type]
            state["messages"] = [HumanMessage(content=description)]
            if "todos" in state:
                del state["todos"]
            result = subagent.invoke(state)
            return _return_command_with_state_update(result, tool_call_id)

    return task


class SubAgentMiddleware(AgentMiddleware):
    """Middleware for providing subagents to an agent via a `task` tool.

    This  middleware adds a `task` tool to the agent that can be used to invoke subagents.
    Subagents are useful for handling complex tasks that require multiple steps, or tasks
    that require a lot of context to resolve.

    A chief benefit of subagents is that they can handle multi-step tasks, and then return
    a clean, concise response to the main agent.

    Subagents are also great for different domains of expertise that require a narrower
    subset of tools and focus.

    This middleware comes with a default general-purpose subagent that can be used to
    handle the same tasks as the main agent, but with isolated context.

    Args:
        default_subagent_model: The model to use for the general-purpose subagent.
            Can be a LanguageModelLike or a dict for init_chat_model.
        default_subagent_tools: The tools to use for the general-purpose subagent.
        subagents: A list of additional subagents to provide to the agent.
        system_prompt_extension: Additional instructions on how the main agent should use subagents.
        is_async: Whether the `task` tool should be asynchronous.

    Example:
        ```python
        from langchain.agents.middleware.subagents import SubAgentMiddleware
        from langchain.agents import create_agent

        agent = create_agent("openai:gpt-4o", middleware=[SubAgentMiddleware(subagents=[])])

        # Agent now has access to the `task` tool
        result = await agent.invoke({"messages": [HumanMessage("Help me refactor my codebase")]})
        ```
    """

    def __init__(
        self,
        *,
        default_subagent_model: str | BaseChatModel,
        default_subagent_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
        subagents: list[DefinedSubAgent | CustomSubAgent] | None = None,
        system_prompt_extension: str | None = None,
        is_async: bool = False,
    ) -> None:
        """Initialize the SubAgentMiddleware."""
        super().__init__()
        self.system_prompt_extension = system_prompt_extension
        task_tool = _create_task_tool(
            default_subagent_model,
            default_subagent_tools or [],
            subagents or [],
            is_async=is_async,
        )
        self.tools = [task_tool]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], AIMessage],
    ) -> AIMessage:
        """Update the system prompt to include instructions on using subagents."""
        if self.system_prompt_extension is not None:
            request.system_prompt = (
                request.system_prompt + "\n\n" + self.system_prompt_extension
                if request.system_prompt
                else self.system_prompt_extension
            )
        return handler(request)
