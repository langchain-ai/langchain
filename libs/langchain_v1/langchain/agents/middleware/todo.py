"""Planning and task management middleware for agents."""

from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal, cast

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, StructuredTool, tool
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel
from typing_extensions import NotRequired, TypedDict, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    OmitFromInput,
    ResponseT,
)
from langchain.tools import ToolRuntime


class Todo(TypedDict):
    """A single todo item with content and status."""

    content: str
    """The content/description of the todo item."""

    status: Literal["pending", "in_progress", "completed"]
    """The current status of the todo item."""


class PlanningState(AgentState[ResponseT]):
    """State schema for the todo middleware.

    Type Parameters:
        ResponseT: The type of the structured response. Defaults to `Any`.
    """

    todos: Annotated[NotRequired[list[Todo]], OmitFromInput]
    """List of todo items for tracking task progress."""


class WriteTodosInput(BaseModel):
    """Input schema for the `write_todos` tool."""

    todos: list[Todo]


WRITE_TODOS_TOOL_DESCRIPTION = """Use this tool to create and manage a structured task list for your current work session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.

Only use this tool if you think it will be helpful in staying organized. If the user's request is trivial and takes less than 3 steps, it is better to NOT use this tool and just do the task directly.

## When to Use This Tool
Use this tool in these scenarios:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests todo list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
5. The plan may need future revisions or updates based on results from the first few steps

## How to Use This Tool
1. When you start working on a task - Mark it as in_progress BEFORE beginning work.
2. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation.
3. You can also update future tasks, such as deleting them if they are no longer necessary, or adding new tasks that are necessary. Don't change previously completed tasks.
4. You can make several updates to the todo list at once. For example, when you complete a task, you can mark the next task you need to start as in_progress.

## When NOT to Use This Tool
It is important to skip using this tool when:
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no benefit
3. The task can be completed in less than 3 trivial steps
4. The task is purely conversational or informational

## Task States and Management

1. **Task States**: Use these states to track progress:
   - pending: Task not yet started
   - in_progress: Currently working on (you can have multiple tasks in_progress at a time if they are not related to each other and can be run in parallel)
   - completed: Task finished successfully

2. **Task Management**:
   - Update task status in real-time as you work
   - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
   - Complete current tasks before starting new ones
   - Remove tasks that are no longer relevant from the list entirely
   - IMPORTANT: When you write this todo list, you should mark your first task (or tasks) as in_progress immediately!.
   - IMPORTANT: Unless all tasks are completed, you should always have at least one task in_progress to show the user that you are working on something.

3. **Task Completion Requirements**:
   - ONLY mark a task as completed when you have FULLY accomplished it
   - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
   - When blocked, create a new task describing what needs to be resolved
   - Never mark a task as completed if:
     - There are unresolved issues or errors
     - Work is partial or incomplete
     - You encountered blockers that prevent completion
     - You couldn't find necessary resources or dependencies
     - Quality standards haven't been met

4. **Task Breakdown**:
   - Create specific, actionable items
   - Break complex tasks into smaller, manageable steps
   - Use clear, descriptive task names

Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully
Remember: If you only need to make a few tool calls to complete a task, and it is clear what you need to do, it is better to just do the task directly and NOT call this tool at all."""  # noqa: E501

WRITE_TODOS_SYSTEM_PROMPT = """## `write_todos`

You have access to the `write_todos` tool to help you manage and plan complex objectives.
Use this tool for complex objectives to ensure that you are tracking each necessary step and giving the user visibility into your progress.
This tool is very helpful for planning complex objectives, and for breaking down these larger complex objectives into smaller steps.

It is critical that you mark todos as completed as soon as you are done with a step. Do not batch up multiple steps before marking them as completed.
For simple objectives that only require a few steps, it is better to just complete the objective directly and NOT use this tool.
Writing todos takes time and tokens, use it when it is helpful for managing complex many-step problems! But not for simple few-step requests.

## Important To-Do List Usage Notes to Remember
- The `write_todos` tool should never be called multiple times in parallel.
- Don't be afraid to revise the To-Do list as you go. New information may reveal new tasks that need to be done, or old tasks that are irrelevant."""  # noqa: E501

_STATUS_MARKERS: dict[str, str] = {
    "pending": "[ ]",
    "in_progress": "[~]",
    "completed": "[x]",
}
"""Mapping from a `Todo` status value to the display marker used in the injected
``<current_todos>`` block."""


def _render_todos_block(todos: list[Todo]) -> str:
    """Render a list of todos as an XML-style block for system prompt injection.

    Each todo item is formatted as ``<marker> <content>`` on its own line, where
    the marker reflects the item's current status:

    - ``[ ]`` — pending
    - ``[~]`` — in_progress
    - ``[x]`` — completed

    Args:
        todos: List of todo items to render. Must be non-empty.

    Returns:
        A ``<current_todos>`` XML block string with one formatted line per item.
    """
    lines = "\n".join(
        f"{_STATUS_MARKERS.get(todo['status'], '[ ]')} {todo['content']}"
        for todo in todos
    )
    return f"<current_todos>\n{lines}\n</current_todos>"


@tool(description=WRITE_TODOS_TOOL_DESCRIPTION)
def write_todos(
    todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command[Any]:
    """Create and manage a structured task list for your current work session."""
    return Command(
        update={
            "todos": todos,
            "messages": [ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)],
        }
    )


# Dynamically create the write_todos tool with the custom description
def _write_todos(
    runtime: ToolRuntime[ContextT, PlanningState[ResponseT]], todos: list[Todo]
) -> Command[Any]:
    """Create and manage a structured task list for your current work session."""
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(f"Updated todo list to {todos}", tool_call_id=runtime.tool_call_id)
            ],
        }
    )


async def _awrite_todos(
    runtime: ToolRuntime[ContextT, PlanningState[ResponseT]], todos: list[Todo]
) -> Command[Any]:
    """Create and manage a structured task list for your current work session."""
    return _write_todos(runtime, todos)


class TodoListMiddleware(AgentMiddleware[PlanningState[ResponseT], ContextT, ResponseT]):
    """Middleware that provides todo list management capabilities to agents.

    This middleware adds a `write_todos` tool that allows agents to create and manage
    structured task lists for complex multi-step operations. It's designed to help
    agents track progress, organize complex tasks, and provide users with visibility
    into task completion status.

    The middleware automatically injects system prompts that guide the agent on when
    and how to use the todo functionality effectively. It also enforces that the
    `write_todos` tool is called at most once per model turn, since the tool replaces
    the entire todo list and parallel calls would create ambiguity about precedence.

    When `inject_current_todos` is ``True`` (the default), the live todo list is
    re-injected into the system prompt on every model call as a ``<current_todos>``
    block. This guarantees the model retains direct visibility of its current plan
    even when message history has been compacted by a summarization middleware or
    cleared for other reasons.

    Example:
        ```python
        from langchain.agents.middleware.todo import TodoListMiddleware
        from langchain.agents import create_agent

        agent = create_agent("openai:gpt-4o", middleware=[TodoListMiddleware()])

        # Agent now has access to write_todos tool and todo state tracking
        result = await agent.invoke({"messages": [HumanMessage("Help me refactor my codebase")]})

        print(result["todos"])  # Array of todo items with status tracking
        ```
    """

    state_schema = PlanningState  # type: ignore[assignment]

    def __init__(
        self,
        *,
        system_prompt: str = WRITE_TODOS_SYSTEM_PROMPT,
        tool_description: str = WRITE_TODOS_TOOL_DESCRIPTION,
        inject_current_todos: bool = True,
    ) -> None:
        """Initialize the `TodoListMiddleware` with optional custom prompts.

        Args:
            system_prompt: Custom system prompt to guide the agent on using the todo
                tool.
            tool_description: Custom description for the `write_todos` tool.
            inject_current_todos: When ``True`` (the default), the live todo list from
                agent state is re-injected into the system prompt on every model call
                as a ``<current_todos>`` block. This prevents plan drift when message
                history is compacted or summarized. Set to ``False`` to restore the
                previous behavior of only including static instructions.
        """
        super().__init__()
        self.system_prompt = system_prompt
        self.tool_description = tool_description
        self.inject_current_todos = inject_current_todos

        self.tools = [
            StructuredTool.from_function(
                name="write_todos",
                description=tool_description,
                func=_write_todos,
                coroutine=_awrite_todos,
                args_schema=WriteTodosInput,
                infer_schema=False,
            )
        ]

    def _build_system_content(
        self, request: ModelRequest[ContextT]
    ) -> list[str | dict[str, str]]:
        """Build the full list of system message content blocks for a model call.

        Combines:

        1. Any existing content blocks from the current system message.
        2. The static ``system_prompt`` instructions.
        3. When `inject_current_todos` is ``True`` and the agent state contains at
           least one todo item, a ``<current_todos>`` block reflecting the live plan.

        Centralising this logic here removes the duplication that previously existed
        between `wrap_model_call` and `awrap_model_call`.

        Args:
            request: The incoming model request, containing the current system
                message and agent state.

        Returns:
            A list of content blocks suitable for constructing a
            `~langchain_core.messages.SystemMessage`.
        """
        blocks: list[str | dict[str, str]]
        if request.system_message is not None:
            blocks = list(request.system_message.content_blocks)
            blocks.append({"type": "text", "text": f"\n\n{self.system_prompt}"})
        else:
            blocks = [{"type": "text", "text": self.system_prompt}]

        if self.inject_current_todos:
            todos: list[Todo] = (
                cast("PlanningState[Any]", request.state).get("todos") or []
            )
            if todos:
                blocks.append(
                    {"type": "text", "text": f"\n\n{_render_todos_block(todos)}"}
                )

        return blocks

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Update the system message to include the todo system prompt and live todos.

        Delegates content construction to `_build_system_content`, which appends
        the static ``system_prompt`` instructions and, when `inject_current_todos`
        is enabled, a ``<current_todos>`` block with the agent's current plan.

        Args:
            request: Model request to execute (includes state and runtime).
            handler: Callback that executes the model request and returns
                `ModelResponse`.

        Returns:
            The model call result.
        """
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", self._build_system_content(request))
        )
        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Async version of `wrap_model_call`.

        Delegates content construction to `_build_system_content`, which appends
        the static ``system_prompt`` instructions and, when `inject_current_todos`
        is enabled, a ``<current_todos>`` block with the agent's current plan.

        Args:
            request: Model request to execute (includes state and runtime).
            handler: Async callback that executes the model request and returns
                `ModelResponse`.

        Returns:
            The model call result.
        """
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", self._build_system_content(request))
        )
        return await handler(request.override(system_message=new_system_message))

    @override
    def after_model(
        self, state: PlanningState[ResponseT], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Check for parallel write_todos tool calls and return errors if detected.

        The todo list is designed to be updated at most once per model turn. Since
        the `write_todos` tool replaces the entire todo list with each call, making
        multiple parallel calls would create ambiguity about which update should take
        precedence. This method prevents such conflicts by rejecting any response that
        contains multiple write_todos tool calls.

        Args:
            state: The current agent state containing messages.
            runtime: The LangGraph runtime instance.

        Returns:
            A dict containing error ToolMessages for each write_todos call if multiple
            parallel calls are detected, otherwise None to allow normal execution.
        """
        messages = state["messages"]
        if not messages:
            return None

        last_ai_msg = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
        if not last_ai_msg or not last_ai_msg.tool_calls:
            return None

        # Count write_todos tool calls
        write_todos_calls = [tc for tc in last_ai_msg.tool_calls if tc["name"] == "write_todos"]

        if len(write_todos_calls) > 1:
            # Create error tool messages for all write_todos calls
            error_messages = [
                ToolMessage(
                    content=(
                        "Error: The `write_todos` tool should never be called multiple times "
                        "in parallel. Please call it only once per model invocation to update "
                        "the todo list."
                    ),
                    tool_call_id=tc["id"],
                    status="error",
                )
                for tc in write_todos_calls
            ]

            # Keep the tool calls in the AI message but return error messages
            # This follows the same pattern as HumanInTheLoopMiddleware
            return {"messages": error_messages}

        return None

    @override
    async def aafter_model(
        self, state: PlanningState[ResponseT], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Check for parallel write_todos tool calls and return errors if detected.

        Async version of `after_model`. The todo list is designed to be updated at
        most once per model turn. Since the `write_todos` tool replaces the entire
        todo list with each call, making multiple parallel calls would create ambiguity
        about which update should take precedence. This method prevents such conflicts
        by rejecting any response that contains multiple write_todos tool calls.

        Args:
            state: The current agent state containing messages.
            runtime: The LangGraph runtime instance.

        Returns:
            A dict containing error ToolMessages for each write_todos call if multiple
            parallel calls are detected, otherwise None to allow normal execution.
        """
        return self.after_model(state, runtime)
