"""Deepagents come with planning, filesystem, and subagents, along with other supportive middlewares.."""
# ruff: noqa: E501

from collections.abc import Callable, Sequence
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from langchain.agents.factory import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware, ToolConfig
from langchain.agents.middleware.planning import PlanningMiddleware
from langchain.agents.middleware.prompt_caching import AnthropicPromptCachingMiddleware
from langchain.agents.middleware.subagents import (
    CustomSubAgent,
    DefinedSubAgent,
    SubAgentMiddleware,
)
from langchain.agents.middleware.summarization import SummarizationMiddleware

BASE_AGENT_PROMPT = """
In order to complete the objective that the user asks of you, you have access to a number of standard tools.  # noqa: E501
"""


def get_default_model() -> ChatAnthropic:
    """Get the default model for deep agents.

    Returns:
        ChatAnthropic instance configured with Claude Sonnet 4.
    """
    return ChatAnthropic(
        model_name="claude-sonnet-4-20250514",
        timeout=None,
        stop=None,
        model_kwargs={"max_tokens": 64000},
    )


def agent_builder(
    tools: Sequence[BaseTool | Callable | dict[str, Any]],
    instructions: str,
    middleware: list[AgentMiddleware] | None = None,
    tool_configs: dict[str, bool | ToolConfig] | None = None,
    model: str | BaseChatModel | None = None,
    subagents: list[DefinedSubAgent | CustomSubAgent] | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    *,
    use_longterm_memory: bool = False,
    is_async: bool = False,
) -> Any:
    """Build a deep agent with standard middleware stack.

    Args:
        tools: The tools the agent should have access to.
        instructions: The instructions for the agent system prompt.
        middleware: Additional middleware to apply after standard middleware.
        tool_configs: Optional tool interrupt configurations.
        model: The model to use. Defaults to Claude Sonnet 4.
        subagents: Optional list of subagent configurations.
        context_schema: Optional schema for the agent context.
        checkpointer: Optional checkpointer for state persistence.
        store: Optional store for longterm memory.
        use_longterm_memory: Whether to enable longterm memory features.
        is_async: Whether to create async subagent tools.

    Returns:
        A configured agent with deep agent middleware stack.
    """
    if model is None:
        model = get_default_model()

    deepagent_middleware = [
        PlanningMiddleware(),
        FilesystemMiddleware(
            use_longterm_memory=use_longterm_memory,
        ),
        SubAgentMiddleware(
            default_subagent_tools=tools,
            default_subagent_model=model,
            subagents=subagents if subagents is not None else [],
            is_async=is_async,
        ),
        SummarizationMiddleware(
            model=model,
            max_tokens_before_summary=120000,
            messages_to_keep=20,
        ),
        AnthropicPromptCachingMiddleware(ttl="5m", unsupported_model_behavior="ignore"),
    ]
    if tool_configs is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=tool_configs))
    if middleware is not None:
        deepagent_middleware.extend(middleware)

    return create_agent(
        model,
        system_prompt=instructions + "\n\n" + BASE_AGENT_PROMPT,
        tools=tools,
        middleware=deepagent_middleware,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
    )


def create_deep_agent(
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    instructions: str = "",
    middleware: list[AgentMiddleware] | None = None,
    model: str | BaseChatModel | None = None,
    subagents: list[DefinedSubAgent | CustomSubAgent] | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    *,
    use_longterm_memory: bool = False,
    tool_configs: dict[str, bool | ToolConfig] | None = None,
) -> Any:
    """Create a deep agent.

    This agent will by default have access to a tool to write todos (write_todos),
    four file editing tools: write_file, ls, read_file, edit_file, and a tool to call
    subagents.

    Args:
        tools: The tools the agent should have access to.
        instructions: The additional instructions the agent should have. Will go in
            the system prompt.
        middleware: Additional middleware to apply after standard middleware.
        model: The model to use.
        subagents: The subagents to use. Each subagent should be a dictionary with the
            following keys:
                - `name`
                - `description` (used by the main agent to decide whether to call the
                  sub agent)
                - `prompt` (used as the system prompt in the subagent)
                - (optional) `tools`
                - (optional) `model` (either a LanguageModelLike instance or dict
                  settings)
                - (optional) `middleware` (list of AgentMiddleware)
        context_schema: The schema of the deep agent.
        checkpointer: Optional checkpointer for persisting agent state between runs.
        store: Optional store for persisting longterm memories.
        use_longterm_memory: Whether to use longterm memory - you must provide a store
            in order to use longterm memory.
        tool_configs: Optional Dict[str, HumanInTheLoopConfig] mapping tool names to
            interrupt configs.

    Returns:
        A configured deep agent.
    """
    if tools is None:
        tools = []
    return agent_builder(
        tools=tools,
        instructions=instructions,
        middleware=middleware,
        model=model,
        subagents=subagents,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        use_longterm_memory=use_longterm_memory,
        tool_configs=tool_configs,
        is_async=False,
    )


def async_create_deep_agent(
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    instructions: str = "",
    middleware: list[AgentMiddleware] | None = None,
    model: str | BaseChatModel | None = None,
    subagents: list[DefinedSubAgent | CustomSubAgent] | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    *,
    use_longterm_memory: bool = False,
    tool_configs: dict[str, bool | ToolConfig] | None = None,
) -> Any:
    """Create an async deep agent.

    This agent will by default have access to a tool to write todos (write_todos),
    four file editing tools: write_file, ls, read_file, edit_file, and a tool to call
    subagents.

    Args:
        tools: The tools the agent should have access to.
        instructions: The additional instructions the agent should have. Will go in
            the system prompt.
        middleware: Additional middleware to apply after standard middleware.
        model: The model to use.
        subagents: The subagents to use. Each subagent should be a dictionary with the
            following keys:
                - `name`
                - `description` (used by the main agent to decide whether to call the
                  sub agent)
                - `prompt` (used as the system prompt in the subagent)
                - (optional) `tools`
                - (optional) `model` (either a LanguageModelLike instance or dict
                  settings)
                - (optional) `middleware` (list of AgentMiddleware)
        context_schema: The schema of the deep agent.
        checkpointer: Optional checkpointer for persisting agent state between runs.
        store: Optional store for persisting longterm memories.
        use_longterm_memory: Whether to use longterm memory - you must provide a store
            in order to use longterm memory.
        tool_configs: Optional Dict[str, HumanInTheLoopConfig] mapping tool names to
            interrupt configs.

    Returns:
        A configured deep agent with async subagent tools.
    """
    if tools is None:
        tools = []
    return agent_builder(
        tools=tools,
        instructions=instructions,
        middleware=middleware,
        model=model,
        subagents=subagents,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        use_longterm_memory=use_longterm_memory,
        tool_configs=tool_configs,
        is_async=True,
    )
