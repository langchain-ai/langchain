from __future__ import annotations

from typing import Any, Callable, List, NamedTuple, Optional, Sequence

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnablePassthrough

from langchain.agents.react.prompt import DEFAULT_PROMPT_INSTRUCTIONS
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
def create_default_prompt(
    instructions: str = "You are a helpful assistant.",
):
    """Create default prompt for ReAct agent.

    Args:
        instructions: String to put at the start of the prompt,
            defaults to "You are a helpful assistant."

    Returns:
        A prompt template. Requires two or three variables:
        - `input`: this is the user input
        - `tools`: the tools the agent has access to (including descriptions).
        - `tool_names`: the list of names of the tools the agent has access to
        - `chat_history`: previous conversation, should be a string.
        - `agent_scratchpad`: this is the history of steps the agent has taken so far.
            This gets populated by the AgentExecutor.
    """
    base_prompt = PromptTemplate.from_template(
        DEFAULT_PROMPT_INSTRUCTIONS
    ).partial(instructions=instructions, chat_history="")
    return base_prompt


def _format_chat_history(_inputs):
    if "chat_history" not in _inputs:
        return ""
    else:
        chat_history = _inputs["chat_history"]
        return f"\nPrevious Conversation:\n{chat_history}\n"


def create_react_agent(
    llm: BaseLanguageModel, tools: Sequence[BaseTool], prompt: PromptTemplate
):
    """Create an agent that uses ReAct prompting.

    Examples:

        .. code-block:: python

            from langchain.agents.xml_agent import (
                create_xml_agent,
                create_default_prompt
            )
            from langchain.chat_models import ChatAnthropic
            from langchain.agents import AgentExecutor

            prompt = create_default_prompt()
            model = ChatAnthropic()
            tools = ...

            agent = create_react_agent(model, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools)

            agent_executor.invoke({"input": "hi"})

    Args:
        llm: LLM to use as the agent.
        tools: Tools this agent has access to.
        prompt: The prompt to use, must have input keys of
            `tools`, `tool_names`, and `agent_scratchpad`.

    Returns:
        A runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    """
    missing_vars = {"tools", "tool_names", "agent_scratchpad"}.difference(prompt.input_variables)
    if missing_vars:
        raise ValueError(
            f"Prompt missing required variables: {missing_vars}"
        )

    prompt = prompt.partial(
        tools=render_text_description(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
    )
    llm_with_stop = llm.bind(stop=["\nObservation"])
    agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_log_to_str(
                    x["intermediate_steps"]
                ),
                chat_history = lambda x: _format_chat_history(
                    x
                )
            )
        | prompt
        | llm_with_stop
        | ReActSingleInputOutputParser()
    )
    return agent
