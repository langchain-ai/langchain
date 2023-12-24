from __future__ import annotations

from typing import Sequence

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool

from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description


def create_react_agent(
    llm: BaseLanguageModel, tools: Sequence[BaseTool], prompt: BasePromptTemplate
) -> Runnable:
    """Create an agent that uses ReAct prompting.

    Examples:

        .. code-block:: python

            from langchain.agents.react_agent import (
                create_react_agent,
            )
            from langchain import hub
            from langchain.llms import OpenAI
            from langchain.agents import AgentExecutor

            prompt = hub.pull("hwchase17/react")
            model = OpenAI()
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
    missing_vars = {"tools", "tool_names", "agent_scratchpad"}.difference(
        prompt.input_variables
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    prompt = prompt.partial(
        tools=render_text_description(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
    )
    llm_with_stop = llm.bind(stop=["\nObservation"])
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
            # Give it a default
            chat_history=lambda x: x.get("chat_history", ""),
        )
        | prompt
        | llm_with_stop
        | ReActSingleInputOutputParser()
    )
    return agent
