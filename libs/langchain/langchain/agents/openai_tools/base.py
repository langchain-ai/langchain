from typing import Sequence

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import BaseTool

from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.tools.render import format_tool_to_openai_tool
def create_openai_tools_agent(
    llm: BaseLanguageModel, tools: Sequence[BaseTool], prompt: ChatPromptTemplate
):
    """Create an agent that uses OpenAI tools.

    Examples:


        .. code-block:: python

            from langchain.agents import (
                create_json_agent,
            )
            from langchain import hub
            from langchain.chat_models import ChatAnthropic
            from langchain.agents import AgentExecutor

            prompt = hub.pull("hwchase17/openai-tools-agent")
            model = ChatOpenAI()
            tools = ...

            agent = create_openai_tools_agent(model, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools)

            agent_executor.invoke({"input": "hi"})

    Args:
        llm: LLM to use as the agent.
        tools: Tools this agent has access to.
        prompt: The prompt to use, must have input keys of `agent_scratchpad`.

    Returns:
        A runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    """
    missing_vars = {"agent_scratchpad"}.difference(
        prompt.input_variables
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    llm_with_stop = llm.bind(tools=[format_tool_to_openai_tool(tool) for tool in tools])

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
            # Give it a default
            chat_history=lambda x: x.get("chat_history", []),
        )
        | prompt
        | llm_with_stop
        | OpenAIToolsAgentOutputParser()
    )
    return agent
