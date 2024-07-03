from typing import Sequence

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.runnables.base import RunnableBindingBase
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from langchain_glm.agents.format_scratchpad.all_tools import (
    format_to_zhipuai_all_tool_messages,
)
from langchain_glm.agents.output_parsers import ZhipuAiALLToolsAgentOutputParser


def create_zhipuai_tools_agent(
    prompt: ChatPromptTemplate,
    llm_with_all_tools: RunnableBindingBase = None,
) -> Runnable:
    """Create an agent that uses OpenAI tools.

    Args:
        prompt: The prompt to use. See Prompt section below for more on the expected
            input variables.
        llm_with_all_tools: Optional. If provided, this will be used as the LLM with all
            tools bound to it. If not provided, the tools will be bound to the LLM
            provided.

    Returns:
        A Runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    Example:

        .. code-block:: python

            from langchain import hub
            from langchain_community.chat_models import ChatOpenAI
            from langchain.agents import AgentExecutor

            from langchain_glm.agents.all_tools_bind import create_zhipuai_tools_agent

            prompt = hub.pull("hwchase17/openai-tools-agent")
            model = ChatOpenAI()
            llm_with_all_tools = model.bind ...

            agent = create_zhipuai_tools_agent(llm_with_all_tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools)

            agent_executor.invoke({"input": "hi"})

            # Using with chat history
            from langchain_core.messages import AIMessage, HumanMessage
            agent_executor.invoke(
                {
                    "input": "what's my name?",
                    "chat_history": [
                        HumanMessage(content="hi! my name is bob"),
                        AIMessage(content="Hello Bob! How can I assist you today?"),
                    ],
                }
            )

    Prompt:

        The agent prompt must have an `agent_scratchpad` key that is a
            ``MessagesPlaceholder``. Intermediate agent actions and tool output
            messages will be passed in here.

        Here's an example:

        .. code-block:: python

            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant"),
                    MessagesPlaceholder("chat_history", optional=True),
                    ("human", "{input}"),
                    MessagesPlaceholder("agent_scratchpad"),
                ]
            )
    """
    missing_vars = {"agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables)
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_zhipuai_all_tool_messages(
                x["intermediate_steps"]
            )
        )
        | prompt
        | llm_with_all_tools
        | ZhipuAiALLToolsAgentOutputParser()
    )

    return agent
