"""Module implements an agent that uses GigaChat's APIs function enabled API."""
from typing import Sequence

from langchain_community.tools.render import format_tool_to_gigachat_function
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool

from langchain.agents.format_scratchpad.gigachat_functions import (
    format_to_gigachat_function_messages,
)
from langchain.agents.output_parsers.gigachat_functions import (
    GigaChatFunctionsAgentOutputParser,
)


def create_gigachat_agent_prompt() -> ChatPromptTemplate:
    messages = [
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
    return ChatPromptTemplate.from_messages(messages=messages)


def create_gigachat_functions_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: ChatPromptTemplate = create_gigachat_agent_prompt(),
) -> Runnable:
    """Create an agent that uses GigaChat function calling.

    Args:
        llm: LLM to use as the agent. Should work with OpenAI function calling,
            so either be an GigaChat model that supports that or a wrapper of
            a different model that adds in equivalent support.
        tools: Tools this agent has access to.
        prompt: The prompt to use, must have input key `agent_scratchpad`, which will
            contain agent action and tool output messages.

    Returns:
        A Runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    TODO: Support gigachat
    Example (old version from openAI):

        Creating an agent with no memory

        .. code-block:: python

            from langchain_community.chat_models import ChatOpenAI
            from langchain.agents import AgentExecutor, create_openai_functions_agent
            from langchain import hub

            prompt = hub.pull("hwchase17/openai-functions-agent")
            model = ChatOpenAI()
            tools = ...

            agent = create_gigachat_functions_agent(model, tools, prompt)
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

    Creating prompt example:

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
    if "agent_scratchpad" not in prompt.input_variables:
        raise ValueError(
            "Prompt must have input variable `agent_scratchpad`, but wasn't found. "
            f"Found {prompt.input_variables} instead."
        )
    llm_with_tools = llm.bind(
        functions=[format_tool_to_gigachat_function(t) for t in tools]
    )
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_gigachat_function_messages(
                x["intermediate_steps"]
            )
        )
        | prompt
        | llm_with_tools
        | GigaChatFunctionsAgentOutputParser()
    )
    return agent
