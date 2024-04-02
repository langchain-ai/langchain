"""
    Cohere multi-hop agent enables multiple tools to be used in sequence to complete a
    task.

    This agent uses a multi hop prompt by Cohere, which is experimental and subject
    to change. The latest prompt can be used by upgrading the langchain-cohere package.
"""
from typing import Sequence

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool

from langchain_cohere.react_multi_hop.parsing import (
    CohereToolsReactAgentOutputParser,
)
from langchain_cohere.react_multi_hop.prompt import (
    multi_hop_prompt,
)


def create_cohere_react_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: ChatPromptTemplate,
) -> Runnable:
    """
    Create an agent that enables multiple tools to be used in sequence to complete
    a task.

    Args:
        llm: The ChatCohere LLM instance to use.
        tools: Tools this agent has access to.
        prompt: The prompt to use.

    Returns:
        A Runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does and returns an AgentAction or
        AgentFinish.

    Example:
        . code-block:: python
            from langchain.agents import AgentExecutor
            from langchain.prompts import ChatPromptTemplate
            from langchain_cohere import ChatCohere, create_cohere_react_agent

            prompt = ChatPromptTemplate.from_template("{input}")
            tools = []  # Populate this with a list of tools you would like to use.

            llm = ChatCohere()

            agent = create_cohere_react_agent(
                llm,
                tools,
                prompt
            )
            agent_executor = AgentExecutor(agent=agent, tools=tools)

            agent_executor.invoke({
                "input": "In what year was the company that was founded as Sound of Music added to the S&P 500?",
            })
    """  # noqa: E501
    agent = (
        RunnablePassthrough.assign(
            # agent_scratchpad isn't used in this chain, but added here for
            # interoperability with other chains that may require it.
            agent_scratchpad=lambda _: [],
        )
        | multi_hop_prompt(tools=tools, prompt=prompt)
        | llm.bind(stop=["\nObservation:"], raw_prompting=True)
        | CohereToolsReactAgentOutputParser()
    )
    return agent
