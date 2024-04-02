"""
    Cohere multi-hop agent enables multiple tools to be used in sequence to complete
    your task.

    This agent uses a multi hop prompt by Cohere, which is experimental and subject
    to change. The latest prompt can be used by upgrading the langchain-cohere package.
"""
from typing import List

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
    tools: List[BaseTool],
    prompt: ChatPromptTemplate,
) -> Runnable:
    agent = (
        RunnablePassthrough.assign(
            # Handled in the multi_hop_prompt
            agent_scratchpad=lambda _: [],
        )
        | multi_hop_prompt(tools=tools, prompt=prompt)
        | llm.bind(stop=["\nObservation:"], raw_prompting=True)
        | CohereToolsReactAgentOutputParser()
    )
    return agent
