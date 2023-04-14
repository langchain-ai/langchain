"""Retrieval QA agent."""
from typing import Any, Optional

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.retrieval_qa.prompt import PREFIX, ROUTER_PREFIX
from langchain.agents.agent_toolkits.retrieval_qa.toolkit import (
    RetrievalQARouterToolkit,
    RetrievalQAToolkit,
)
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.schema.base import BaseLanguageModel


def create_retrieval_qa_agent(
    llm: BaseLanguageModel,
    toolkit: RetrievalQAToolkit,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = PREFIX,
    verbose: bool = False,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct an agent from an LLM and a RetrievalQAToolkit."""
    return _create_agent(
        llm,
        toolkit,
        callback_manager,
        prefix,
        verbose,
        **kwargs,
    )


def create_retrieval_qa_router_agent(
    llm: BaseLanguageModel,
    toolkit: RetrievalQARouterToolkit,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = ROUTER_PREFIX,
    verbose: bool = False,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct an agent from an LLM and RetrievalQAToolkits."""
    return _create_agent(
        llm,
        toolkit,
        callback_manager,
        prefix,
        verbose,
        **kwargs,
    )


def _create_agent(
    llm: BaseLanguageModel,
    toolkit: Any,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = PREFIX,
    verbose: bool = False,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct an agent from an LLM and Toolkit."""
    tools = toolkit.get_tools()
    prompt = ZeroShotAgent.create_prompt(tools, prefix=prefix)
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        callback_manager=callback_manager,
    )
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose)
