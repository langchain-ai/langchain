"""Retrieval QA agent."""
from typing import Any, List, Optional

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.retrieval_qa.prompt import (
    PREFIX,
    ROUTER_PREFIX,
    ROUTER_SUFFIX,
    SUFFIX,
)
from langchain.agents.agent_toolkits.retrieval_qa.toolkit import (
    RetrievalQARouterToolkit,
    RetrievalQAToolkit,
)
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.schema.base import BaseLanguageModel


def create_retrieval_qa_agent(
    llm: BaseLanguageModel,
    toolkit: RetrievalQAToolkit,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = PREFIX,
    suffix: str = SUFFIX,
    verbose: bool = False,
    format_instructions: str = FORMAT_INSTRUCTIONS,
    input_variables: Optional[List[str]] = None,
    return_intermediate_steps: bool = False,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct an agent from an LLM and a RetrievalQAToolkit."""
    return _create_agent(
        llm,
        toolkit,
        callback_manager,
        prefix,
        suffix,
        verbose,
        format_instructions,
        input_variables,
        return_intermediate_steps,
        **kwargs,
    )


def create_retrieval_qa_router_agent(
    llm: BaseLanguageModel,
    toolkit: RetrievalQARouterToolkit,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = ROUTER_PREFIX,
    suffix: str = ROUTER_SUFFIX,
    verbose: bool = False,
    format_instructions: str = FORMAT_INSTRUCTIONS,
    input_variables: Optional[List[str]] = None,
    return_intermediate_steps: bool = False,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct an agent from an LLM and RetrievalQAToolkits."""
    return _create_agent(
        llm,
        toolkit,
        callback_manager,
        prefix,
        suffix,
        verbose,
        format_instructions,
        input_variables,
        return_intermediate_steps,
        **kwargs,
    )


def _create_agent(
    llm: BaseLanguageModel,
    toolkit: Any,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = PREFIX,
    suffix: str = SUFFIX,
    verbose: bool = False,
    format_instructions: str = FORMAT_INSTRUCTIONS,
    input_variables: Optional[List[str]] = None,
    return_intermediate_steps: bool = False,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct an agent from an LLM and Toolkit."""
    tools = toolkit.get_tools()
    prompt = ZeroShotAgent.create_prompt(
        tools,
        suffix=suffix,
        prefix=prefix,
        format_instructions=format_instructions,
        input_variables=input_variables,
    )
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        callback_manager=callback_manager,
    )
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
    )
