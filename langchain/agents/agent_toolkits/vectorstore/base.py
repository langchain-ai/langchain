"""VectorStore agent."""
from typing import Any, Dict, Optional

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.vectorstore.prompt import PREFIX, ROUTER_PREFIX
from langchain.agents.agent_toolkits.vectorstore.toolkit import (
    VectorStoreRouterToolkit,
    VectorStoreToolkit,
)
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain


def create_vectorstore_agent(
    llm: BaseLanguageModel,
    toolkit: VectorStoreToolkit,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = PREFIX,
    verbose: bool = False,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Dict[str, Any],
) -> AgentExecutor:
    """Construct a vectorstore agent from an LLM and tools."""
    tools = toolkit.get_tools()
    prompt = ZeroShotAgent.create_prompt(tools, prefix=prefix)
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
        callback_manager=callback_manager,
        verbose=verbose,
        **(agent_executor_kwargs or {}),
    )


def create_vectorstore_router_agent(
    llm: BaseLanguageModel,
    toolkit: VectorStoreRouterToolkit,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = ROUTER_PREFIX,
    verbose: bool = False,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Dict[str, Any],
) -> AgentExecutor:
    """Construct a vectorstore router agent from an LLM and tools."""
    tools = toolkit.get_tools()
    prompt = ZeroShotAgent.create_prompt(tools, prefix=prefix)
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
        callback_manager=callback_manager,
        verbose=verbose,
        **(agent_executor_kwargs or {}),
    )
