"""VectorStore agent."""
from typing import Any, Dict, Optional

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.vectorstore.prompt import PREFIX, ROUTER_PREFIX
from langchain.agents.agent_toolkits.vectorstore.toolkit import (
    VectorStoreRouterToolkit,
    VectorStoreToolkit,
)
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.schema.language_model import BaseLanguageModel


def create_vectorstore_agent(
    llm: BaseLanguageModel,
    toolkit: VectorStoreToolkit,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = PREFIX,
    verbose: bool = False,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Dict[str, Any],
) -> AgentExecutor:
    """Construct a VectorStore agent from an LLM and tools.

    Args:
        llm (BaseLanguageModel): LLM that will be used by the agent
        toolkit (VectorStoreToolkit): Set of tools for the agent
        callback_manager (Optional[BaseCallbackManager], optional): Object to handle the callback [ Defaults to None. ]
        prefix (str, optional): The prefix prompt for the agent. If not provided uses default PREFIX.
        verbose (bool, optional): If you want to see the content of the scratchpad. [ Defaults to False ]
        agent_executor_kwargs (Optional[Dict[str, Any]], optional): If there is any other parameter you want to send to the agent. [ Defaults to None ]
        **kwargs: Additional named parameters to pass to the ZeroShotAgent.

    Returns:
        AgentExecutor: Returns a callable AgentExecutor object. Either you can call it or use run method with the query to get the response
    """  # noqa: E501
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
    """Construct a VectorStore router agent from an LLM and tools.

    Args:
        llm (BaseLanguageModel): LLM that will be used by the agent
        toolkit (VectorStoreRouterToolkit): Set of tools for the agent which have routing capability with multiple vector stores
        callback_manager (Optional[BaseCallbackManager], optional): Object to handle the callback [ Defaults to None. ]
        prefix (str, optional): The prefix prompt for the router agent. If not provided uses default ROUTER_PREFIX.
        verbose (bool, optional): If you want to see the content of the scratchpad. [ Defaults to False ]
        agent_executor_kwargs (Optional[Dict[str, Any]], optional): If there is any other parameter you want to send to the agent. [ Defaults to None ]
        **kwargs: Additional named parameters to pass to the ZeroShotAgent.

    Returns:
        AgentExecutor: Returns a callable AgentExecutor object. Either you can call it or use run method with the query to get the response.
    """  # noqa: E501
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
