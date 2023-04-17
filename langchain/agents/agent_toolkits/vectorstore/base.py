"""VectorStore agent."""
from typing import Any, List, Optional

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.retrieval_qa.base import _create_agent
from langchain.agents.agent_toolkits.vectorstore.prompt import (
    PREFIX,
    ROUTER_PREFIX,
    ROUTER_SUFFIX,
    SUFFIX,
)
from langchain.agents.agent_toolkits.vectorstore.toolkit import (
    VectorStoreRouterToolkit,
    VectorStoreToolkit,
)
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.callbacks.base import BaseCallbackManager
from langchain.schema.base import BaseLanguageModel


def create_vectorstore_agent(
    llm: BaseLanguageModel,
    toolkit: VectorStoreToolkit,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = PREFIX,
    suffix: str = SUFFIX,
    verbose: bool = False,
    format_instructions: str = FORMAT_INSTRUCTIONS,
    input_variables: Optional[List[str]] = None,
    return_intermediate_steps: bool = False,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct an agent from an LLM and VectorStoreToolkit."""
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


def create_vectorstore_router_agent(
    llm: BaseLanguageModel,
    toolkit: VectorStoreRouterToolkit,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = ROUTER_PREFIX,
    suffix: str = ROUTER_SUFFIX,
    verbose: bool = False,
    format_instructions: str = FORMAT_INSTRUCTIONS,
    input_variables: Optional[List[str]] = None,
    return_intermediate_steps: bool = False,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct an agent from an LLM and VectorStoreRouterToolkit."""
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
