"""OpenAPI spec agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel

from langchain_community.agent_toolkits.openapi.prompt import (
    OPENAPI_PREFIX,
    OPENAPI_SUFFIX,
)
from langchain_community.agent_toolkits.openapi.toolkit import OpenAPIToolkit

if TYPE_CHECKING:
    from langchain.agents.agent import AgentExecutor


def create_openapi_agent(
    llm: BaseLanguageModel,
    toolkit: OpenAPIToolkit,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = OPENAPI_PREFIX,
    suffix: str = OPENAPI_SUFFIX,
    format_instructions: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    verbose: bool = False,
    return_intermediate_steps: bool = False,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct an OpenAPI agent from an LLM and tools.

    *Security Note*: When creating an OpenAPI agent, check the permissions
        and capabilities of the underlying toolkit.

        For example, if the default implementation of OpenAPIToolkit
        uses the RequestsToolkit which contains tools to make arbitrary
        network requests against any URL (e.g., GET, POST, PATCH, PUT, DELETE),

        Control access to who can submit issue requests using this toolkit and
        what network access it has.

        See https://python.langchain.com/docs/security for more information.

    Args:
        llm: The language model to use.
        toolkit: The OpenAPI toolkit.
        callback_manager: Optional. The callback manager. Default is None.
        prefix: Optional. The prefix for the prompt. Default is OPENAPI_PREFIX.
        suffix: Optional. The suffix for the prompt. Default is OPENAPI_SUFFIX.
        format_instructions: Optional. The format instructions for the prompt.
            Default is None.
        input_variables: Optional. The input variables for the prompt. Default is None.
        max_iterations: Optional. The maximum number of iterations. Default is 15.
        max_execution_time: Optional. The maximum execution time. Default is None.
        early_stopping_method: Optional. The early stopping method. Default is "force".
        verbose: Optional. Whether to print verbose output. Default is False.
        return_intermediate_steps: Optional. Whether to return intermediate steps.
            Default is False.
        agent_executor_kwargs: Optional. Additional keyword arguments
            for the agent executor.
        **kwargs: Additional arguments.

    Returns:
        The agent executor.
    """
    from langchain.agents.agent import AgentExecutor
    from langchain.agents.mrkl.base import ZeroShotAgent
    from langchain.chains.llm import LLMChain

    tools = toolkit.get_tools()
    prompt_params = (
        {"format_instructions": format_instructions}
        if format_instructions is not None
        else {}
    )
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=input_variables,
        **prompt_params,
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
        callback_manager=callback_manager,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        **(agent_executor_kwargs or {}),
    )
