"""OpenAPI spec agent."""
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.openapi.prompt import (
    OPENAPI_PREFIX,
    OPENAPI_SUFFIX,
)
from langchain.agents.agent_toolkits.openapi.toolkit import OpenAPIToolkit
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain


def create_openapi_agent(
    llm: BaseLanguageModel,
    toolkit: OpenAPIToolkit,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = OPENAPI_PREFIX,
    suffix: str = OPENAPI_SUFFIX,
    format_instructions: str = FORMAT_INSTRUCTIONS,
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
    """
    tools = toolkit.get_tools()
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
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
        callback_manager=callback_manager,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        **(agent_executor_kwargs or {}),
    )
