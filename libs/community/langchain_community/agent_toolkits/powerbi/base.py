"""Power BI agent."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel

from langchain_community.agent_toolkits.powerbi.prompt import (
    POWERBI_PREFIX,
    POWERBI_SUFFIX,
)
from langchain_community.agent_toolkits.powerbi.toolkit import PowerBIToolkit
from langchain_community.utilities.powerbi import PowerBIDataset

if TYPE_CHECKING:
    from langchain.agents import AgentExecutor


def create_pbi_agent(
    llm: BaseLanguageModel,
    toolkit: Optional[PowerBIToolkit] = None,
    powerbi: Optional[PowerBIDataset] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = POWERBI_PREFIX,
    suffix: str = POWERBI_SUFFIX,
    format_instructions: Optional[str] = None,
    examples: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    top_k: int = 10,
    verbose: bool = False,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a Power BI agent from an LLM and tools."""
    from langchain.agents import AgentExecutor
    from langchain.agents.mrkl.base import ZeroShotAgent
    from langchain.chains.llm import LLMChain

    if toolkit is None:
        if powerbi is None:
            raise ValueError("Must provide either a toolkit or powerbi dataset")
        toolkit = PowerBIToolkit(powerbi=powerbi, llm=llm, examples=examples)
    tools = toolkit.get_tools()
    tables = powerbi.table_names if powerbi else toolkit.powerbi.table_names
    prompt_params = (
        {"format_instructions": format_instructions}
        if format_instructions is not None
        else {}
    )
    agent = ZeroShotAgent(
        llm_chain=LLMChain(
            llm=llm,
            prompt=ZeroShotAgent.create_prompt(
                tools,
                prefix=prefix.format(top_k=top_k).format(tables=tables),
                suffix=suffix,
                input_variables=input_variables,
                **prompt_params,
            ),
            callback_manager=callback_manager,  # type: ignore
            verbose=verbose,
        ),
        allowed_tools=[tool.name for tool in tools],
        **kwargs,
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        **(agent_executor_kwargs or {}),
    )
