"""Spark SQL agent."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackManager, Callbacks
from langchain_core.language_models import BaseLanguageModel

from langchain_community.agent_toolkits.spark_sql.prompt import SQL_PREFIX, SQL_SUFFIX
from langchain_community.agent_toolkits.spark_sql.toolkit import SparkSQLToolkit

if TYPE_CHECKING:
    from langchain.agents.agent import AgentExecutor


def create_spark_sql_agent(
    llm: BaseLanguageModel,
    toolkit: SparkSQLToolkit,
    callback_manager: Optional[BaseCallbackManager] = None,
    callbacks: Callbacks = None,
    prefix: str = SQL_PREFIX,
    suffix: str = SQL_SUFFIX,
    format_instructions: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    top_k: int = 10,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    verbose: bool = False,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a Spark SQL agent from an LLM and tools."""
    from langchain.agents.agent import AgentExecutor
    from langchain.agents.mrkl.base import ZeroShotAgent
    from langchain.chains.llm import LLMChain

    tools = toolkit.get_tools()
    prefix = prefix.format(top_k=top_k)
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
        callbacks=callbacks,
    )
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        callbacks=callbacks,
        verbose=verbose,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        **(agent_executor_kwargs or {}),
    )
