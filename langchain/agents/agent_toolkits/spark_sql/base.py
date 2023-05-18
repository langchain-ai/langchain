"""Spark SQL agent."""
from typing import Any, Dict, List, Optional

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.spark_sql.prompt import SQL_PREFIX, SQL_SUFFIX
from langchain.agents.agent_toolkits.spark_sql.toolkit import SparkSQLToolkit
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain


def create_spark_sql_agent(
    llm: BaseLanguageModel,
    toolkit: SparkSQLToolkit,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = SQL_PREFIX,
    suffix: str = SQL_SUFFIX,
    format_instructions: str = FORMAT_INSTRUCTIONS,
    input_variables: Optional[List[str]] = None,
    top_k: int = 10,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    verbose: bool = False,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Dict[str, Any],
) -> AgentExecutor:
    """Construct a sql agent from an LLM and tools."""
    tools = toolkit.get_tools()
    prefix = prefix.format(top_k=top_k)
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
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        **(agent_executor_kwargs or {}),
    )
