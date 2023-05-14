"""SQL agent."""
from typing import Any, Dict, List, Optional

from langchain.agents.agent import AgentExecutor, AgentOutputParser
from langchain.agents.agent_toolkits.spark_sql.prompt import (
    FLEXIBLE_SQL_PREFIX,
    FLEXIBLE_SQL_SUFFIX,
    SQL_PREFIX,
    SQL_SUFFIX,
    SQL_SUFFIX_WITH_MEMORY,
)
from langchain.agents.agent_toolkits.spark_sql.spark_freestyle_parser import (
    SparkSQLFreeStyleOutputParser,
)
from langchain.agents.agent_toolkits.spark_sql.toolkit import (
    SparkFlexibleSQLToolkit,
    SparkSQLToolkit,
)
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.memory import ConversationBufferMemory
from langchain.spark_sql import SparkSQL
from langchain.tools.base import BaseTool


def create_spark_sql_agent(
    llm: BaseLanguageModel,
    db: SparkSQL,
    tools: Optional[List[BaseTool]] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = SQL_PREFIX,
    suffix: str = SQL_SUFFIX,
    input_variables: Optional[List[str]] = None,
    top_k: int = 10,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    verbose: bool = False,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    output_parser: Optional[AgentOutputParser] = None,
    enable_memory: bool = False,
    enable_freestyle: bool = False,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a sql agent from an LLM and tools."""
    if tools is None:
        tools = SparkSQLToolkit(db=db, llm=llm).get_tools()
    if enable_freestyle:
        enable_memory = True
        tools = SparkFlexibleSQLToolkit(db=db, llm=llm).get_tools()
        prefix = FLEXIBLE_SQL_PREFIX
        suffix = FLEXIBLE_SQL_SUFFIX
        output_parser = SparkSQLFreeStyleOutputParser()
    prefix = prefix.format(top_k=top_k)
    memory = None
    if input_variables is None and enable_memory:
        memory = ConversationBufferMemory(memory_key="chat_history")
        input_variables = ["input", "chat_history", "agent_scratchpad"]
        suffix = SQL_SUFFIX_WITH_MEMORY
    # otherwise, None input_variables will get the default during prompt creation.
    agent = ZeroShotAgent.from_llm_and_tools(
        llm=llm,
        tools=tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=input_variables,
        output_parser=output_parser,
        callback_manager=callback_manager,
        **kwargs,
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        memory=memory,
        early_stopping_method=early_stopping_method,
        **(agent_executor_kwargs or {}),
    )
