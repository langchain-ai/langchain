"""SQL agent."""
from typing import Any, Dict, List, Optional

from langchain.agents.agent import AgentExecutor, AgentOutputParser
from langchain.agents.agent_toolkits.spark_sql.prompt import (
    FLEXIBLE_SQL_PREFIX,
    FLEXIBLE_SQL_SUFFIX,
    SQL_PREFIX,
    SQL_SUFFIX,
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
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.spark_sql import SparkSQL


def create_spark_analytics_agent_verified(
    schema: str,
    allow_freestyle: bool = False,
) -> AgentExecutor:
    """
    Construct a spark analytics agent with ChatGPT-4,
    which is verified during development.
    """
    spark_sql = SparkSQL(schema=schema)
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    toolkit = SparkSQLToolkit(db=spark_sql, llm=llm)
    prefix: str = SQL_PREFIX
    suffix: str = SQL_SUFFIX
    output_parser = None
    if allow_freestyle:
        toolkit = SparkFlexibleSQLToolkit(db=spark_sql, llm=llm)
        prefix = FLEXIBLE_SQL_PREFIX
        suffix = FLEXIBLE_SQL_SUFFIX
        output_parser = SparkSQLFreeStyleOutputParser()
    return create_spark_sql_agent(
        llm=llm,
        toolkit=toolkit,
        enable_memory=True,
        prefix=prefix,
        suffix=suffix,
        output_parser=output_parser,
        verbose=True,
    )


def create_spark_sql_agent(
    llm: BaseLanguageModel,
    toolkit: SparkSQLToolkit,
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
    **kwargs: Dict[str, Any],
) -> AgentExecutor:
    """Construct a sql agent from an LLM and tools."""
    tools = toolkit.get_tools()
    prefix = prefix.format(top_k=top_k)
    memory = None
    if input_variables is None and enable_memory:
        memory = ConversationBufferMemory(memory_key="chat_history")
        input_variables = ["input", "chat_history", "agent_scratchpad"]
    # otherwise, None input_variables will get the default during prompt creation.
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=input_variables,
    )
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        callback_manager=callback_manager,
    )
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(
        llm_chain=llm_chain,
        allowed_tools=tool_names,
        output_parser=output_parser,
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
