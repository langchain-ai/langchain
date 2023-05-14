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
    allow_freestyle: bool = False,
    **kwargs: Dict[str, Any],
) -> AgentExecutor:
    """Construct a sql agent from an LLM and tools."""
    if tools is None:
        tools = SparkSQLToolkit(db=db, llm=llm).get_tools()
    if allow_freestyle:
        tools = SparkFlexibleSQLToolkit(db=db, llm=llm).get_tools()
        prefix = FLEXIBLE_SQL_PREFIX
        suffix = FLEXIBLE_SQL_SUFFIX
        output_parser = SparkSQLFreeStyleOutputParser()
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
