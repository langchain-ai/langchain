"""Agent for working with pandas objects."""
from typing import Any, Dict, List, Optional

from langchain._api import warn_deprecated
from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.spark.prompt import PREFIX, SUFFIX
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.tools.python.tool import PythonAstREPLTool


def _validate_spark_df(df: Any) -> bool:
    try:
        from pyspark.sql import DataFrame as SparkLocalDataFrame

        return isinstance(df, SparkLocalDataFrame)
    except ImportError:
        return False


def _validate_spark_connect_df(df: Any) -> bool:
    try:
        from pyspark.sql.connect.dataframe import DataFrame as SparkConnectDataFrame

        return isinstance(df, SparkConnectDataFrame)
    except ImportError:
        return False


def create_spark_dataframe_agent(
    llm: BaseLLM,
    df: Any,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = PREFIX,
    suffix: str = SUFFIX,
    input_variables: Optional[List[str]] = None,
    verbose: bool = False,
    return_intermediate_steps: bool = False,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a Spark agent from an LLM and dataframe."""
    warn_deprecated(
        since="0.0.314",
        message=(
            "On 2023-10-27 this module will be be deprecated from langchain, and "
            "will be available from the langchain-experimental package."
            "This code is already available in langchain-experimental."
            "See https://github.com/langchain-ai/langchain/discussions/11680."
        ),
        pending=True,
    )

    if not _validate_spark_df(df) and not _validate_spark_connect_df(df):
        raise ImportError("Spark is not installed. run `pip install pyspark`.")

    if input_variables is None:
        input_variables = ["df", "input", "agent_scratchpad"]
    tools = [PythonAstREPLTool(locals={"df": df})]
    prompt = ZeroShotAgent.create_prompt(
        tools, prefix=prefix, suffix=suffix, input_variables=input_variables
    )
    partial_prompt = prompt.partial(df=str(df.first()))
    llm_chain = LLMChain(
        llm=llm,
        prompt=partial_prompt,
        callback_manager=callback_manager,
    )
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(
        llm_chain=llm_chain,
        allowed_tools=tool_names,
        callback_manager=callback_manager,
        **kwargs,
    )
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
