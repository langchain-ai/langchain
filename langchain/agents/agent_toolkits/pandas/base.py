"""Agent for working with pandas objects."""
from typing import Any, Dict, List, Optional

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.pandas.prompt import (
    MULTI_DF_PREFIX,
    PREFIX,
    SUFFIX_NO_DF,
    SUFFIX_WITH_DF,
    SUFFIX_WITH_MULTI_DF,
)
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.tools.python.tool import PythonAstREPLTool


def create_pandas_dataframe_agent(
    llm: BaseLanguageModel,
    df: Any,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    verbose: bool = False,
    return_intermediate_steps: bool = False,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    include_df_in_prompt: Optional[bool] = True,
    **kwargs: Dict[str, Any],
) -> AgentExecutor:
    """Construct a pandas agent from an LLM and dataframe."""
    try:
        import pandas as pd
    except ImportError:
        raise ValueError(
            "pandas package not found, please install with `pip install pandas`"
        )

    num_dfs = 1
    if isinstance(df, list):
        for item in df:
            if not isinstance(item, pd.DataFrame):
                raise ValueError(f"Expected pandas object, got {type(df)}")
        num_dfs = len(df)
    else:
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected pandas object, got {type(df)}")
    if include_df_in_prompt is not None and suffix is not None:
        raise ValueError("If suffix is specified, include_df_in_prompt should not be.")

    if suffix is not None:
        suffix_to_use = suffix
        if input_variables is None:
            input_variables = ["df_head", "input", "agent_scratchpad"]
        elif num_dfs > 1:
            input_variables = ["dfs_head", "input", "agent_scratchpad"]
    elif include_df_in_prompt:
        if num_dfs > 1:
            suffix_to_use = SUFFIX_WITH_MULTI_DF
            input_variables = ["dfs_head", "input", "agent_scratchpad"]
        else:
            suffix_to_use = SUFFIX_WITH_DF
            input_variables = ["df_head", "input", "agent_scratchpad"]
    else:
        suffix_to_use = SUFFIX_NO_DF
        input_variables = ["input", "agent_scratchpad"]

    if num_dfs > 1:
        input_variables += ["num_dfs"]

    if prefix is None:
        prefix = MULTI_DF_PREFIX if num_dfs > 1 else PREFIX

    if num_dfs > 1:
        df_locals = {}
        for i, dataframe in enumerate(df):
            df_locals[f"df{i + 1}"] = dataframe
        tools = [PythonAstREPLTool(locals=df_locals)]
    else:
        tools = [PythonAstREPLTool(locals={"df": df})]

    prompt = ZeroShotAgent.create_prompt(
        tools, prefix=prefix, suffix=suffix_to_use, input_variables=input_variables
    )

    partial_prompt = prompt.partial()
    if "df_head" in input_variables:
        partial_prompt = partial_prompt.partial(df_head=str(df.head().to_markdown()))
    elif "dfs_head" in input_variables:
        dfs_head = "\n\n".join([d.head().to_markdown() for d in df])
        partial_prompt = partial_prompt.partial(num_dfs=str(num_dfs), dfs_head=dfs_head)
    if "num_dfs" in input_variables:
        partial_prompt = partial_prompt.partial(num_dfs=str(num_dfs))

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
