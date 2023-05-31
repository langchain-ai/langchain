"""Agent for working with pandas objects."""
from typing import Any, Dict, List, Optional, Tuple

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
from langchain.prompts.base import BasePromptTemplate
from langchain.tools.python.tool import PythonAstREPLTool


def _get_multi_prompt(
    dfs: List[Any],
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    include_df_in_prompt: Optional[bool] = True,
) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
    num_dfs = len(dfs)
    if suffix is not None:
        suffix_to_use = suffix
        include_dfs_head = True
    elif include_df_in_prompt:
        suffix_to_use = SUFFIX_WITH_MULTI_DF
        include_dfs_head = True
    else:
        suffix_to_use = SUFFIX_NO_DF
        include_dfs_head = False
    if input_variables is None:
        input_variables = ["input", "agent_scratchpad", "num_dfs"]
        if include_dfs_head:
            input_variables += ["dfs_head"]

    if prefix is None:
        prefix = MULTI_DF_PREFIX

    df_locals = {}
    for i, dataframe in enumerate(dfs):
        df_locals[f"df{i + 1}"] = dataframe
    tools = [PythonAstREPLTool(locals=df_locals)]

    prompt = ZeroShotAgent.create_prompt(
        tools, prefix=prefix, suffix=suffix_to_use, input_variables=input_variables
    )

    partial_prompt = prompt.partial()
    if "dfs_head" in input_variables:
        dfs_head = "\n\n".join([d.head().to_markdown() for d in dfs])
        partial_prompt = partial_prompt.partial(num_dfs=str(num_dfs), dfs_head=dfs_head)
    if "num_dfs" in input_variables:
        partial_prompt = partial_prompt.partial(num_dfs=str(num_dfs))
    return partial_prompt, tools


def _get_single_prompt(
    df: Any,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    include_df_in_prompt: Optional[bool] = True,
) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
    if suffix is not None:
        suffix_to_use = suffix
        include_df_head = True
    elif include_df_in_prompt:
        suffix_to_use = SUFFIX_WITH_DF
        include_df_head = True
    else:
        suffix_to_use = SUFFIX_NO_DF
        include_df_head = False

    if input_variables is None:
        input_variables = ["input", "agent_scratchpad"]
        if include_df_head:
            input_variables += ["df_head"]

    if prefix is None:
        prefix = PREFIX

    tools = [PythonAstREPLTool(locals={"df": df})]

    prompt = ZeroShotAgent.create_prompt(
        tools, prefix=prefix, suffix=suffix_to_use, input_variables=input_variables
    )

    partial_prompt = prompt.partial()
    if "df_head" in input_variables:
        partial_prompt = partial_prompt.partial(df_head=str(df.head().to_markdown()))
    return partial_prompt, tools


def _get_prompt_and_tools(
    df: Any,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    include_df_in_prompt: Optional[bool] = True,
) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
    try:
        import pandas as pd
    except ImportError:
        raise ValueError(
            "pandas package not found, please install with `pip install pandas`"
        )

    if include_df_in_prompt is not None and suffix is not None:
        raise ValueError("If suffix is specified, include_df_in_prompt should not be.")

    if isinstance(df, list):
        for item in df:
            if not isinstance(item, pd.DataFrame):
                raise ValueError(f"Expected pandas object, got {type(df)}")
        return _get_multi_prompt(
            df,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
            include_df_in_prompt=include_df_in_prompt,
        )
    else:
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected pandas object, got {type(df)}")
        return _get_single_prompt(
            df,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
            include_df_in_prompt=include_df_in_prompt,
        )


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

    prompt, tools = _get_prompt_and_tools(
        df,
        prefix=prefix,
        suffix=suffix,
        input_variables=input_variables,
        include_df_in_prompt=include_df_in_prompt,
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
