"""Agent for working with pandas objects."""
import warnings
from typing import Any, List, Optional

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.pandas.prompt import PREFIX, SUFFIX
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import CallbackManager, Callbacks
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.tools.python.tool import PythonAstREPLTool


def _get_callbacks(
    callbacks: Callbacks,
    callbacks_manager: Optional[BaseCallbackManager],
    verbose: bool,
) -> CallbackManager:
    if callbacks_manager is not None and callbacks is not None:
        raise ValueError(
            "Arguments 'callbacks' and "
            "'callbacks_manager' are mutually exclusive. "
            " Please only provide 'callbacks'"
        )
    if callbacks_manager is not None:
        warnings.warn(
            "callback_manager is deprecated. Please use callbacks instead.",
            DeprecationWarning,
        )
        callbacks = callbacks_manager
    return CallbackManager.configure(inheritable_callbacks=callbacks, verbose=verbose)


def create_pandas_dataframe_agent(
    llm: BaseLLM,
    df: Any,
    callbacks: Callbacks = None,
    prefix: str = PREFIX,
    suffix: str = SUFFIX,
    input_variables: Optional[List[str]] = None,
    verbose: bool = False,
    return_intermediate_steps: bool = False,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    callback_manager: Optional[BaseCallbackManager] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a pandas agent from an LLM and dataframe.

    Args:
        llm: The LLM to use
        df: The dataframe to provide to the REPL
        callbacks: The callbacks to use for tracing
        prefix: The prefix to use for the prompt
        suffix: The suffix to use for the prompt
        input_variables: The input variables
        verbose: Whether to print output
        return_intermediate_steps: Whether to return intermediate steps
        max_iterations: The maximum number of iterations before exiting
        max_execution_time: The maximum execution time before exiting
        early_stopping_method: The early stopping method
        callback_manager: (DEPRECATED) Use 'callbacks' instead
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas object, got {type(df)}")
    if input_variables is None:
        input_variables = ["df", "input", "agent_scratchpad"]

    callback_manager_ = _get_callbacks(
        callbacks=callbacks,
        callbacks_manager=callback_manager,
        verbose=verbose,
    )

    tools = [
        PythonAstREPLTool(
            locals={"df": df},
            callbacks=callback_manager_,
        )
    ]
    prompt = ZeroShotAgent.create_prompt(
        tools, prefix=prefix, suffix=suffix, input_variables=input_variables
    )
    partial_prompt = prompt.partial(df=str(df.head().to_markdown()))
    llm_chain = LLMChain(
        llm=llm,
        prompt=partial_prompt,
        callbacks=callback_manager_,
    )
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(
        llm_chain=llm_chain,
        allowed_tools=tool_names,
        callbacks=callback_manager_,
        **kwargs,
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callbacks=callback_manager_,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
    )
