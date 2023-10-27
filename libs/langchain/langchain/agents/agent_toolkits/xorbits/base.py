"""Agent for working with xorbits objects."""
from typing import Any, Dict, List, Optional

from langchain._api import warn_deprecated
from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.xorbits.prompt import (
    NP_PREFIX,
    NP_SUFFIX,
    PD_PREFIX,
    PD_SUFFIX,
)
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.tools.python.tool import PythonAstREPLTool


def create_xorbits_agent(
    llm: BaseLLM,
    data: Any,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = "",
    suffix: str = "",
    input_variables: Optional[List[str]] = None,
    verbose: bool = False,
    return_intermediate_steps: bool = False,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a xorbits agent from an LLM and dataframe."""
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
    try:
        from xorbits import numpy as np
        from xorbits import pandas as pd
    except ImportError:
        raise ImportError(
            "Xorbits package not installed, please install with `pip install xorbits`"
        )

    if not isinstance(data, (pd.DataFrame, np.ndarray)):
        raise ValueError(
            f"Expected Xorbits DataFrame or ndarray object, got {type(data)}"
        )
    if input_variables is None:
        input_variables = ["data", "input", "agent_scratchpad"]
    tools = [PythonAstREPLTool(locals={"data": data})]
    prompt, partial_input = None, None

    if isinstance(data, pd.DataFrame):
        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=PD_PREFIX if prefix == "" else prefix,
            suffix=PD_SUFFIX if suffix == "" else suffix,
            input_variables=input_variables,
        )
        partial_input = str(data.head())
    else:
        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=NP_PREFIX if prefix == "" else prefix,
            suffix=NP_SUFFIX if suffix == "" else suffix,
            input_variables=input_variables,
        )
        partial_input = str(data[: len(data) // 2])
    partial_prompt = prompt.partial(data=partial_input)
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
