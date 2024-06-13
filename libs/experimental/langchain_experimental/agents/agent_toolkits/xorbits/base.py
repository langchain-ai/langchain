"""Agent for working with xorbits objects."""
from typing import Any, Dict, List, Optional

from langchain.agents.agent import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.chains.llm import LLMChain
from langchain_core.callbacks.base import BaseCallbackManager
from langchain_core.language_models import BaseLLM

from langchain_experimental.agents.agent_toolkits.xorbits.prompt import (
    NP_PREFIX,
    NP_SUFFIX,
    PD_PREFIX,
    PD_SUFFIX,
)
from langchain_experimental.tools.python.tool import PythonAstREPLTool


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
    allow_dangerous_code: bool = False,
    **kwargs: Dict[str, Any],
) -> AgentExecutor:
    """Construct a xorbits agent from an LLM and dataframe.

    Security Notice:
        This agent relies on access to a python repl tool which can execute
        arbitrary code. This can be dangerous and requires a specially sandboxed
        environment to be safely used. Failure to run this code in a properly
        sandboxed environment can lead to arbitrary code execution vulnerabilities,
        which can lead to data breaches, data loss, or other security incidents.

        Do not use this code with untrusted inputs, with elevated permissions,
        or without consulting your security team about proper sandboxing!

        You must opt in to use this functionality by setting allow_dangerous_code=True.

    Args:
        allow_dangerous_code: bool, default False
            This agent relies on access to a python repl tool which can execute
            arbitrary code. This can be dangerous and requires a specially sandboxed
            environment to be safely used.
            Failure to properly sandbox this class can lead to arbitrary code execution
            vulnerabilities, which can lead to data breaches, data loss, or
            other security incidents.
            You must opt in to use this functionality by setting
            allow_dangerous_code=True.
    """
    if not allow_dangerous_code:
        raise ValueError(
            "This agent relies on access to a python repl tool which can execute "
            "arbitrary code. This can be dangerous and requires a specially sandboxed "
            "environment to be safely used. Please read the security notice in the "
            "doc-string of this function. You must opt-in to use this functionality "
            "by setting allow_dangerous_code=True."
            "For general security guidelines, please see: "
            "https://python.langchain.com/v0.2/docs/security/"
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
