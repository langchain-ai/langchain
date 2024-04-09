"""Agent for working with pandas objects."""
import warnings
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from langchain.agents import AgentType, create_openai_tools_agent, create_react_agent
from langchain.agents.agent import (
    AgentExecutor,
    BaseMultiActionAgent,
    BaseSingleActionAgent,
    RunnableAgent,
    RunnableMultiActionAgent,
)
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.agents.openai_functions_agent.base import (
    OpenAIFunctionsAgent,
    create_openai_functions_agent,
)
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate,
)
from langchain_core.tools import BaseTool
from langchain_core.utils.interactive_env import is_interactive_env

from langchain_experimental.agents.agent_toolkits.pandas.prompt import (
    FUNCTIONS_WITH_DF,
    FUNCTIONS_WITH_MULTI_DF,
    MULTI_DF_PREFIX,
    MULTI_DF_PREFIX_FUNCTIONS,
    PREFIX,
    PREFIX_FUNCTIONS,
    SUFFIX_NO_DF,
    SUFFIX_WITH_DF,
    SUFFIX_WITH_MULTI_DF,
)
from langchain_experimental.tools.python.tool import PythonAstREPLTool


def _get_multi_prompt(
    dfs: List[Any],
    *,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
) -> BasePromptTemplate:
    if suffix is not None:
        suffix_to_use = suffix
    elif include_df_in_prompt:
        suffix_to_use = SUFFIX_WITH_MULTI_DF
    else:
        suffix_to_use = SUFFIX_NO_DF
    prefix = prefix if prefix is not None else MULTI_DF_PREFIX

    template = "\n\n".join([prefix, "{tools}", FORMAT_INSTRUCTIONS, suffix_to_use])
    prompt = PromptTemplate.from_template(template)
    partial_prompt = prompt.partial()
    if "dfs_head" in partial_prompt.input_variables:
        dfs_head = "\n\n".join([d.head(number_of_head_rows).to_markdown() for d in dfs])
        partial_prompt = partial_prompt.partial(dfs_head=dfs_head)
    if "num_dfs" in partial_prompt.input_variables:
        partial_prompt = partial_prompt.partial(num_dfs=str(len(dfs)))
    return partial_prompt


def _get_single_prompt(
    df: Any,
    *,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
) -> BasePromptTemplate:
    if suffix is not None:
        suffix_to_use = suffix
    elif include_df_in_prompt:
        suffix_to_use = SUFFIX_WITH_DF
    else:
        suffix_to_use = SUFFIX_NO_DF
    prefix = prefix if prefix is not None else PREFIX

    template = "\n\n".join([prefix, "{tools}", FORMAT_INSTRUCTIONS, suffix_to_use])
    prompt = PromptTemplate.from_template(template)

    partial_prompt = prompt.partial()
    if "df_head" in partial_prompt.input_variables:
        df_head = str(df.head(number_of_head_rows).to_markdown())
        partial_prompt = partial_prompt.partial(df_head=df_head)
    return partial_prompt


def _get_prompt(df: Any, **kwargs: Any) -> BasePromptTemplate:
    return (
        _get_multi_prompt(df, **kwargs)
        if isinstance(df, list)
        else _get_single_prompt(df, **kwargs)
    )


def _get_functions_single_prompt(
    df: Any,
    *,
    prefix: Optional[str] = None,
    suffix: str = "",
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
) -> ChatPromptTemplate:
    if include_df_in_prompt:
        df_head = str(df.head(number_of_head_rows).to_markdown())
        suffix = (suffix or FUNCTIONS_WITH_DF).format(df_head=df_head)
    prefix = prefix if prefix is not None else PREFIX_FUNCTIONS
    system_message = SystemMessage(content=prefix + suffix)
    prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
    return prompt


def _get_functions_multi_prompt(
    dfs: Any,
    *,
    prefix: str = "",
    suffix: str = "",
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
) -> ChatPromptTemplate:
    if include_df_in_prompt:
        dfs_head = "\n\n".join([d.head(number_of_head_rows).to_markdown() for d in dfs])
        suffix = (suffix or FUNCTIONS_WITH_MULTI_DF).format(dfs_head=dfs_head)
    prefix = (prefix or MULTI_DF_PREFIX_FUNCTIONS).format(num_dfs=str(len(dfs)))
    system_message = SystemMessage(content=prefix + suffix)
    prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
    return prompt


def _get_functions_prompt(df: Any, **kwargs: Any) -> ChatPromptTemplate:
    return (
        _get_functions_multi_prompt(df, **kwargs)
        if isinstance(df, list)
        else _get_functions_single_prompt(df, **kwargs)
    )


def create_pandas_dataframe_agent(
    llm: LanguageModelLike,
    df: Any,
    agent_type: Union[
        AgentType, Literal["openai-tools"]
    ] = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
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
    number_of_head_rows: int = 5,
    extra_tools: Sequence[BaseTool] = (),
    engine: Literal["pandas", "modin"] = "pandas",
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a Pandas agent from an LLM and dataframe(s).

    Args:
        llm: Language model to use for the agent.
        df: Pandas dataframe or list of Pandas dataframes.
        agent_type: One of "openai-tools", "openai-functions", or
            "zero-shot-react-description". Defaults to "zero-shot-react-description".
            "openai-tools" is recommended over "openai-functions".
        callback_manager: DEPRECATED. Pass "callbacks" key into 'agent_executor_kwargs'
            instead to pass constructor callbacks to AgentExecutor.
        prefix: Prompt prefix string.
        suffix: Prompt suffix string.
        input_variables: DEPRECATED. Input variables automatically inferred from
            constructed prompt.
        verbose: AgentExecutor verbosity.
        return_intermediate_steps: Passed to AgentExecutor init.
        max_iterations: Passed to AgentExecutor init.
        max_execution_time: Passed to AgentExecutor init.
        early_stopping_method: Passed to AgentExecutor init.
        agent_executor_kwargs: Arbitrary additional AgentExecutor args.
        include_df_in_prompt: Whether to include the first number_of_head_rows in the
            prompt. Must be None if suffix is not None.
        number_of_head_rows: Number of initial rows to include in prompt if
            include_df_in_prompt is True.
        extra_tools: Additional tools to give to agent on top of a PythonAstREPLTool.
        engine: One of "modin" or "pandas". Defaults to "pandas".
        **kwargs: DEPRECATED. Not used, kept for backwards compatibility.

    Returns:
        An AgentExecutor with the specified agent_type agent and access to
        a PythonAstREPLTool with the DataFrame(s) and any user-provided extra_tools.

    Example:
        .. code-block:: python

            from langchain_openai import ChatOpenAI
            from langchain_experimental.agents import create_pandas_dataframe_agent
            import pandas as pd

            df = pd.read_csv("titanic.csv")
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            agent_executor = create_pandas_dataframe_agent(
                llm,
                df,
                agent_type="openai-tools",
                verbose=True
            )

    """  # noqa: E501
    try:
        if engine == "modin":
            import modin.pandas as pd
        elif engine == "pandas":
            import pandas as pd
        else:
            raise ValueError(
                f"Unsupported engine {engine}. It must be one of 'modin' or 'pandas'."
            )
    except ImportError as e:
        raise ImportError(
            f"`{engine}` package not found, please install with `pip install {engine}`"
        ) from e

    if is_interactive_env():
        pd.set_option("display.max_columns", None)

    for _df in df if isinstance(df, list) else [df]:
        if not isinstance(_df, pd.DataFrame):
            raise ValueError(f"Expected pandas DataFrame, got {type(_df)}")

    if input_variables:
        kwargs = kwargs or {}
        kwargs["input_variables"] = input_variables
    if kwargs:
        warnings.warn(
            f"Received additional kwargs {kwargs} which are no longer supported."
        )

    df_locals = {}
    if isinstance(df, list):
        for i, dataframe in enumerate(df):
            df_locals[f"df{i + 1}"] = dataframe
    else:
        df_locals["df"] = df
    tools = [PythonAstREPLTool(locals=df_locals)] + list(extra_tools)

    if agent_type == AgentType.ZERO_SHOT_REACT_DESCRIPTION:
        if include_df_in_prompt is not None and suffix is not None:
            raise ValueError(
                "If suffix is specified, include_df_in_prompt should not be."
            )
        prompt = _get_prompt(
            df,
            prefix=prefix,
            suffix=suffix,
            include_df_in_prompt=include_df_in_prompt,
            number_of_head_rows=number_of_head_rows,
        )
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent] = RunnableAgent(
            runnable=create_react_agent(llm, tools, prompt),  # type: ignore
            input_keys_arg=["input"],
            return_keys_arg=["output"],
        )
    elif agent_type in (AgentType.OPENAI_FUNCTIONS, "openai-tools"):
        prompt = _get_functions_prompt(
            df,
            prefix=prefix,
            suffix=suffix,
            include_df_in_prompt=include_df_in_prompt,
            number_of_head_rows=number_of_head_rows,
        )
        if agent_type == AgentType.OPENAI_FUNCTIONS:
            agent = RunnableAgent(
                runnable=create_openai_functions_agent(llm, tools, prompt),  # type: ignore
                input_keys_arg=["input"],
                return_keys_arg=["output"],
            )
        else:
            agent = RunnableMultiActionAgent(
                runnable=create_openai_tools_agent(llm, tools, prompt),  # type: ignore
                input_keys_arg=["input"],
                return_keys_arg=["output"],
            )
    else:
        raise ValueError(
            f"Agent type {agent_type} not supported at the moment. Must be one of "
            "'openai-tools', 'openai-functions', or 'zero-shot-react-description'."
        )
    return AgentExecutor(
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
