from __future__ import annotations

from io import IOBase
from typing import TYPE_CHECKING, Any, List, Optional, Union

from langchain_experimental.agents.agent_toolkits.pandas.base import (
    create_pandas_dataframe_agent,
)

if TYPE_CHECKING:
    from langchain.agents.agent import AgentExecutor
    from langchain_core.language_models import LanguageModelLike


def create_csv_agent(
    llm: LanguageModelLike,
    path: Union[str, IOBase, List[Union[str, IOBase]]],
    pandas_kwargs: Optional[dict] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Create pandas dataframe agent by loading csv to a dataframe.

    Args:
        llm: Language model to use for the agent.
        path: A string path, file-like object or a list of string paths/file-like
            objects that can be read in as pandas DataFrames with pd.read_csv().
        pandas_kwargs: Named arguments to pass to pd.read_csv().
        **kwargs: Additional kwargs to pass to langchain_experimental.agents.agent_toolkits.pandas.base.create_pandas_dataframe_agent().

    Returns:
        An AgentExecutor with the specified agent_type agent and access to
        a PythonAstREPLTool with the loaded DataFrame(s) and any user-provided extra_tools.

    Example:
        .. code-block:: python

            from langchain_openai import ChatOpenAI
            from langchain_experimental.agents import create_csv_agent

            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            agent_executor = create_pandas_dataframe_agent(
                llm,
                "titanic.csv",
                agent_type="openai-tools",
                verbose=True
            )
    """  # noqa: E501
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas package not found, please install with `pip install pandas`."
        )

    _kwargs = pandas_kwargs or {}
    if isinstance(path, (str, IOBase)):
        df = pd.read_csv(path, **_kwargs)
    elif isinstance(path, list):
        df = []
        for item in path:
            if not isinstance(item, (str, IOBase)):
                raise ValueError(f"Expected str or file-like object, got {type(path)}")
            df.append(pd.read_csv(item, **_kwargs))
    else:
        raise ValueError(f"Expected str, list, or file-like object, got {type(path)}")
    return create_pandas_dataframe_agent(llm, df, **kwargs)
