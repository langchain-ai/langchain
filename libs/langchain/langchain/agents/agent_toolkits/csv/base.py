from io import IOBase
from typing import Any, List, Optional, Union

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.schema.language_model import BaseLanguageModel


def create_csv_agent(
    llm: BaseLanguageModel,
    path: Union[str, IOBase, List[Union[str, IOBase]]],
    pandas_kwargs: Optional[dict] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Create csv agent by loading to a dataframe and using pandas agent."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas package not found, please install with `pip install pandas`"
        )

    _kwargs = pandas_kwargs or {}
    if isinstance(path, (str, IOBase)):
        df = pd.read_csv(path, **_kwargs)
    elif isinstance(path, list):
        if not all(isinstance(item, (str, IOBase)) for item in path):
            raise ValueError(
                f"Expected all elements in the list to be strings, got {type(path)}."
            )
        dfs = [pd.read_csv(item, **_kwargs) for item in path]
        df = pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError(f"Expected str or list, got {type(path)}")

    return create_pandas_dataframe_agent(llm, df, **kwargs)
