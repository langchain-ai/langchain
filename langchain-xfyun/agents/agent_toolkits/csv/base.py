from typing import Any, List, Optional, Union

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.schema.language_model import BaseLanguageModel


def create_csv_agent(
    llm: BaseLanguageModel,
    path: Union[str, List[str]],
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
    if isinstance(path, str):
        df = pd.read_csv(path, **_kwargs)
    elif isinstance(path, list):
        df = []
        for item in path:
            if not isinstance(item, str):
                raise ValueError(f"Expected str, got {type(path)}")
            df.append(pd.read_csv(item, **_kwargs))
    else:
        raise ValueError(f"Expected str or list, got {type(path)}")
    return create_pandas_dataframe_agent(llm, df, **kwargs)
