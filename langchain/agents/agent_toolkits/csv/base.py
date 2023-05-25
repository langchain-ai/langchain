"""Agent for working with csvs."""
from typing import Any, Optional

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.base_language import BaseLanguageModel


def create_csv_agent(
    llm: BaseLanguageModel,
    path: str,
    pandas_kwargs: Optional[dict] = None,
    **kwargs: Any
) -> AgentExecutor:
    """Create csv agent by loading to a dataframe and using pandas agent."""
    import pandas as pd

    _kwargs = pandas_kwargs or {}
    df = pd.read_csv(path, **_kwargs)
    return create_pandas_dataframe_agent(llm, df, **kwargs)
