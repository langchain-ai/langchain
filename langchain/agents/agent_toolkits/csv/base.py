"""Agent for working with csvs."""
from typing import Any, Dict, Optional

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.llms.base import BaseLLM


def create_csv_agent(
    llm: BaseLLM,
    path: str,
    pandas_kwargs: Optional[Dict[str, Any]] = None,
    agent_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Dict[str, Any]
) -> AgentExecutor:
    """Create csv agent by loading to a dataframe and using pandas agent."""
    import pandas as pd

    df = pd.read_csv(path, **(pandas_kwargs or {}))
    return create_pandas_dataframe_agent(
        llm=llm, df=df, agent_kwargs=agent_kwargs, **kwargs
    )
