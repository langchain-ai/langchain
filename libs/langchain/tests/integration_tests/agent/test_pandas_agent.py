import re

import numpy as np
import pytest
from pandas import DataFrame

from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent import AgentExecutor
from langchain.llms import OpenAI


@pytest.fixture(scope="module")
def df() -> DataFrame:
    random_data = np.random.rand(4, 4)
    df = DataFrame(random_data, columns=["name", "age", "food", "sport"])
    return df


# Figure out type hint here
@pytest.fixture(scope="module")
def df_list() -> list:
    random_data = np.random.rand(4, 4)
    df1 = DataFrame(random_data, columns=["name", "age", "food", "sport"])
    random_data = np.random.rand(2, 2)
    df2 = DataFrame(random_data, columns=["name", "height"])
    df_list = [df1, df2]
    return df_list


def test_pandas_agent_creation(df: DataFrame) -> None:
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df)
    assert isinstance(agent, AgentExecutor)


def test_data_reading(df: DataFrame) -> None:
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df)
    assert isinstance(agent, AgentExecutor)
    response = agent.run("how many rows in df? Give me a number.")
    result = re.search(rf".*({df.shape[0]}).*", response)
    assert result is not None
    assert result.group(1) is not None


def test_data_reading_no_df_in_prompt(df: DataFrame) -> None:
    agent = create_pandas_dataframe_agent(
        OpenAI(temperature=0), df, include_df_in_prompt=False
    )
    assert isinstance(agent, AgentExecutor)
    response = agent.run("how many rows in df? Give me a number.")
    result = re.search(rf".*({df.shape[0]}).*", response)
    assert result is not None
    assert result.group(1) is not None


def test_multi_df(df_list: list) -> None:
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df_list, verbose=True)
    response = agent.run("how many total rows in the two dataframes? Give me a number.")
    result = re.search(r".*(6).*", response)
    assert result is not None
    assert result.group(1) is not None


def test_multi_df_no_df_in_prompt(df_list: list) -> None:
    agent = create_pandas_dataframe_agent(
        OpenAI(temperature=0), df_list, include_df_in_prompt=False
    )
    response = agent.run("how many total rows in the two dataframes? Give me a number.")
    result = re.search(r".*(6).*", response)
    assert result is not None
    assert result.group(1) is not None
