import re

import numpy as np
import pytest
from pandas import DataFrame
from pydantic import BaseModel

from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent import AgentExecutor
from langchain.llms import OpenAI


class TestData(BaseModel):
    df: DataFrame
    dim: int

    class Config:
        arbitrary_types_allowed = True


@pytest.fixture(scope="function")
def data() -> TestData:
    dim = 4
    random_data = np.random.rand(dim, dim)
    df = DataFrame(random_data, columns=["name", "age", "food", "sport"])
    return TestData(df=df, dim=4)


def test_pandas_agent_creation(data: TestData) -> None:
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), data.df)
    assert isinstance(agent, AgentExecutor)

def test_data_reading(data: TestData) -> None:
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), data.df)
    assert isinstance(agent, AgentExecutor)
    response = agent.run("how many rows in df? Give me a number.")
    result = re.search(rf".*({data.dim}).*", response)
    assert result is not None
    assert result.group(1) is not None
