import re

import numpy as np
import pytest
from _pytest.tmpdir import TempPathFactory
from pandas import DataFrame

from langchain.agents import create_csv_agent
from langchain.agents.agent import AgentExecutor
from langchain.llms import OpenAI


@pytest.fixture(scope="module")
def csv(tmp_path_factory: TempPathFactory) -> DataFrame:
    random_data = np.random.rand(4, 4)
    df = DataFrame(random_data, columns=["name", "age", "food", "sport"])
    filename = str(tmp_path_factory.mktemp("data") / "test.csv")
    df.to_csv(filename)
    return filename


@pytest.fixture(scope="module")
def csv_list(tmp_path_factory: TempPathFactory) -> DataFrame:
    random_data = np.random.rand(4, 4)
    df1 = DataFrame(random_data, columns=["name", "age", "food", "sport"])
    filename1 = str(tmp_path_factory.mktemp("data") / "test1.csv")
    df1.to_csv(filename1)

    random_data = np.random.rand(2, 2)
    df2 = DataFrame(random_data, columns=["name", "height"])
    filename2 = str(tmp_path_factory.mktemp("data") / "test2.csv")
    df2.to_csv(filename2)

    return [filename1, filename2]


def test_csv_agent_creation(csv: str) -> None:
    agent = create_csv_agent(OpenAI(temperature=0), csv)
    assert isinstance(agent, AgentExecutor)


def test_single_csv(csv: str) -> None:
    agent = create_csv_agent(OpenAI(temperature=0), csv)
    assert isinstance(agent, AgentExecutor)
    response = agent.run("How many rows in the csv? Give me a number.")
    result = re.search(r".*(4).*", response)
    assert result is not None
    assert result.group(1) is not None


def test_multi_csv(csv_list: list) -> None:
    agent = create_csv_agent(OpenAI(temperature=0), csv_list, verbose=True)
    assert isinstance(agent, AgentExecutor)
    response = agent.run("How many combined rows in the two csvs? Give me a number.")
    result = re.search(r".*(6).*", response)
    assert result is not None
    assert result.group(1) is not None
