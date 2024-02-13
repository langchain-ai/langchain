import pytest

from langchain_experimental.agents import create_pandas_dataframe_agent
from tests.unit_tests.fake_llm import FakeLLM


@pytest.mark.requires("pandas", "tabulate")
def test_create_pandas_dataframe_agent() -> None:
    import pandas as pd

    create_pandas_dataframe_agent(FakeLLM(), pd.DataFrame())
    create_pandas_dataframe_agent(FakeLLM(), [pd.DataFrame(), pd.DataFrame()])
