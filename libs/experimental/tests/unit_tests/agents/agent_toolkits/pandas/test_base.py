import sys

import pytest

from langchain_experimental.agents import create_pandas_dataframe_agent
from tests.unit_tests.fake_llm import FakeLLM


@pytest.mark.requires("pandas", "tabulate")
@pytest.mark.skipif(sys.version_info < (3, 9), reason="requires python3.9 or higher")
def test_create_pandas_dataframe_agent() -> None:
    import pandas as pd

    create_pandas_dataframe_agent(FakeLLM(), pd.DataFrame())
    create_pandas_dataframe_agent(FakeLLM(), [pd.DataFrame(), pd.DataFrame()])
