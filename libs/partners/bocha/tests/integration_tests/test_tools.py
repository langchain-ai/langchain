import os
import pytest
from langchain_bocha import BochaSearchRun, BochaSearchResults


@pytest.mark.skipif(
    not os.environ.get("BOCHA_API_KEY"),
    reason="BOCHA_API_KEY not set"
)
def test_bocha_search_run_live():
    tool = BochaSearchRun()
    result = tool._run("Python programming language")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.skipif(
    not os.environ.get("BOCHA_API_KEY"),
    reason="BOCHA_API_KEY not set"
)
def test_bocha_search_results_live():
    tool = BochaSearchResults()
    result = tool._run("Python programming language")
    assert isinstance(result, dict)
