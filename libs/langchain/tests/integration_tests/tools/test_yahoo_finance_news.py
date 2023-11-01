import pytest

from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool

# skip all tests if yfinance is not installed
yfinance = pytest.importorskip("yfinance")


def test_success() -> None:
    """Test that the tool runs successfully."""
    tool = YahooFinanceNewsTool()
    query = "Microsoft"
    result = tool.run(query)
    assert result is not None
    assert f"Company ticker {query} not found." not in result


def test_failure_no_ticker() -> None:
    """Test that the tool fails."""
    tool = YahooFinanceNewsTool()
    query = ""
    result = tool.run(query)
    assert f"Company ticker {query} not found." in result


def test_failure_wrong_ticker() -> None:
    """Test that the tool fails."""
    tool = YahooFinanceNewsTool()
    query = "NOT_A_COMPANY"
    result = tool.run(query)
    assert f"Company ticker {query} not found." in result
