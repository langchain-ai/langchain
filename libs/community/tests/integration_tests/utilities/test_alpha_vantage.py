"""Integration test for Alpha Vantage API Wrapper."""
import pytest

from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper


@pytest.fixture
def api_wrapper() -> AlphaVantageAPIWrapper:
    # Ensure that the ALPHAVANTAGE_API_KEY environment variable is set
    return AlphaVantageAPIWrapper()


def test_search_symbols(api_wrapper: AlphaVantageAPIWrapper) -> None:
    """Test the search symbols API call for successful response."""
    response = api_wrapper.search_symbols("AAPL")
    assert response is not None
    assert isinstance(response, dict)


def test_market_news_sentiment(api_wrapper: AlphaVantageAPIWrapper) -> None:
    """Test the market news sentiment API call for successful response."""
    response = api_wrapper._get_market_news_sentiment("AAPL")
    assert response is not None
    assert isinstance(response, dict)


def test_time_series_daily(api_wrapper: AlphaVantageAPIWrapper) -> None:
    """Test the time series daily API call for successful response."""
    response = api_wrapper._get_time_series_daily("AAPL")
    assert response is not None
    assert isinstance(response, dict)


def test_quote_endpoint(api_wrapper: AlphaVantageAPIWrapper) -> None:
    """Test the quote endpoint API call for successful response."""
    response = api_wrapper._get_quote_endpoint("AAPL")
    assert response is not None
    assert isinstance(response, dict)


def test_time_series_weekly(api_wrapper: AlphaVantageAPIWrapper) -> None:
    """Test the time series weekly API call for successful response."""
    response = api_wrapper._get_time_series_weekly("AAPL")
    assert response is not None
    assert isinstance(response, dict)


def test_top_gainers_losers(api_wrapper: AlphaVantageAPIWrapper) -> None:
    """Test the top gainers and losers API call for successful response."""
    response = api_wrapper._get_top_gainers_losers()
    assert response is not None
    assert isinstance(response, dict)


def test_exchange_rate(api_wrapper: AlphaVantageAPIWrapper) -> None:
    """Test the exchange rate API call for successful response."""
    response = api_wrapper._get_exchange_rate("USD", "EUR")
    assert response is not None
    assert isinstance(response, dict)


def test_run_method(api_wrapper: AlphaVantageAPIWrapper) -> None:
    """Test the run method for successful response."""
    response = api_wrapper.run("USD", "EUR")
    assert response is not None
    assert isinstance(response, dict)
