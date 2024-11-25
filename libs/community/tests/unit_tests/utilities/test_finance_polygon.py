import unittest
from unittest.mock import MagicMock, patch

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper

API_KEY = "test_key"  # Test API key


class TestFinancePolygonAPIWrapper(unittest.TestCase):
    def setUp(self) -> None:
        self.wrapper: FinancePolygonAPIWrapper = FinancePolygonAPIWrapper(
            polygon_api_key=API_KEY
        )

    @patch("requests.get")
    def test_get_crypto_aggregates(self, mock_get: MagicMock) -> None:
        mock_get.return_value.json.return_value = {
            "status": "OK",
            "results": {"key": "value"},
        }
        result = self.wrapper.get_crypto_aggregates(
            "BTC/USD", from_date="2022-01-01", to_date="2022-01-10"
        )
        self.assertEqual(result, {"key": "value"})
        mock_get.assert_called_once_with(
            f"https://api.polygon.io/v2/aggs/ticker/BTC/USD/range/1/day/2022-01-01/2022-01-10"
            f"?apiKey={API_KEY}&adjusted=True&sort=asc"
        )

    @patch("requests.get")
    def test_get_ipos(self, mock_get: MagicMock) -> None:
        mock_get.return_value.json.return_value = {
            "status": "OK",
            "results": {"ipo": "data"},
        }
        result = self.wrapper.get_ipos(limit=5, sort="ticker")
        self.assertEqual(result, {"ipo": "data"})
        mock_get.assert_called_once_with(
            f"https://api.polygon.io/vX/reference/ipos?order=asc&limit=5&sort=ticker&apiKey={API_KEY}"
        )

    @patch("requests.get")
    def test_get_reference_tickers(self, mock_get: MagicMock) -> None:
        mock_get.return_value.json.return_value = {
            "status": "OK",
            "results": {"tickers": ["AAPL", "GOOG"]},
        }
        result = self.wrapper.get_reference_tickers(ticker="AAPL", limit=2)
        self.assertEqual(result, {"tickers": ["AAPL", "GOOG"]})
        mock_get.assert_called_once_with(
            f"https://api.polygon.io/v3/reference/tickers?ticker=AAPL&active=true&limit=2&apiKey={API_KEY}"
        )

    @patch("requests.get")
    def test_get_reference_ticker_news(self, mock_get: MagicMock) -> None:
        mock_get.return_value.json.return_value = {
            "status": "OK",
            "results": {"news": ["News1", "News2"]},
        }
        result = self.wrapper.get_reference_ticker_news(ticker="AAPL", limit=2)
        self.assertEqual(result, {"news": ["News1", "News2"]})
        mock_get.assert_called_once_with(
            f"https://api.polygon.io/v3/reference/tickers?ticker=AAPL&limit=2&apiKey={API_KEY}"
        )

    @patch("requests.get")
    def test_get_reference_ticker_details(self, mock_get: MagicMock) -> None:
        mock_get.return_value.json.return_value = {
            "status": "OK",
            "results": {"ticker": "AAPL"},
        }
        result = self.wrapper.get_reference_ticker_details("AAPL")
        self.assertEqual(result, {"ticker": "AAPL"})
        mock_get.assert_called_once_with(
            f"https://api.polygon.io/v3/reference/tickers/AAPL?apiKey={API_KEY}"
        )

    @patch("requests.get")
    def test_get_related_companies(self, mock_get: MagicMock) -> None:
        mock_get.return_value.json.return_value = {
            "status": "OK",
            "results": ["MSFT", "GOOG"],
        }
        result = self.wrapper.get_related_companies("AAPL")
        self.assertEqual(result, ["MSFT", "GOOG"])
        mock_get.assert_called_once_with(
            f"https://api.polygon.io/v1/related-companies/AAPL?apiKey={API_KEY}"
        )

    @patch("requests.get")
    def test_get_exchanges(self, mock_get: MagicMock) -> None:
        mock_get.return_value.json.return_value = {
            "status": "OK",
            "results": {"exchanges": ["NYSE", "NASDAQ"]},
        }
        result = self.wrapper.get_exchanges()
        self.assertEqual(result, {"exchanges": ["NYSE", "NASDAQ"]})
        mock_get.assert_called_once_with(
            f"https://api.polygon.io/v3/reference/exchanges?asset_class=stocks&apiKey={API_KEY}"
        )

    @patch("requests.get")
    def test_get_conditions(self, mock_get: MagicMock) -> None:
        mock_get.return_value.json.return_value = {
            "status": "OK",
            "results": {"conditions": ["condition1"]},
        }
        result = self.wrapper.get_conditions()
        self.assertEqual(result, {"conditions": ["condition1"]})
        mock_get.assert_called_once_with(
            f"https://api.polygon.io/v3/reference/conditions?asset_class=stocks&limit=10&apiKey={API_KEY}"
        )

    @patch("requests.get")
    def test_get_stock_splits(self, mock_get: MagicMock) -> None:
        mock_get.return_value.json.return_value = {
            "status": "OK",
            "results": {"splits": ["split1"]},
        }
        result = self.wrapper.get_stock_splits(ticker="AAPL")
        self.assertEqual(result, {"splits": ["split1"]})
        mock_get.assert_called_once_with(
            f"https://api.polygon.io/v3/reference/splits?ticker=AAPL&limit=10&apiKey={API_KEY}"
        )

    @patch("requests.get")
    def test_get_last_trade(self, mock_get: MagicMock) -> None:
        mock_get.return_value.json.return_value = {
            "status": "OK",
            "results": {"trade": "data"},
        }
        result = self.wrapper.get_last_trade("AAPL")
        self.assertEqual(result, {"trade": "data"})
        mock_get.assert_called_once_with(
            f"https://api.polygon.io/v2/last/trade/AAPL?apiKey={API_KEY}"
        )

    @patch("requests.get")
    def test_get_previous_close(self, mock_get: MagicMock) -> None:
        mock_get.return_value.json.return_value = {
            "status": "OK",
            "results": {"close": "data"},
        }
        result = self.wrapper.get_previous_close("AAPL")
        self.assertEqual(result, {"close": "data"})
        mock_get.assert_called_once_with(
            f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?adjusted=True&apiKey={API_KEY}"
        )

    @patch("requests.get")
    def test_get_market_status(self, mock_get: MagicMock) -> None:
        mock_get.return_value.json.return_value = {"status": "OK", "market": "open"}
        result = self.wrapper.get_market_status()
        self.assertEqual(result, {"status": "OK", "market": "open"})
        mock_get.assert_called_once_with(
            f"https://api.polygon.io/v1/marketstatus/now?apiKey={API_KEY}"
        )

    @patch("requests.get")
    def test_get_market_holidays(self, mock_get: MagicMock) -> None:
        mock_get.return_value.json.return_value = [{"status": "close"}]
        result = self.wrapper.get_market_holidays()
        self.assertEqual(result, [{"status": "close"}])
        mock_get.assert_called_once_with(
            f"https://api.polygon.io/v1/marketstatus/upcoming?apiKey={API_KEY}"
        )

    @patch("requests.get")
    def test_get_dividends(self, mock_get: MagicMock) -> None:
        mock_get.return_value.json.return_value = {
            "status": "OK",
            "results": [{"dividend": "data"}],
        }
        result = self.wrapper.get_dividends("AAPL")
        self.assertEqual(result, [{"dividend": "data"}])
        mock_get.assert_called_once_with(
            f"https://api.polygon.io/v3/reference/dividends?ticker=AAPL&limit=10&apiKey={API_KEY}"
        )

    def test_run_invalid_mode(self) -> None:
        with self.assertRaises(ValueError) as context:
            self.wrapper.run("invalid_mode")
        self.assertIn("Invalid mode", str(context.exception))


if __name__ == "__main__":
    unittest.main()
