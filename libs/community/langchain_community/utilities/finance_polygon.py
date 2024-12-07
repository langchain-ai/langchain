"""
Util that calls several of Polygon's REST APIs.
Docs: https://polygon.io/docs/stocks/getting-started
"""

import json
from typing import Any, Dict, Optional

import requests
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, model_validator

POLYGON_BASE_URL = "https://api.polygon.io/"


class FinancePolygonAPIWrapper(BaseModel):
    """Wrapper for Polygon Finance API."""

    polygon_api_key: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key in environment."""
        polygon_api_key = get_from_dict_or_env(
            values, "polygon_api_key", "POLYGON_API_KEY"
        )
        values["polygon_api_key"] = polygon_api_key

        return values

    @staticmethod
    def _get_response(endpoint: str) -> dict:
        """Make Get Request to `endpoint` and return `data` if OK"""
        response = requests.get(endpoint)
        data = response.json()
        status = data.get("status", None)
        if status not in ("OK"):
            raise ValueError(f"API Error: {data}")
        return data.get("results", None)

    def get_crypto_aggregates(self, ticker: str, **kwargs: Any) -> Optional[dict]:
        """
        Get aggregate bars for a cryptocurrency over a given date range
        in custom time window sizes.

        /v2/aggs/ticker/{cryptoTicker}/range/{multiplier}/{timespan}/{from}/{to}
        """
        timespan = kwargs.get("timespan", "day")
        multiplier = kwargs.get("timespan_multiplier", 1)
        from_date = kwargs.get("from_date", None)
        to_date = kwargs.get("to_date", None)
        adjusted = kwargs.get("adjusted", True)
        sort = kwargs.get("sort", "asc")

        url = (
            f"{POLYGON_BASE_URL}v2/aggs"
            f"/ticker/{ticker}"
            f"/range/{multiplier}"
            f"/{timespan}"
            f"/{from_date}"
            f"/{to_date}"
            f"?apiKey={self.polygon_api_key}"
            f"&adjusted={adjusted}"
            f"&sort={sort}"
        )

        return self._get_response(url)

    def get_ipos(self, **kwargs: Any) -> Optional[dict]:
        """
        Get IPOs for a specific ticker or other criteria.

        /vX/reference/ipos?limit={limit}&apiKey={apiKey}
        """
        ticker = kwargs.get("ticker", None)
        us_code = kwargs.get("us_code", None)
        isin = kwargs.get("isin", None)
        listing_date = kwargs.get("listing_date", None)
        ipo_status = kwargs.get("ipo_status", None)
        order = kwargs.get("order", "asc")
        limit = kwargs.get("limit", 10)
        sort = kwargs.get("sort", "listing_date")

        url = POLYGON_BASE_URL + "vX/reference/ipos?"
        if ticker is not None:
            url = url + f"ticker={ticker}&"

        if us_code is not None:
            url = url + f"us_code={us_code}&"

        if isin is not None:
            url = url + f"isin={isin}&"

        if listing_date is not None:
            url = url + f"listing_date={listing_date}&"

        if ipo_status is not None:
            url = url + f"ipo_status={ipo_status}&"

        url = (
            url
            + f"order={order}&limit={limit}&sort={sort}&apiKey={self.polygon_api_key}"
        )

        return self._get_response(url)

    def get_reference_tickers(self, **kwargs: Any) -> Optional[dict]:
        """
        Get all supported reference tickers by Polygon.

        /v3/reference/tickers/
        """
        ticker = kwargs.get("ticker", "")
        ticker_type = kwargs.get("ticker_type", "")
        market = kwargs.get("market", "")
        exchange = kwargs.get("exchange", "")
        cusip = kwargs.get("cusip", "")
        cik = kwargs.get("cik", "")
        date = kwargs.get("date", "")  # add default date
        search = kwargs.get("search", "")
        active = kwargs.get("active", "true")
        order = kwargs.get("order", "")
        limit = kwargs.get("limit", 100)
        sort = kwargs.get("sort", "")

        url = POLYGON_BASE_URL + "v3/reference/tickers?"

        if ticker != "":
            url = url + f"ticker={ticker}&"
        if ticker_type != "":
            url = url + f"ticker_type={ticker_type}&"
        if market != "":
            url = url + f"market={market}&"
        if exchange != "":
            url = url + f"exchange={exchange}&"
        if cusip != "":
            url = url + f"cusip={cusip}&"
        if cik != "":
            url = url + f"exchange={cik}&"
        if date != "":
            url = url + f"date={date}&"
        if search != "":
            url = url + f"search={search}&"
        url = url + f"active={active}&"
        if order != "":
            url = url + f"order={order}&"
        url = url + f"limit={limit}&"
        if sort != "":
            url = url + f"sort={sort}&"
        url = url + f"apiKey={self.polygon_api_key}"

        return self._get_response(url)

    def get_reference_ticker_news(self, **kwargs: Any) -> Optional[dict]:
        """
        Get the most recent news articles relating to a stock ticker symbol,
        including a summary of the article and a link to the original source.

        /v2/reference/news/
        """
        ticker = kwargs.get("ticker", "")
        published_utc = kwargs.get("published_utc", "")
        order = kwargs.get("order", "")
        limit = kwargs.get("limit", 10)
        sort = kwargs.get("sort", "")

        url = POLYGON_BASE_URL + "v3/reference/tickers?"

        if ticker != "":
            url = url + f"ticker={ticker}&"
        if published_utc != "":
            url = url + f"published_utc={published_utc}&"
        if order != "":
            url = url + f"order={order}&"
        url = url + f"limit={limit}&"
        if sort != "":
            url = url + f"sort={sort}&"
        url = url + f"apiKey={self.polygon_api_key}"

        return self._get_response(url)

    def get_reference_ticker_details(
        self, ticker: str, **kwargs: Any
    ) -> Optional[dict]:
        """
        Get ticker details from Polygon.

        /v3/reference/tickers/
        """
        date = kwargs.get("date", "")
        url = POLYGON_BASE_URL + f"v3/reference/tickers/{ticker}?"

        if date != "":
            url = url + f"date={date}&"

        url = url + f"apiKey={self.polygon_api_key}"

        return self._get_response(url)

    def get_reference_ticker_types(self, **kwargs: Any) -> Optional[dict]:
        """
        List all ticker types that Polygon.io has.

        /v3/reference/tickers/types
        """
        asset_class = kwargs.get("asset_class", "")
        locale = kwargs.get("locale", "")
        url = POLYGON_BASE_URL + "v3/reference/tickers/types?"

        if asset_class != "":
            url = url + f"asset_class={asset_class}&"
        if locale != "":
            url = url + f"locale={locale}&"

        url = url + f"apiKey={self.polygon_api_key}"

        return self._get_response(url)

    def get_reference_ticker_events(
        self, event_id: str, **kwargs: Any
    ) -> Optional[dict]:
        """
        Get events details for an asset from Polygon.

        /vX/reference/tickers/{id}/events
        """
        types = kwargs.get("types", "")
        url = POLYGON_BASE_URL + f"v3/reference/tickers/{event_id}/events?"

        if types != "":
            url = url + f"types={types}&"

        url = url + f"apiKey={self.polygon_api_key}"

        return self._get_response(url)

    def get_related_companies(self, ticker: str) -> Optional[dict]:
        """
        Get a list of tickers related to the queried ticker based on News
        and Returns data.

        /v1/related-companies/{ticker}?apiKey={apiKey}
        """

        url = (
            f"{POLYGON_BASE_URL}v1/related-companies"
            f"/{ticker}"
            f"?apiKey={self.polygon_api_key}"
        )
        return self._get_response(url)

    def get_exchanges(self, **kwagrs: Any) -> Optional[dict]:
        """
        Get a list of exchanges that Polygon.io knows about.

        v3/reference/exchanges?asset_class={asset_class}&apiKey={apiKey}
        """

        asset_class = kwagrs.get("asset_class", "stocks")
        locale = kwagrs.get("locale", None)

        url = (
            POLYGON_BASE_URL + "v3/reference/exchanges?" + f"asset_class={asset_class}&"
        )

        if locale is not None:
            url = url + f"locale={locale}&"

        url = url + f"apiKey={self.polygon_api_key}"

        return self._get_response(url)

    def get_conditions(self, **kwargs: Any) -> Optional[dict]:
        """
        Get a list of conditions that Polygon.io uses.

        /v3/reference/conditions?asset_class={asset_class}&limit={limit}&apiKey={apiKey}
        """

        asset_class = kwargs.get("asset_class", "stocks")
        data_type = kwargs.get("data_type", None)
        id = kwargs.get("id", None)
        sip = kwargs.get("sip", None)
        order = kwargs.get("order", None)
        limit = kwargs.get("limit", 10)
        sort = kwargs.get("sort", None)

        url = (
            POLYGON_BASE_URL
            + "v3/reference/conditions?"
            + f"asset_class={asset_class}&"
        )

        if data_type is not None:
            url = url + f"data_type={data_type}&"

        if id is not None:
            url = url + f"id={id}&"

        if sip is not None:
            url = url + f"sip={sip}&"

        if order is not None:
            url = url + f"order={order}&"

        url = url + f"limit={limit}&"

        if sort is not None:
            url = url + f"sort={sort}&"

        url = url + f"apiKey={self.polygon_api_key}"

        return self._get_response(url)

    def get_stock_splits(self, **kwargs: Any) -> Optional[dict]:
        """
        Get a list of historical stock splits for a specific ticker.

        /v3/reference/splits?limit={limit}&apiKey={apiKey}
        """

        ticker = kwargs.get("ticker", None)
        execution_date = kwargs.get("execution_date", None)
        reverse_split = kwargs.get("reverse_split", None)
        order = kwargs.get("order", None)
        limit = kwargs.get("limit", 10)
        sort = kwargs.get("sort", None)

        url = POLYGON_BASE_URL + "v3/reference/splits?"

        if ticker is not None:
            url = url + f"ticker={ticker}&"

        if execution_date is not None:
            url = url + f"execution_date={execution_date}&"

        if reverse_split is not None:
            url = url + f"reverse_split={reverse_split}&"

        if order is not None:
            url = url + f"order={order}&"

        url = url + f"limit={limit}&"

        if sort is not None:
            url = url + f"sort={sort}&"

        url = url + f"apiKey={self.polygon_api_key}"

        return self._get_response(url)

    def get_stocks_financials(self, **kwargs: Any) -> Optional[dict]:
        """
        Get a list of historical financial data for a stock ticker.

        /vX/reference/financials?limit={limit}&apiKey={apiKey}
        """

        ticker = kwargs.get("ticker", None)
        cik = kwargs.get("cik", None)
        company_name = kwargs.get("company_name", None)
        sic = kwargs.get("sic", None)
        filing_date = kwargs.get("filing_date", None)
        period_of_report_date = kwargs.get("period_of_report_date", None)
        timeframe = kwargs.get("timeframe", None)
        include_sources = kwargs.get("include_sources", None)
        order = kwargs.get("order", None)
        limit = kwargs.get("limit", 10)
        sort = kwargs.get("sort", None)

        url = POLYGON_BASE_URL + "vX/reference/financials?"

        if ticker is not None:
            url = url + f"ticker={ticker}&"

        if cik is not None:
            url = url + f"cik={cik}&"

        if company_name is not None:
            url = url + f"company_name={company_name}&"

        if sic is not None:
            url = url + f"sic={sic}&"

        if filing_date is not None:
            url = url + f"filing_date={filing_date}&"

        if period_of_report_date is not None:
            url = url + f"period_of_report_date={period_of_report_date}&"

        if timeframe is not None:
            url = url + f"timeframe={timeframe}&"

        if include_sources is not None:
            url = url + f"include_sources={include_sources}&"

        if order is not None:
            url = url + f"order={order}&"

        url = url + f"limit={limit}&"

        if sort is not None:
            url = url + f"sort={sort}&"

        url = url + f"apiKey={self.polygon_api_key}"

        return self._get_response(url)

    def get_last_trade(self, ticker: str) -> Optional[dict]:
        """
        Get the last trade data for a given stock.

        /v2/last/trade/{ticker}?apiKey={apiKey}
        """
        url = (
            f"{POLYGON_BASE_URL}v2/last/trade"
            f"/{ticker}"
            f"?apiKey={self.polygon_api_key}"
        )
        return self._get_response(url)

    def get_market_holidays(self, **kwargs: Any) -> Optional[dict]:
        """
        Get the upcoming market holidays and their open/close times.

        /v1/marketstatus/upcoming?apiKey={apiKey}
        """

        url = (
            POLYGON_BASE_URL
            + "v1/marketstatus/upcoming?"
            + f"apiKey={self.polygon_api_key}"
        )

        response = requests.get(url)
        data = response.json()

        for i in range(len(data)):
            status = data[i].get("status", None)
            if status not in ("close" or "early-close"):
                raise ValueError(f"API Error: {data}")

        if data is None:
            return None

        return data

    def get_market_status(self, **kwargs: Any) -> Optional[dict]:
        """
        Get the current trading status of the exchanges
        and overall financial markets.

        v1/marketstatus/now?apiKey={apiKey}
        """

        url = (
            POLYGON_BASE_URL + "v1/marketstatus/now?" + f"apiKey={self.polygon_api_key}"
        )

        response = requests.get(url)
        data = response.json()

        status = data.get("market", None)
        if status not in ("open" or "closed" or "extended-hours"):
            raise ValueError(f"API Error: {data}")

        return data

    def get_all_tickers(self, **kwargs: Any) -> Optional[dict]:
        """
        Get market data for all traded stock symbols or specific tickers.

        /v2/snapshot/locale/us/markets/stocks/tickers
        """
        tickers = kwargs.get("tickers", "")
        include_otc = kwargs.get("include_otc", False)

        url = (
            f"{POLYGON_BASE_URL}v2/snapshot/locale/us/markets/stocks/tickers?"
            f"tickers={tickers}&include_otc={str(include_otc).lower()}&apiKey={self.polygon_api_key}"
        )
        return self._get_response(url)

    def get_top_gainers_losers(self, **kwargs: Any) -> Optional[dict]:
        """
        Get the top gainers or losers in the market.

        /v2/snapshot/locale/us/markets/stocks/{direction}
        """
        direction = kwargs.get("direction")
        include_otc = kwargs.get("include_otc", False)

        if direction not in ["gainers", "losers"]:
            raise ValueError("Direction must be either 'gainers' or 'losers'")

        url = (
            f"{POLYGON_BASE_URL}v2/snapshot/locale/us/markets/stocks/{direction}?"
            f"include_otc={str(include_otc).lower()}&apiKey={self.polygon_api_key}"
        )
        return self._get_response(url)

    def get_single_ticker(self, ticker: str, **kwargs: Any) -> Optional[dict]:
        """
        Get market data for a single stock ticker.

        /v2/snapshot/locale/us/markets/stocks/tickers/{stocksTicker}
        """
        url = (
            f"{POLYGON_BASE_URL}v2/snapshot/locale/us/markets/stocks/tickers/{ticker}?"
            f"apiKey={self.polygon_api_key}"
        )
        return self._get_response(url)

    def get_universal_snapshot(self, **kwargs: Any) -> Optional[dict]:
        """
        Get snapshots for multiple asset types.

        /v3/snapshot
        """
        tickers = kwargs.get("tickers", "")
        limit = kwargs.get("limit", 10)

        url = (
            f"{POLYGON_BASE_URL}v3/snapshot?"
            f"ticker.any_of={tickers}&limit={limit}&apiKey={self.polygon_api_key}"
        )
        return self._get_response(url)

    def get_sma(self, ticker: str, **kwargs: Any) -> Optional[dict]:
        """
        Get the Simple Moving Average (SMA) for a ticker.

        /v1/indicators/sma/{stockTicker}
        """
        timespan = kwargs.get("timespan", "day")
        adjusted = kwargs.get("adjusted", True)
        window = kwargs.get("window", 50)
        series_type = kwargs.get("series_type", "close")
        limit = kwargs.get("limit", 10)

        url = (
            f"{POLYGON_BASE_URL}v1/indicators/sma/{ticker}?"
            f"timespan={timespan}&adjusted={str(adjusted).lower()}&window={window}&"
            f"series_type={series_type}&limit={limit}&apiKey={self.polygon_api_key}"
        )
        return self._get_response(url)

    def get_ema(self, ticker: str, **kwargs: Any) -> Optional[dict]:
        """
        Get the Exponential Moving Average (EMA) for a ticker.

        /v1/indicators/ema/{stockTicker}
        """
        timespan = kwargs.get("timespan", "day")
        adjusted = kwargs.get("adjusted", True)
        window = kwargs.get("window", 50)
        series_type = kwargs.get("series_type", "close")
        limit = kwargs.get("limit", 10)

        url = (
            f"{POLYGON_BASE_URL}v1/indicators/ema/{ticker}?"
            f"timespan={timespan}&adjusted={str(adjusted).lower()}&window={window}&"
            f"series_type={series_type}&limit={limit}&apiKey={self.polygon_api_key}"
        )
        return self._get_response(url)

    def get_macd(self, ticker: str, **kwargs: Any) -> Optional[dict]:
        """
        Get Moving Average Convergence/Divergence (MACD) data for a ticker.

        /v1/indicators/macd/{stockTicker}
        """
        timespan = kwargs.get("timespan", "day")
        adjusted = kwargs.get("adjusted", True)
        short_window = kwargs.get("short_window", 12)
        long_window = kwargs.get("long_window", 26)
        signal_window = kwargs.get("signal_window", 9)
        series_type = kwargs.get("series_type", "close")
        limit = kwargs.get("limit", 10)

        url = (
            f"{POLYGON_BASE_URL}v1/indicators/macd/{ticker}?"
            f"timespan={timespan}&adjusted={str(adjusted).lower()}&"
            f"short_window={short_window}&long_window={long_window}&"
            f"signal_window={signal_window}&series_type={series_type}&limit={limit}&apiKey={self.polygon_api_key}"
        )
        return self._get_response(url)

    def get_rsi(self, ticker: str, **kwargs: Any) -> Optional[dict]:
        """
        Get the Relative Strength Index (RSI) for a ticker.

        /v1/indicators/rsi/{stockTicker}
        """
        timespan = kwargs.get("timespan", "day")
        adjusted = kwargs.get("adjusted", True)
        window = kwargs.get("window", 14)
        series_type = kwargs.get("series_type", "close")
        limit = kwargs.get("limit", 10)

        url = (
            f"{POLYGON_BASE_URL}v1/indicators/rsi/{ticker}?"
            f"timespan={timespan}&adjusted={str(adjusted).lower()}&window={window}&"
            f"series_type={series_type}&limit={limit}&apiKey={self.polygon_api_key}"
        )
        return self._get_response(url)

    def get_aggregates(self, ticker: str, **kwargs: Any) -> Optional[dict]:
        """
        Get aggregate bars for a stock over a given date range
        in custom time window sizes.

        /v2/aggs/ticker/{stocksTicker}/range/{multiplier}/{timespan}/{from}/{to}
        """
        multiplier = kwargs.get("timespan_multiplier", 1)
        timespan = kwargs.get("timespan", "day")
        from_date = kwargs.get("from_date", None)
        to_date = kwargs.get("to_date", None)
        adjusted = kwargs.get("adjusted", True)
        sort = kwargs.get("sort", "asc")
        limit = kwargs.get("limit", 5000)

        url = POLYGON_BASE_URL + "/v2/aggs/ticker"

        if ticker is not None:
            url += f"/{ticker}"

        url += f"/range/{multiplier}/{timespan}"

        if from_date is not None:
            url += f"/{from_date}"
        if to_date is not None:
            url += f"/{to_date}"

        url += (
            f"?adjusted={adjusted}&sort={sort}&limit={limit}"
            f"&apiKey={self.polygon_api_key}"
        )

        return self._get_response(url)

    def get_daily_open_close(self, stocksTicker: str, **kwargs: Any) -> Optional[dict]:
        """
        Get daily open/close for a stock given the ticker symbol and date

        /v1/open-close/{stocksTicker}/{date}
        """

        date = kwargs.get("date", None)
        adjusted = kwargs.get("adjusted", True)

        url = POLYGON_BASE_URL + "/v1/open-close"

        if stocksTicker is not None:
            url += f"/{stocksTicker}"
        if date is not None:
            url += f"/{date}"

        url += f"?adjusted={adjusted}&apiKey={self.polygon_api_key}"

        return self._get_response(url)

    def get_grouped_daily(self, **kwargs: Any) -> Optional[dict]:
        """
        Get the daily open, high, low, and close (OHLC) for
        the entire stocks/equities markets, given the beginning date
        for the aggregate window

        /v2/aggs/grouped/locale/us/market/stocks/{date}
        """

        date = kwargs.get("date", None)
        adjusted = kwargs.get("adjusted", True)
        include_otc = kwargs.get("include_otc", False)

        url = f"{POLYGON_BASE_URL}v2/aggs/grouped/locale/us/market/stocks"

        if date is not None:
            url += f"/{date}"
        url += (
            f"?adjusted={adjusted}"
            f"&include_otc={include_otc}"
            f"&apiKey={self.polygon_api_key}"
        )

        return self._get_response(url)

    def get_last_quote(self, ticker: str) -> Optional[dict]:
        """
        Get the most recent NBBO (Quote) tick for a given stock.

        /v2/last/nbbo/{stocksTicker}
        """

        url = f"{POLYGON_BASE_URL}vs/last/nbbo"
        if ticker is not None:
            url += f"/{ticker}"

        url += f"?apiKey={self.polygon_api_key}"

        return self._get_response(url)

    def get_previous_close(self, ticker: str, **kwargs: Any) -> Optional[dict]:
        """
        Get the previous day's open, high, low, and close (OHLC)
        given the stock ticker.

        /v2/aggs/ticker/{stocksTicker}/prev
        """

        adjusted = kwargs.get("adjusted", True)

        url = f"{POLYGON_BASE_URL}v2/aggs/ticker"

        if ticker is not None:
            url += f"/{ticker}"

        url += f"/prev?adjusted={adjusted}&apiKey={self.polygon_api_key}"

        return self._get_response(url)

    def get_trades(self, ticker: str, **kwargs: Any) -> Optional[dict]:
        """
        Get trades for a ticker symbol in a given the timestamp.

        /v3/trades/{stockTicker}
        """

        timestamp = kwargs.get("timestamp", None)
        order = kwargs.get("order", None)
        limit = kwargs.get("limit", 1000)
        sort = kwargs.get("sort", None)

        url = f"{POLYGON_BASE_URL}v3/trades"

        if ticker is not None:
            url += f"/{ticker}"
        if timestamp is not None:
            url += f"?timestamp={timestamp}"
        if order is not None:
            url += f"&order={order}"

        url += f"&limit={limit}"

        if sort is not None:
            url += f"&sort={sort}"

        url += f"&apiKey={self.polygon_api_key}"

        return self._get_response(url)

    def get_dividends(self, ticker: str, **kwargs: Any) -> Optional[dict]:
        """
        Get a list of historical cash dividends, including the ticker symbol,
        declaration date, ex-dividend date, record date, pay date, frequency,
        and amount.

        /v3/reference/dividends
        """

        ex_dividend_date = kwargs.get("ex_dividend_date", None)
        record_date = kwargs.get("record_date", None)
        declaration_date = kwargs.get("declaration_date", None)
        pay_date = kwargs.get("pay_date", None)
        frequency = kwargs.get("frequency", None)
        cash_amount = kwargs.get("cash_amount", None)
        dividend_type = kwargs.get("dividend_type", None)
        order = kwargs.get("order", None)
        limit = kwargs.get("limit", 10)
        sort = kwargs.get("sort", None)

        url = f"{POLYGON_BASE_URL}v3/reference/dividends"

        if ticker is not None:
            url += f"?ticker={ticker}"
        if ex_dividend_date is not None:
            url += f"&ex_dividend_date={ex_dividend_date}"
        if record_date is not None:
            url += f"&record_date={record_date}"
        if declaration_date is not None:
            url += f"&declaration_date={declaration_date}"
        if pay_date is not None:
            url += f"&pay_date={pay_date}"
        if frequency is not None:
            url += f"&frequency={frequency}"
        if cash_amount is not None:
            url += f"&cash_amount={cash_amount}"
        if dividend_type is not None:
            url += f"&dividend_type={dividend_type}"
        if order is not None:
            url += f"&order={order}"

        url += f"&limit={limit}"

        if sort is not None:
            url += f"&sort={sort}"

        url += f"&apiKey={self.polygon_api_key}"

        return self._get_response(url)

    def run(self, mode: str, ticker: str = "", **kwargs: Any) -> str:
        if mode == "get_crypto_aggregate":
            return json.dumps(self.get_crypto_aggregates(ticker))
        elif mode == "get_ipos":
            return json.dumps(self.get_ipos(**kwargs))
        elif mode == "get_related_companies":
            return json.dumps(self.get_related_companies(ticker))
        elif mode == "get_exchanges":
            return json.dumps(self.get_exchanges(**kwargs))
        elif mode == "get_conditions":
            return json.dumps(self.get_conditions(**kwargs))
        elif mode == "get_stock_splits":
            return json.dumps(self.get_stock_splits(**kwargs))
        elif mode == "get_stocks_financials":
            return json.dumps(self.get_stocks_financials(**kwargs))
        elif mode == "get_last_trade":
            return json.dumps(self.get_last_trade(ticker))
        elif mode == "get_market_holidays":
            return json.dumps(self.get_market_holidays(**kwargs))
        elif mode == "get_market_status":
            return json.dumps(self.get_market_status(**kwargs))
        elif mode == "get_all_tickers":
            return json.dumps(self.get_all_tickers(**kwargs))
        elif mode == "get_top_gainers_losers":
            return json.dumps(self.get_top_gainers_losers(**kwargs))
        elif mode == "get_single_ticker":
            return json.dumps(self.get_single_ticker(ticker, **kwargs))
        elif mode == "get_universal_snapshot":
            return json.dumps(self.get_universal_snapshot(**kwargs))
        elif mode == "get_sma":
            return json.dumps(self.get_sma(ticker, **kwargs))
        elif mode == "get_ema":
            return json.dumps(self.get_ema(ticker, **kwargs))
        elif mode == "get_macd":
            return json.dumps(self.get_macd(ticker, **kwargs))
        elif mode == "get_rsi":
            return json.dumps(self.get_rsi(ticker, **kwargs))
        elif mode == "get_aggregates":
            return json.dumps(self.get_aggregates(ticker, **kwargs))
        elif mode == "get_daily_open_close":
            return json.dumps(self.get_daily_open_close(ticker, **kwargs))
        elif mode == "get_grouped_daily":
            return json.dumps(self.get_grouped_daily(**kwargs))
        elif mode == "get_last_quote":
            return json.dumps(self.get_last_quote(ticker))
        elif mode == "get_previous_close":
            return json.dumps(self.get_previous_close(ticker, **kwargs))
        elif mode == "get_trades":
            return json.dumps(self.get_trades(ticker, **kwargs))
        elif mode == "get_dividends":
            return json.dumps(self.get_dividends(ticker, **kwargs))
        elif mode == "get_reference_tickers":
            return json.dumps(self.get_reference_tickers(**kwargs))
        elif mode == "get_reference_ticker_details":
            return json.dumps(self.get_reference_ticker_details(ticker, **kwargs))
        elif mode == "get_reference_ticker_events":
            return json.dumps(self.get_reference_ticker_events(ticker, **kwargs))
        elif mode == "get_reference_ticker_news":
            return json.dumps(self.get_reference_ticker_news(**kwargs))
        elif mode == "get_reference_ticker_types":
            return json.dumps(self.get_reference_ticker_types(**kwargs))
        else:
            raise ValueError(f"Invalid mode {mode} for Polygon API.")
