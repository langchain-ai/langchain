"""LangChain tools for the ConvexLake market-data API."""

from __future__ import annotations

import io
from typing import Any, Literal

from langchain_core.tools import BaseTool
from langchain_core.utils import convert_to_secret_str
from pydantic import Field, SecretStr, model_validator

from langchain_convexlake._utilities import initialize_client


# ── shared summary helper ─────────────────────────────────────────────────────

def _summarise(df: Any, max_rows: int = 20) -> tuple[str, Any]:
    """Return (text_summary, full_dataframe) for a DataFrame.

    Args:
        df: pandas DataFrame.
        max_rows: Maximum rows to include in the text preview.

    Returns:
        Tuple of (summary string for the model, full DataFrame as artifact).
    """
    rows_shown = min(max_rows, len(df))
    lines = [
        f"rows: {len(df)}",
        f"columns: {list(df.columns)}",
        "",
        df.head(rows_shown).to_string(index=False),
    ]
    num_df = df.select_dtypes(include="number")
    if not num_df.empty:
        buf = io.StringIO()
        num_df.describe().to_string(buf)
        lines += ["", "describe:", buf.getvalue()]
    return "\n".join(lines), df


# ── base mixin ────────────────────────────────────────────────────────────────

class _ConvexLakeBaseTool(BaseTool):
    """Mixin that holds the shared api_key / client fields."""

    api_key: SecretStr = Field(default=SecretStr(""))
    _client: Any = None  # ConvexLake instance, set by validator

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Initialise the ConvexLake client from the api_key or environment."""
        return initialize_client(values)

    def _error(self, e: Exception, noun: str) -> str:
        """Format SDK exceptions as model-readable strings."""
        from convexlake.exceptions import (  # noqa: PLC0415
            MultipleChoicesError,
            NotFoundError,
            PlanError,
        )

        if isinstance(e, MultipleChoicesError):
            return f"Multiple files match — pass slug= one of: {e.slugs}"
        if isinstance(e, NotFoundError):
            return f"No {noun} found. Call convexlake_catalog to verify available dates and slugs."
        if isinstance(e, PlanError):
            return "This endpoint is not in your subscription tier."
        return f"Error fetching {noun}: {e}"


# ── tools ─────────────────────────────────────────────────────────────────────

class ConvexLakeCatalog(_ConvexLakeBaseTool):
    """Browse the ConvexLake data catalog.

    Setup:
        Install ``langchain-convexlake`` and set ``CONVEXLAKE_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-convexlake
            export CONVEXLAKE_API_KEY="do_..."

    Instantiation:
        .. code-block:: python

            from langchain_convexlake import ConvexLakeCatalog

            tool = ConvexLakeCatalog()

    Invocation:
        .. code-block:: python

            tool.invoke({})
            tool.invoke({"exchange": "kalshi"})
            tool.invoke({"exchange": "kalshi", "ticker": "crypto", "data_type": "trades"})
    """

    name: str = "convexlake_catalog"
    description: str = (
        "Browse the ConvexLake data catalog. "
        "Call with no arguments to list exchanges, then drill down with exchange=, "
        "ticker=, data_type=, and date= to find available files and slugs. "
        "Supported exchanges: polymarket, kalshi, limitless, predict, deribit, "
        "binance, hyperliquid, okx, bybit, bitfinex."
    )

    def _run(
        self,
        exchange: str | None = None,
        ticker: str | None = None,
        data_type: str | None = None,
        date: str | None = None,
    ) -> dict[str, Any] | str:
        """Run the catalog tool.

        Args:
            exchange: Exchange name, e.g. "kalshi".
            ticker: Ticker / category, e.g. "crypto".
            data_type: Data type filter, e.g. "trades", "candles".
            date: Date in YYYY-MM-DD format.

        Returns:
            Catalog dict or error string.
        """
        try:
            return self._client.catalog(
                exchange=exchange, ticker=ticker, data_type=data_type, date=date
            )
        except Exception as e:
            return self._error(e, "catalog")


class ConvexLakeTrades(_ConvexLakeBaseTool):
    """Download trade history for a prediction-market or options instrument.

    Setup:
        Install ``langchain-convexlake`` and set ``CONVEXLAKE_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-convexlake
            export CONVEXLAKE_API_KEY="do_..."

    Instantiation:
        .. code-block:: python

            from langchain_convexlake import ConvexLakeTrades

            tool = ConvexLakeTrades()

    Invocation:
        .. code-block:: python

            summary, df = tool.invoke({
                "exchange": "kalshi",
                "ticker": "crypto",
                "date": "2026-10-01",
                "slug": "KXBTC15M-26OCT01-trades.csv",
            })
    """

    name: str = "convexlake_trades"
    description: str = (
        "Download trade history for a prediction market or options instrument. "
        "Returns a text summary (row count, columns, preview) as content and the "
        "full pandas DataFrame as an artifact. "
        "Use convexlake_catalog to find valid dates and slugs first."
    )
    response_format: Literal["content_and_artifact"] = "content_and_artifact"
    max_rows: int = 20

    def _run(
        self,
        exchange: Literal[
            "polymarket", "kalshi", "limitless", "deribit", "binance", "okx", "bybit"
        ],
        ticker: str,
        date: str,
        slug: str | None = None,
    ) -> tuple[str, Any]:
        """Run the trades tool.

        Args:
            exchange: Exchange name.
            ticker: Ticker or category, e.g. "crypto", "BTC".
            date: Settlement/expiry date in YYYY-MM-DD.
            slug: File slug — required when multiple files exist for the date.

        Returns:
            Tuple of (summary string, full DataFrame).
        """
        try:
            df = self._client.pm.trades(exchange, ticker, date, slug=slug)
            return _summarise(df, self.max_rows)
        except Exception as e:
            return self._error(e, "trades"), None


class ConvexLakeCandles(_ConvexLakeBaseTool):
    """Download OHLCV candle data for a prediction market or perpetual instrument.

    Setup:
        Install ``langchain-convexlake`` and set ``CONVEXLAKE_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-convexlake
            export CONVEXLAKE_API_KEY="do_..."

    Instantiation:
        .. code-block:: python

            from langchain_convexlake import ConvexLakeCandles

            tool = ConvexLakeCandles()

    Invocation:
        .. code-block:: python

            # Prediction market
            summary, df = tool.invoke({
                "exchange": "kalshi", "ticker": "crypto",
                "date": "2026-10-01", "slug": "KXBTC15M-26OCT01.csv",
            })

            # Crypto perpetual
            summary, df = tool.invoke({
                "exchange": "deribit", "ticker": "BTC-PERPETUAL", "timeframe": "1h",
            })
    """

    name: str = "convexlake_candles"
    description: str = (
        "Download OHLCV candle data for a prediction market or perpetual. "
        "For prediction markets (polymarket, kalshi, predict): pass date and slug. "
        "For crypto perpetuals (deribit, hyperliquid): pass timeframe and ticker. "
        "Returns text summary as content and the full DataFrame as artifact."
    )
    response_format: Literal["content_and_artifact"] = "content_and_artifact"
    max_rows: int = 20

    def _run(
        self,
        exchange: Literal[
            "polymarket", "kalshi", "predict", "deribit", "hyperliquid"
        ],
        ticker: str,
        date: str | None = None,
        timeframe: str | None = None,
        slug: str | None = None,
    ) -> tuple[str, Any]:
        """Run the candles tool.

        Args:
            exchange: Exchange name.
            ticker: Ticker or instrument name.
            date: Settlement/expiry date (prediction markets).
            timeframe: Candle timeframe, e.g. "1h", "4h" (crypto perpetuals).
            slug: File slug or instrument name (crypto perpetuals).

        Returns:
            Tuple of (summary string, full DataFrame).
        """
        try:
            if exchange in ("deribit", "hyperliquid") and timeframe:
                df = self._client.cx.candles(exchange, ticker, timeframe)
            elif date:
                df = self._client.pm.candles(exchange, ticker, date, slug=slug)
            else:
                return (
                    "Provide date (prediction markets) or timeframe (crypto perpetuals).",
                    None,
                )
            return _summarise(df, self.max_rows)
        except Exception as e:
            return self._error(e, "candles"), None


class ConvexLakeSurface(_ConvexLakeBaseTool):
    """Download an implied-volatility surface snapshot for an options exchange.

    Setup:
        Install ``langchain-convexlake`` and set ``CONVEXLAKE_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-convexlake
            export CONVEXLAKE_API_KEY="do_..."

    Instantiation:
        .. code-block:: python

            from langchain_convexlake import ConvexLakeSurface

            tool = ConvexLakeSurface()

    Invocation:
        .. code-block:: python

            summary, df = tool.invoke({
                "exchange": "deribit", "ticker": "BTC", "date": "2026-10-01",
            })
    """

    name: str = "convexlake_surface"
    description: str = (
        "Download the implied-volatility surface snapshot for an options exchange. "
        "Returns text summary as content and the full DataFrame as artifact. "
        "Supported exchanges: deribit, binance, okx, bybit. Tickers: BTC, ETH."
    )
    response_format: Literal["content_and_artifact"] = "content_and_artifact"
    max_rows: int = 20

    def _run(
        self,
        exchange: Literal["deribit", "binance", "okx", "bybit"],
        ticker: Literal["BTC", "ETH"],
        date: str,
    ) -> tuple[str, Any]:
        """Run the surface tool.

        Args:
            exchange: Options exchange.
            ticker: Underlying asset.
            date: Date in YYYY-MM-DD.

        Returns:
            Tuple of (summary string, full DataFrame).
        """
        try:
            df = self._client.cx.surface(exchange, ticker, date)
            return _summarise(df, self.max_rows)
        except Exception as e:
            return self._error(e, "IV surface"), None


class ConvexLakeLeaderboard(_ConvexLakeBaseTool):
    """Download a daily leaderboard snapshot for a prediction-market exchange.

    Setup:
        Install ``langchain-convexlake`` and set ``CONVEXLAKE_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-convexlake
            export CONVEXLAKE_API_KEY="do_..."

    Instantiation:
        .. code-block:: python

            from langchain_convexlake import ConvexLakeLeaderboard

            tool = ConvexLakeLeaderboard()

    Invocation:
        .. code-block:: python

            summary, df = tool.invoke({"exchange": "polymarket", "date": "2026-10-01"})
    """

    name: str = "convexlake_leaderboard"
    description: str = (
        "Download the daily trader leaderboard for a prediction-market exchange. "
        "Returns text summary as content and the full DataFrame as artifact. "
        "Supported exchanges: polymarket, limitless, predict. "
        "Use convexlake_catalog to find the available date range."
    )
    response_format: Literal["content_and_artifact"] = "content_and_artifact"
    max_rows: int = 20

    def _run(
        self,
        exchange: Literal["polymarket", "limitless", "predict"],
        date: str,
    ) -> tuple[str, Any]:
        """Run the leaderboard tool.

        Args:
            exchange: Exchange name.
            date: Date in YYYY-MM-DD.

        Returns:
            Tuple of (summary string, full DataFrame).
        """
        try:
            df = self._client.pm.leaderboard(exchange, date)
            return _summarise(df, self.max_rows)
        except Exception as e:
            return self._error(e, "leaderboard"), None


class ConvexLakeLeaderboardTrades(_ConvexLakeBaseTool):
    """List trader addresses or fetch one trader's fill history.

    Setup:
        Install ``langchain-convexlake`` and set ``CONVEXLAKE_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-convexlake
            export CONVEXLAKE_API_KEY="do_..."

    Instantiation:
        .. code-block:: python

            from langchain_convexlake import ConvexLakeLeaderboardTrades

            tool = ConvexLakeLeaderboardTrades()

    Invocation:
        .. code-block:: python

            # Step 1 — list addresses
            summary, addresses = tool.invoke({"exchange": "polymarket"})

            # Step 2 — fetch fills for one address
            summary, df = tool.invoke({"exchange": "polymarket", "address": "0xabc..."})
    """

    name: str = "convexlake_leaderboard_trades"
    description: str = (
        "List trader addresses or fetch one trader's fill history. "
        "Without address=: returns the top trader addresses (artifact is a list). "
        "With address=: returns that trader's fill history (artifact is a DataFrame). "
        "Supported exchanges: polymarket, limitless, predict."
    )
    response_format: Literal["content_and_artifact"] = "content_and_artifact"
    max_rows: int = 20
    max_addresses: int = 20

    def _run(
        self,
        exchange: Literal["polymarket", "limitless", "predict"],
        address: str | None = None,
    ) -> tuple[str, Any]:
        """Run the leaderboard trades tool.

        Args:
            exchange: Exchange name.
            address: Wallet address (0x...). If None, returns the address list.

        Returns:
            Tuple of (summary string, addresses list or full DataFrame).
        """
        try:
            if address is None:
                addresses = self._client.pm.traders(exchange)
                shown = addresses[: self.max_addresses]
                content = (
                    f"total traders: {len(addresses)}\n"
                    f"first {len(shown)}:\n" + "\n".join(shown)
                )
                return content, addresses
            df = self._client.pm.trader_fills(exchange, address)
            return _summarise(df, self.max_rows)
        except Exception as e:
            return self._error(e, "trader data"), None


class ConvexLakeOrderbookSummary(_ConvexLakeBaseTool):
    """Return a summary of an orderbook file: message count, types, timestamps, best BBO.

    Setup:
        Install ``langchain-convexlake`` and set ``CONVEXLAKE_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-convexlake
            export CONVEXLAKE_API_KEY="do_..."

    Instantiation:
        .. code-block:: python

            from langchain_convexlake import ConvexLakeOrderbookSummary

            tool = ConvexLakeOrderbookSummary()

    Invocation:
        .. code-block:: python

            result = tool.invoke({
                "exchange": "polymarket", "ticker": "btc",
                "timeframe": "15m", "date": "2026-10-01",
                "slug": "btc-15m-123.gz",
            })
    """

    name: str = "convexlake_orderbook_summary"
    description: str = (
        "Summarise an orderbook file: message count, type breakdown, first/last "
        "timestamps, and first/last best bid/ask from snapshots. "
        "Never returns raw price levels or order sizes. "
        "Supported exchanges: polymarket, kalshi, limitless, predict, deribit, "
        "binance, hyperliquid, okx, bybit, bitfinex."
    )

    def _run(
        self,
        exchange: str,
        ticker: str,
        timeframe: str,
        date: str,
        slug: str | None = None,
        md_type: str | None = None,
    ) -> str:
        """Run the orderbook summary tool.

        Args:
            exchange: Exchange name.
            ticker: Instrument ticker.
            timeframe: File timeframe, e.g. "15m", "1h".
            date: Date in YYYY-MM-DD.
            slug: File slug — required when multiple files exist.
            md_type: Optional "bbo" for best-bid/ask-only files.

        Returns:
            Plain-text summary string.
        """
        PM_EXCHANGES = {"polymarket", "kalshi", "limitless", "predict"}
        try:
            if exchange in PM_EXCHANGES:
                msgs = self._client.pm.orderbook(
                    exchange, ticker, timeframe, date, slug=slug, md_type=md_type
                )
            else:
                result = self._client.cx.orderbook(
                    exchange, ticker, timeframe, date, slug=slug
                )
                if not isinstance(result, list):
                    df = result
                    return (
                        f"Orderbook (CSV, {exchange}): "
                        f"{len(df)} rows, columns: {list(df.columns)}"
                    )
                msgs = result

            if not msgs:
                return "Orderbook is empty."

            types: dict[str, int] = {}
            for m in msgs:
                t = m.get("type") or m.get("event_type", "unknown")
                types[t] = types.get(t, 0) + 1

            def _ts(m: dict[str, Any]) -> int | None:
                for f in ("local_timestamp", "timestamp", "ts", "time", "created_at"):
                    v = m.get(f)
                    if v is not None:
                        return int(v)
                return None

            timestamps = [t for m in msgs if (t := _ts(m)) is not None]
            ts_summary = ""
            if timestamps:
                ts_summary = (
                    f"\nfirst_ts: {min(timestamps)}, last_ts: {max(timestamps)}"
                )

            def _best_bbo(
                m: dict[str, Any],
            ) -> tuple[float | None, float | None]:
                bids = m.get("bids") or []
                asks = m.get("asks") or []
                best_bid = max(
                    (float(b[0]) for b in bids if b), default=None
                )
                best_ask = min(
                    (float(a[0]) for a in asks if a), default=None
                )
                return best_bid, best_ask

            snap_types = {"orderbook_snapshot", "snapshot", "book_snapshot"}
            snapshots = [
                m
                for m in msgs
                if (m.get("type") or m.get("event_type", "")) in snap_types
            ]
            bbo_summary = ""
            if snapshots:
                fb, fa = _best_bbo(snapshots[0])
                lb, la = _best_bbo(snapshots[-1])
                if fb is not None or fa is not None:
                    bbo_summary = (
                        f"\nfirst_best_bid: {fb}, first_best_ask: {fa}"
                        f"\nlast_best_bid: {lb}, last_best_ask: {la}"
                    )

            return (
                f"messages: {len(msgs)}"
                f"\ntypes: {types}"
                f"{ts_summary}"
                f"{bbo_summary}"
            )
        except Exception as e:
            return self._error(e, "orderbook")


# ── convenience factory ───────────────────────────────────────────────────────

def get_tools(
    api_key: str | None = None,
    *,
    max_rows: int = 20,
    max_addresses: int = 20,
) -> list[BaseTool]:
    """Return all ConvexLake tools configured with a shared API key.

    Args:
        api_key: ConvexLake API key. Falls back to ``CONVEXLAKE_API_KEY`` env var.
        max_rows: Maximum DataFrame rows to include in text summaries.
        max_addresses: Maximum trader addresses to show in leaderboard tool.

    Returns:
        List of seven :class:`BaseTool` instances.

    Example:
        .. code-block:: python

            import os
            from langchain.agents import create_agent
            from langchain_convexlake import get_tools

            tools = get_tools(api_key=os.environ["CONVEXLAKE_API_KEY"])
            agent = create_agent(model="anthropic:claude-sonnet-5-5", tools=tools)
    """
    import os  # noqa: PLC0415

    key = api_key or os.environ.get("CONVEXLAKE_API_KEY") or ""
    kwargs: dict[str, Any] = {"api_key": convert_to_secret_str(key)}
    return [
        ConvexLakeCatalog(**kwargs),
        ConvexLakeTrades(**kwargs, max_rows=max_rows),
        ConvexLakeCandles(**kwargs, max_rows=max_rows),
        ConvexLakeSurface(**kwargs, max_rows=max_rows),
        ConvexLakeLeaderboard(**kwargs, max_rows=max_rows),
        ConvexLakeLeaderboardTrades(
            **kwargs, max_rows=max_rows, max_addresses=max_addresses
        ),
        ConvexLakeOrderbookSummary(**kwargs),
    ]
