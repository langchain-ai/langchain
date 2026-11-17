# langchain-convexlake

LangChain tools for [ConvexLake](https://convexlake.com) — a market-data API covering prediction markets (Polymarket, Kalshi, Limitless, Predict.fun) and crypto derivatives (Deribit, Binance, Hyperliquid, OKX, Bybit, Bitfinex).

## Installation

```bash
pip install langchain-convexlake
```

Get a free API key at [convexlake.com](https://convexlake.com).

## Quick start

```python
import os
from langchain.agents import create_agent
from langchain_convexlake import get_tools

tools = get_tools(api_key=os.environ["CONVEXLAKE_API_KEY"])
agent = create_agent(model="anthropic:claude-sonnet-5-5", tools=tools)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Who are the top Polymarket traders today?"}]
})
print(result["messages"][-1].content)
```

## Tools

| Tool | Description |
|------|-------------|
| `convexlake_catalog` | Browse available exchanges, tickers, dates, and file slugs |
| `convexlake_trades` | Trade history for prediction markets and options |
| `convexlake_candles` | OHLCV candle data for prediction markets and crypto perpetuals |
| `convexlake_surface` | Implied-volatility surface snapshots (Deribit, Binance, OKX, Bybit) |
| `convexlake_leaderboard` | Daily trader leaderboard (Polymarket, Limitless, Predict) |
| `convexlake_leaderboard_trades` | Individual trader fill history |
| `convexlake_orderbook_summary` | Orderbook statistics: message types, timestamps, best BBO |

Tools that return tabular data use `response_format="content_and_artifact"`: the LLM receives a text summary while `ToolMessage.artifact` holds the full pandas DataFrame.

## Individual tool usage

```python
from langchain_convexlake import ConvexLakeCatalog, ConvexLakeTrades

catalog = ConvexLakeCatalog(api_key="do_...")
print(catalog.invoke({}))                               # list exchanges
print(catalog.invoke({"exchange": "kalshi"}))           # list tickers

trades = ConvexLakeTrades(api_key="do_...")
summary, df = trades.func(
    exchange="kalshi", ticker="crypto",
    date="2026-10-01", slug="KXBTC15M-26OCT01-trades.csv",
)
```

## Development

```bash
uv sync --dev
make test_unit
make test_integration   # requires CONVEXLAKE_API_KEY
```

## License

MIT
