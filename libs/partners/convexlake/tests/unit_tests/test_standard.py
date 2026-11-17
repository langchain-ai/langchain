"""Standard LangChain tool compliance tests."""

import pytest
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_convexlake import (
    ConvexLakeCatalog,
    ConvexLakeCandles,
    ConvexLakeLeaderboard,
    ConvexLakeLeaderboardTrades,
    ConvexLakeOrderbookSummary,
    ConvexLakeSurface,
    ConvexLakeTrades,
)

_FAKE_KEY = "do_test_fake_key"


class TestConvexLakeCatalogStandard(ToolsUnitTests):
    @property
    def tool_constructor(self):
        return ConvexLakeCatalog

    @property
    def tool_constructor_params(self):
        return {"api_key": _FAKE_KEY}

    @property
    def tool_invoke_params_example(self):
        return {}


class TestConvexLakeTradesStandard(ToolsUnitTests):
    @property
    def tool_constructor(self):
        return ConvexLakeTrades

    @property
    def tool_constructor_params(self):
        return {"api_key": _FAKE_KEY}

    @property
    def tool_invoke_params_example(self):
        return {
            "exchange": "kalshi",
            "ticker": "crypto",
            "date": "2026-10-01",
            "slug": "example.csv",
        }


class TestConvexLakeCandlesStandard(ToolsUnitTests):
    @property
    def tool_constructor(self):
        return ConvexLakeCandles

    @property
    def tool_constructor_params(self):
        return {"api_key": _FAKE_KEY}

    @property
    def tool_invoke_params_example(self):
        return {
            "exchange": "kalshi",
            "ticker": "crypto",
            "date": "2026-10-01",
            "slug": "example.csv",
        }


class TestConvexLakeSurfaceStandard(ToolsUnitTests):
    @property
    def tool_constructor(self):
        return ConvexLakeSurface

    @property
    def tool_constructor_params(self):
        return {"api_key": _FAKE_KEY}

    @property
    def tool_invoke_params_example(self):
        return {"exchange": "deribit", "ticker": "BTC", "date": "2026-10-01"}


class TestConvexLakeLeaderboardStandard(ToolsUnitTests):
    @property
    def tool_constructor(self):
        return ConvexLakeLeaderboard

    @property
    def tool_constructor_params(self):
        return {"api_key": _FAKE_KEY}

    @property
    def tool_invoke_params_example(self):
        return {"exchange": "polymarket", "date": "2026-10-01"}


class TestConvexLakeLeaderboardTradesStandard(ToolsUnitTests):
    @property
    def tool_constructor(self):
        return ConvexLakeLeaderboardTrades

    @property
    def tool_constructor_params(self):
        return {"api_key": _FAKE_KEY}

    @property
    def tool_invoke_params_example(self):
        return {"exchange": "polymarket"}


class TestConvexLakeOrderbookSummaryStandard(ToolsUnitTests):
    @property
    def tool_constructor(self):
        return ConvexLakeOrderbookSummary

    @property
    def tool_constructor_params(self):
        return {"api_key": _FAKE_KEY}

    @property
    def tool_invoke_params_example(self):
        return {
            "exchange": "polymarket",
            "ticker": "btc",
            "timeframe": "15m",
            "date": "2026-10-01",
        }
