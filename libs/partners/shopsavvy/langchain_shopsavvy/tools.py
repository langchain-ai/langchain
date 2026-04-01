"""Tools for the ShopSavvy Data API."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import Field, SecretStr, model_validator
from shopsavvy import ShopSavvyDataAPI  # type: ignore[import-untyped]

from langchain_shopsavvy._utilities import initialize_client


class ShopSavvyProductSearch(BaseTool):  # type: ignore[override]
    """Search for products across ShopSavvy's database.

    Setup:
        Install ``langchain-shopsavvy`` and set environment variable
        ``SHOPSAVVY_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-shopsavvy
            export SHOPSAVVY_API_KEY="ss_live_your_api_key"

    Instantiation:
        .. code-block:: python

            from langchain_shopsavvy import ShopSavvyProductSearch

            tool = ShopSavvyProductSearch()

    Invocation with args:
        .. code-block:: python

            tool.invoke({"query": "iphone 15 pro", "max_results": 3})

    Invocation with ToolCall:
        .. code-block:: python

            tool.invoke(
                {
                    "args": {"query": "iphone 15 pro", "max_results": 3},
                    "id": "1",
                    "name": tool.name,
                    "type": "tool_call",
                }
            )
    """

    name: str = "shopsavvy_product_search"
    description: str = (
        "Search for products by keyword across ShopSavvy's database of 100M+ "
        "products. Returns product names, brands, categories, barcodes, and "
        "identifiers. Use this to find products before looking up prices."
    )
    client: ShopSavvyDataAPI = Field(default=None)  # type: ignore[assignment]
    shopsavvy_api_key: SecretStr = Field(default=SecretStr(""))

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate the environment and initialize the client."""
        return initialize_client(values)

    def _run(
        self,
        query: str,
        max_results: int = 5,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Search for products by keyword.

        Args:
            query: The search query (e.g. "iphone 15 pro", "samsung tv").
            max_results: Maximum number of results to return (1 to 100).
                Defaults to 5.
            run_manager: The run manager for callbacks.

        Returns:
            JSON string of the product search results.
        """
        try:
            max_results = max(1, min(100, max_results))
            result = self.client.search_products(query, limit=max_results)
            products = []
            for product in result.data:
                products.append({
                    "title": product.title,
                    "brand": product.brand,
                    "category": product.category,
                    "barcode": product.barcode,
                    "asin": product.amazon,
                    "shopsavvy_id": product.shopsavvy,
                    "model": product.model,
                })
            return json.dumps(products, indent=2)
        except Exception as e:
            return repr(e)


class ShopSavvyPriceComparison(BaseTool):  # type: ignore[override]
    """Get current prices for a product from all retailers.

    Setup:
        Install ``langchain-shopsavvy`` and set environment variable
        ``SHOPSAVVY_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-shopsavvy
            export SHOPSAVVY_API_KEY="ss_live_your_api_key"

    Instantiation:
        .. code-block:: python

            from langchain_shopsavvy import ShopSavvyPriceComparison

            tool = ShopSavvyPriceComparison()

    Invocation with args:
        .. code-block:: python

            tool.invoke({"identifier": "B0CHXCQ5SR"})
    """

    name: str = "shopsavvy_price_comparison"
    description: str = (
        "Get current prices for a product from all retailers. Input a product "
        "identifier (barcode, ASIN, ShopSavvy ID, URL, or model number). "
        "Returns offers sorted by price."
    )
    client: ShopSavvyDataAPI = Field(default=None)  # type: ignore[assignment]
    shopsavvy_api_key: SecretStr = Field(default=SecretStr(""))

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate the environment and initialize the client."""
        return initialize_client(values)

    def _run(
        self,
        identifier: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Get current offers for a product.

        Args:
            identifier: Product identifier (barcode, ASIN, ShopSavvy ID, URL,
                or model number).
            run_manager: The run manager for callbacks.

        Returns:
            JSON string of the current offers sorted by price.
        """
        try:
            result = self.client.get_current_offers(identifier)
            offers = []
            for product in result.data:
                for offer in product.offers:
                    offers.append({
                        "retailer": offer.retailer,
                        "price": offer.price,
                        "currency": offer.currency,
                        "availability": offer.availability,
                        "condition": offer.condition,
                        "url": offer.url,
                        "seller": offer.seller,
                        "last_updated": offer.timestamp,
                    })
            offers.sort(key=lambda x: x.get("price") or float("inf"))
            return json.dumps(offers, indent=2)
        except Exception as e:
            return repr(e)


class ShopSavvyPriceHistory(BaseTool):  # type: ignore[override]
    """Get historical price data for a product.

    Setup:
        Install ``langchain-shopsavvy`` and set environment variable
        ``SHOPSAVVY_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-shopsavvy
            export SHOPSAVVY_API_KEY="ss_live_your_api_key"

    Instantiation:
        .. code-block:: python

            from langchain_shopsavvy import ShopSavvyPriceHistory

            tool = ShopSavvyPriceHistory()

    Invocation with args:
        .. code-block:: python

            tool.invoke({"identifier": "B0CHXCQ5SR", "days_back": 30})
    """

    name: str = "shopsavvy_price_history"
    description: str = (
        "Get historical price data for a product. Shows how prices changed "
        "over time. Useful for determining if current price is a good deal."
    )
    client: ShopSavvyDataAPI = Field(default=None)  # type: ignore[assignment]
    shopsavvy_api_key: SecretStr = Field(default=SecretStr(""))

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate the environment and initialize the client."""
        return initialize_client(values)

    def _run(
        self,
        identifier: str,
        days_back: int = 30,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Get price history for a product.

        Args:
            identifier: Product identifier (barcode, ASIN, ShopSavvy ID, URL,
                or model number).
            days_back: Number of days of history to retrieve (1 to 365).
                Defaults to 30.
            run_manager: The run manager for callbacks.

        Returns:
            JSON string with per-retailer price statistics including min, max,
            and average prices.
        """
        try:
            days_back = max(1, min(365, days_back))
            end_date = datetime.now()  # noqa: DTZ005
            start_date = end_date - timedelta(days=days_back)
            result = self.client.get_price_history(
                identifier,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )
            history_summary = []
            for offer in result.data:
                prices = [
                    entry.price
                    for entry in offer.price_history
                    if entry.price is not None
                ]
                summary: dict[str, Any] = {
                    "retailer": offer.retailer,
                    "data_points": len(offer.price_history),
                }
                if prices:
                    summary["min_price"] = min(prices)
                    summary["max_price"] = max(prices)
                    summary["avg_price"] = round(sum(prices) / len(prices), 2)
                history_summary.append(summary)
            return json.dumps(history_summary, indent=2)
        except Exception as e:
            return repr(e)
