"""Utilities for ShopSavvy API integration."""

from __future__ import annotations

import os

from shopsavvy import ShopSavvyConfig, ShopSavvyDataAPI  # type: ignore[import-untyped]

from langchain_core.utils import convert_to_secret_str


def initialize_client(values: dict) -> dict:
    """Initialize the ShopSavvy API client.

    Args:
        values: Dictionary of field values to validate and transform.

    Returns:
        Updated dictionary with initialized client and secret key.
    """
    api_key = (
        values.get("shopsavvy_api_key")
        or os.environ.get("SHOPSAVVY_API_KEY")
        or ""
    )
    values["shopsavvy_api_key"] = convert_to_secret_str(api_key)
    config = ShopSavvyConfig(
        api_key=values["shopsavvy_api_key"].get_secret_value(),
    )
    values["client"] = ShopSavvyDataAPI(config)
    return values
