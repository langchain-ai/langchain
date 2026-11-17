"""Client initialisation utilities."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.utils import convert_to_secret_str


def initialize_client(values: dict[str, Any]) -> dict[str, Any]:
    """Read API key from values or environment and build the ConvexLake client."""
    from convexlake import ConvexLake  # noqa: PLC0415

    api_key = values.get("api_key") or os.environ.get("CONVEXLAKE_API_KEY") or ""
    values["api_key"] = convert_to_secret_str(api_key)
    values["_client"] = ConvexLake(api_key=values["api_key"].get_secret_value())
    return values
