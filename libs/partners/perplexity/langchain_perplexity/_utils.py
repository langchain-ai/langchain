import os
from importlib import metadata
from typing import Any

from langchain_core.utils import convert_to_secret_str
from perplexity import Perplexity

_INTEGRATION_HEADER = "X-Pplx-Integration"
_INTEGRATION_NAME = "langchain"
_PACKAGE_NAME = "langchain-perplexity"


def get_integration_headers() -> dict[str, str]:
    """Return the Perplexity attribution headers for this integration."""
    package_version = metadata.version(_PACKAGE_NAME)
    return {_INTEGRATION_HEADER: f"{_INTEGRATION_NAME}/{package_version}"}


def initialize_client(values: dict[str, Any]) -> dict[str, Any]:
    """Initialize the Perplexity client."""
    pplx_api_key = (
        values.get("pplx_api_key")
        or os.environ.get("PPLX_API_KEY")
        or os.environ.get("PERPLEXITY_API_KEY")
        or ""
    )
    values["pplx_api_key"] = convert_to_secret_str(pplx_api_key)

    api_key = (
        values["pplx_api_key"].get_secret_value() if values["pplx_api_key"] else None
    )

    if not values.get("client"):
        values["client"] = Perplexity(
            api_key=api_key, default_headers=get_integration_headers()
        )

    return values
