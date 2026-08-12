"""Utility functions for the langchain-ipfs package."""

from typing import Any

from langchain_core.utils import convert_to_secret_str
from pydantic import SecretStr


def initialize_client(values: dict) -> dict:
    """Initialize the ipfs-pay-to-pin client.

    Args:
        values: Pydantic model values dict.

    Returns:
        Updated values dict with client initialized.
    """
    api_key = values.get("api_key") or (
        SecretStr("")
        if isinstance(values.get("api_key"), SecretStr)
        else ""
    )
    api_key_value = (
        api_key.get_secret_value()
        if isinstance(api_key, SecretStr)
        else str(api_key)
    ) or ""

    # Check environment variable fallback
    import os  # type: ignore[import-not-found]

    env_key = os.environ.get("IPFS_PIN_API_KEY", "") or api_key_value
    values["api_key"] = convert_to_secret_str(env_key)

    try:
        from ipfs_pay_to_pin_client import PinClient  # type: ignore[import-untyped]

        client_kwargs: dict[str, Any] = {
            "api_key": values["api_key"].get_secret_value(),
        }
        if values.get("base_url"):
            client_kwargs["base_url"] = values["base_url"]
        if values.get("chain"):
            client_kwargs["chain"] = values["chain"]
        if values.get("max_price_usdc") is not None:
            client_kwargs["max_price_usdc"] = values["max_price_usdc"]

        values["client"] = PinClient(**client_kwargs)  # type: ignore[arg-type]
    except ImportError as e:
        raise ImportError(
            "The ipfs-pay-to-pin-client package is required to use IPFSPinTool. "
            "Install it with: pip install ipfs-pay-to-pin-client"
        ) from e

    return values
