"""AINetwork Blockchain tool utils."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from ain.ain import Ain


def authenticate(network: Optional[Literal["mainnet", "testnet"]] = "testnet") -> Ain:
    """Authenticate using the AIN Blockchain"""

    try:
        from ain.ain import Ain
    except ImportError as e:
        raise ImportError(
            "Cannot import ain-py related modules. Please install the package with "
            "`pip install ain-py`."
        ) from e

    if network == "mainnet":
        provider_url = "https://mainnet-api.ainetwork.ai/"
        chain_id = 1
        if "AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY" in os.environ:
            private_key = os.environ["AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY"]
        else:
            raise EnvironmentError(
                "Error: The AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY environmental variable "
                "has not been set."
            )
    elif network == "testnet":
        provider_url = "https://testnet-api.ainetwork.ai/"
        chain_id = 0
        if "AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY" in os.environ:
            private_key = os.environ["AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY"]
        else:
            raise EnvironmentError(
                "Error: The AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY environmental variable "
                "has not been set."
            )
    elif network is None:
        if (
            "AIN_BLOCKCHAIN_PROVIDER_URL" in os.environ
            and "AIN_BLOCKCHAIN_CHAIN_ID" in os.environ
            and "AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY" in os.environ
        ):
            provider_url = os.environ["AIN_BLOCKCHAIN_PROVIDER_URL"]
            chain_id = int(os.environ["AIN_BLOCKCHAIN_CHAIN_ID"])
            private_key = os.environ["AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY"]
        else:
            raise EnvironmentError(
                "Error: The AIN_BLOCKCHAIN_PROVIDER_URL and "
                "AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY and AIN_BLOCKCHAIN_CHAIN_ID "
                "environmental variable has not been set."
            )
    else:
        raise ValueError(f"Unsupported 'network': {network}")

    ain = Ain(provider_url, chain_id)
    ain.wallet.addAndSetDefaultAccount(private_key)
    return ain
