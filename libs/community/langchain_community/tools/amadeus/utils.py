"""O365 tool utils."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from amadeus import Client

logger = logging.getLogger(__name__)


def authenticate() -> Client:
    """Authenticate using the Amadeus API"""
    try:
        from amadeus import Client
    except ImportError as e:
        raise ImportError(
            "Cannot import amadeus. Please install the package with "
            "`pip install amadeus`."
        ) from e

    if "AMADEUS_CLIENT_ID" in os.environ and "AMADEUS_CLIENT_SECRET" in os.environ:
        client_id = os.environ["AMADEUS_CLIENT_ID"]
        client_secret = os.environ["AMADEUS_CLIENT_SECRET"]
    else:
        logger.error(
            "Error: The AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET environmental "
            "variables have not been set. Visit the following link on how to "
            "acquire these authorization tokens: "
            "https://developers.amadeus.com/register"
        )
        return None

    hostname = "test"  # Default hostname
    if "AMADEUS_HOSTNAME" in os.environ:
        hostname = os.environ["AMADEUS_HOSTNAME"]

    client = Client(client_id=client_id, client_secret=client_secret, hostname=hostname)

    return client
