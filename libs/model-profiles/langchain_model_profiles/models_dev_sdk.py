"""Minimal Python SDK for models.dev API.

Temporary solution until we persist data in the package.
"""

from typing import Any

import requests


class _ModelsDevClient:
    """Client for interacting with the models.dev API."""

    API_URL = "https://models.dev/api.json"

    def __init__(self) -> None:
        """Initialize the client."""
        self._data: dict[str, Any] | None = None

    def _fetch_data(self) -> dict[str, Any]:
        """Fetch data from the API (with caching)."""
        if self._data is not None:
            return self._data

        response = requests.get(self.API_URL, timeout=30)
        response.raise_for_status()
        self._data = response.json()

        return self._data

    def get_profile_data(
        self, provider_id: str, model_id: str
    ) -> dict[str, Any] | None:
        """Get a specific model from a provider.

        Args:
            provider_id: The provider identifier.
            model_id: The model identifier.

        Returns:
            Model data dictionary or None if not found.
        """
        data = self._fetch_data()
        provider = data.get(provider_id)
        if provider is None:
            return None

        models = provider.get("models", {})
        return models.get(model_id)

    def clear_cache(self) -> None:
        """Clear the cached API data."""
        self._data = None
