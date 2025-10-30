"""Minimal Python SDK for models.dev API.

Temporary solution until we persist data in the package.
"""

from functools import cached_property
from typing import Any

import httpx


class _ModelsDevClient:
    API_URL = "https://models.dev/api.json"

    def __init__(self, timeout: int = 30) -> None:
        self._timeout = timeout

    @cached_property
    def _data(self) -> dict[str, Any]:
        """Fetch data from the API (cached)."""
        # TODO: ttl
        response = httpx.get(self.API_URL, timeout=self._timeout)
        response.raise_for_status()
        return response.json()

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
        provider = self._data.get(provider_id)
        if provider is None:
            return None

        models = provider.get("models", {})
        return models.get(model_id)
