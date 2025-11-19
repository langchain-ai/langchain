"""Utilities for loading model profiles from provider packages."""

from functools import lru_cache
from pathlib import Path

from langchain_core.language_models.profile._data_loader import _DataLoader
from langchain_core.language_models.profile.model_profile import (
    ModelProfileRegistry,
    map_raw_data_to_profile,
)


def load_profiles_from_data_dir(
    data_dir: Path, provider_id: str
) -> ModelProfileRegistry | None:
    """Load model profiles from a provider's data directory.

    Args:
        data_dir: Path to the provider's data directory.
        provider_id: The provider identifier (e.g., 'anthropic', 'openai').

    Returns:
        ModelProfile with model capabilities, or None if not found.
    """
    loader = _get_loader(data_dir)
    data = loader.get_profile_data(provider_id)
    if not data:
        return None
    return {
        model_name: map_raw_data_to_profile(raw_profile)
        for model_name, raw_profile in data.items()
    }


@lru_cache(maxsize=32)
def _get_loader(data_dir: Path) -> _DataLoader:
    """Get a cached loader for a data directory.

    Args:
        data_dir: Path to the data directory.

    Returns:
        DataLoader instance.
    """
    return _DataLoader(data_dir)
