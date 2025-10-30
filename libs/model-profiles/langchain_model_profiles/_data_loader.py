"""Data loader for model profiles with augmentation support."""

import json
import sys
from functools import cached_property
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class _DataLoader:
    """Loads and merges model profile data from base and augmentations."""

    def __init__(self) -> None:
        """Initialize the data loader."""
        self._data_dir = Path(__file__).parent / "data"

    @property
    def _base_data_path(self) -> Path:
        """Get path to base data file."""
        return self._data_dir / "models.json"

    @property
    def _augmentations_dir(self) -> Path:
        """Get path to augmentations directory."""
        return self._data_dir / "augmentations"

    @cached_property
    def _merged_data(self) -> dict[str, Any]:
        """Load and merge all data once at startup.

        Returns:
            Fully merged provider data with all augmentations applied.
        """
        # Load base data
        with self._base_data_path.open("r") as f:
            data = json.load(f)

        provider_augmentations = self._load_provider_augmentations()
        model_augmentations = self._load_model_augmentations()

        # Merge everything
        for provider_id, provider_data in data.items():
            models = provider_data.get("models", {})
            provider_aug = provider_augmentations.get(provider_id, {})

            for model_id, model_data in models.items():
                if provider_aug:
                    model_data.update(provider_aug)

                # Apply model-level augmentations (highest priority)
                model_aug = model_augmentations.get(provider_id, {}).get(model_id, {})
                if model_aug:
                    model_data.update(model_aug)

        return data

    def _load_provider_augmentations(self) -> dict[str, dict[str, Any]]:
        """Load all provider-level augmentations.

        Returns:
            Dictionary mapping provider IDs to their augmentation data.
        """
        augmentations: dict[str, dict[str, Any]] = {}
        providers_dir = self._augmentations_dir / "providers"

        if not providers_dir.exists():
            return augmentations

        for toml_file in providers_dir.glob("*.toml"):
            provider_id = toml_file.stem
            with toml_file.open("rb") as f:
                data = tomllib.load(f)
                if "profile" in data:
                    augmentations[provider_id] = data["profile"]

        return augmentations

    def _load_model_augmentations(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Load all model-level augmentations.

        Returns:
            Nested dictionary: provider_id -> model_id -> augmentation data.
        """
        augmentations: dict[str, dict[str, dict[str, Any]]] = {}
        models_dir = self._augmentations_dir / "models"

        if not models_dir.exists():
            return augmentations

        for provider_dir in models_dir.iterdir():
            if not provider_dir.is_dir():
                continue

            provider_id = provider_dir.name
            augmentations[provider_id] = {}

            for toml_file in provider_dir.glob("*.toml"):
                model_id = toml_file.stem
                with toml_file.open("rb") as f:
                    data = tomllib.load(f)
                    if "profile" in data:
                        augmentations[provider_id][model_id] = data["profile"]

        return augmentations

    def get_profile_data(
        self, provider_id: str, model_id: str
    ) -> dict[str, Any] | None:
        """Get merged profile data for a specific model.

        Args:
            provider_id: The provider identifier.
            model_id: The model identifier.

        Returns:
            Merged model data dictionary or None if not found.
        """
        provider = self._merged_data.get(provider_id)
        if provider is None:
            return None

        models = provider.get("models", {})
        return models.get(model_id)
