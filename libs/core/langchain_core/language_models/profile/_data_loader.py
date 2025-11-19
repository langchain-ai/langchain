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
    """Loads and merges model profile data from base and augmentations.

    See the README in `data/augmentations` directory for more details on the
    augmentation structure and merge priority.
    """

    def __init__(self, data_dir: Path) -> None:
        """Initialize the loader.

        Args:
            data_dir: Path to the data directory containing models.json and
                augmentations.
        """
        self._data_dir = data_dir

    @property
    def _base_data_path(self) -> Path:
        """Get path to base data file.

        `models.json` is the downloaded data from models.dev.
        """
        return self._data_dir / "models.json"

    @property
    def _augmentations_dir(self) -> Path:
        """Get path to augmentations directory."""
        return self._data_dir / "augmentations"

    @cached_property
    def _merged_data(self) -> dict[str, Any]:
        """Load and merge all data once at startup.

        Merging order:

        1. Base data from `models.json`
        2. Provider-level augmentations from `augmentations/providers/{provider}.toml`
        3. Model-level augmentations from `augmentations/models/{provider}/{model}.toml`

        Returns:
            Fully merged provider data with all augmentations applied.
        """
        # Load base data; let exceptions propagate to user
        with self._base_data_path.open("r") as f:
            data = json.load(f)

        provider_augmentations = self._load_provider_augmentations()
        model_augmentations = self._load_model_augmentations()

        # Merge contents
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
        """Load provider-level augmentations from profiles.toml.

        Returns:
            `dict` mapping provider IDs to their augmentation data.
        """
        augmentations: dict[str, dict[str, Any]] = {}
        profiles_file = self._augmentations_dir / "profiles.toml"

        if not profiles_file.exists():
            return augmentations

        with profiles_file.open("rb") as f:
            data = tomllib.load(f)
            if "profile" in data:
                # Load all provider IDs from base data and apply augmentation to all
                try:
                    with self._base_data_path.open("r") as base_f:
                        base_data = json.load(base_f)
                        for provider_id in base_data.keys():
                            augmentations[provider_id] = data["profile"]
                except (OSError, json.JSONDecodeError):
                    pass

        return augmentations

    def _load_model_augmentations(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Load all model-level augmentations.

        Returns:
            Nested `dict`: `provider_id` -> `model_id` -> augmentation data.
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
            Merged model data `dict` or `None` if not found.
        """
        provider = self._merged_data.get(provider_id)
        if provider is None:
            return None

        models = provider.get("models", {})
        return models.get(model_id)
