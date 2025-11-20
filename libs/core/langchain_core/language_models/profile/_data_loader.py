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
    def _augmentations_file(self) -> Path:
        """Get path to profile augmentations file."""
        return self._data_dir / "profile_augmentations.toml"

    @cached_property
    def _merged_data(self) -> dict[str, Any]:
        """Load and merge all data once at startup.

        Merging order:

        1. Base data from `models.json`
        2. Provider-level augmentations from `[overrides]` in `profile_augmentations.toml`
        3. Model-level augmentations from `[overrides."model-name"]` in `profile_augmentations.toml`

        Returns:
            Fully merged provider data with all augmentations applied.
        """
        # Load base data; let exceptions propagate to user
        with self._base_data_path.open("r") as f:
            data = json.load(f)

        # Load augmentations from profile_augmentations.toml
        provider_aug, model_augs = self._load_augmentations()

        # Merge augmentations into data
        for provider_id, provider_data in data.items():
            models = provider_data.get("models", {})

            for model_id, model_data in models.items():
                # Apply provider-level augmentations
                if provider_aug:
                    model_data.update(provider_aug)

                # Apply model-level augmentations (highest priority)
                if model_id in model_augs:
                    model_data.update(model_augs[model_id])

        return data

    def _load_augmentations(
        self,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Load augmentations from profile_augmentations.toml.

        Returns:
            Tuple of (provider_augmentations, model_augmentations) where:
            - provider_augmentations: dict of fields to apply to all models
            - model_augmentations: dict mapping model IDs to their specific augmentations
        """
        if not self._augmentations_file.exists():
            return {}, {}

        with self._augmentations_file.open("rb") as f:
            data = tomllib.load(f)

        overrides = data.get("overrides", {})

        # Separate provider-level augmentations from model-specific ones
        # Model-specific overrides are nested dicts, while provider-level are primitives
        provider_aug: dict[str, Any] = {}
        model_augs: dict[str, dict[str, Any]] = {}

        for key, value in overrides.items():
            if isinstance(value, dict):
                # This is a model-specific override like [overrides."claude-sonnet-4-5"]
                model_augs[key] = value
            else:
                # This is a provider-level field
                provider_aug[key] = value

        return provider_aug, model_augs

    def get_profile_data(self, provider_id: str) -> dict[str, Any] | None:
        """Get merged profile data for all models.

        Args:
            provider_id: The provider identifier.

        Returns:
            Merged model data `dict` or `None` if not found.
        """
        provider = self._merged_data.get(provider_id)
        if provider is None:
            return None

        return provider.get("models", {})
