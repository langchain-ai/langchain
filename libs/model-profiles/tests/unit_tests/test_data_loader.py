"""Tests for data loader with augmentation support."""

import json
from pathlib import Path

import pytest

from langchain_model_profiles._data_loader import _DataLoader


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory structure."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create base models.json
    base_data = {
        "test-provider": {
            "id": "test-provider",
            "models": {
                "test-model": {
                    "id": "test-model",
                    "name": "Test Model",
                    "tool_call": True,
                    "limit": {"context": 8000, "output": 4000},
                    "modalities": {"input": ["text"], "output": ["text"]},
                }
            },
        }
    }

    with (data_dir / "models.json").open("w") as f:
        json.dump(base_data, f)

    # Create augmentations directories
    aug_dir = data_dir / "augmentations"
    (aug_dir / "providers").mkdir(parents=True)
    (aug_dir / "models" / "test-provider").mkdir(parents=True)

    return data_dir


def test_load_base_data_only(temp_data_dir: Path) -> None:
    """Test loading base data without augmentations."""
    loader = _DataLoader()
    # Patch before any property access
    loader._data_dir = temp_data_dir
    result = loader.get_profile_data("test-provider", "test-model")

    assert result is not None
    assert result["id"] == "test-model"
    assert result["name"] == "Test Model"
    assert result["tool_call"] is True


def test_provider_level_augmentation(temp_data_dir: Path) -> None:
    """Test provider-level augmentations are applied."""
    # Add provider augmentation
    provider_toml = temp_data_dir / "augmentations" / "providers" / "test-provider.toml"
    provider_toml.write_text("""
[profile]
image_url_inputs = true
pdf_inputs = true
""")

    loader = _DataLoader()
    loader._data_dir = temp_data_dir
    result = loader.get_profile_data("test-provider", "test-model")

    assert result is not None
    assert result["image_url_inputs"] is True
    assert result["pdf_inputs"] is True
    # Base data should still be present
    assert result["tool_call"] is True


def test_model_level_augmentation_overrides_provider(temp_data_dir: Path) -> None:
    """Test model-level augmentations override provider augmentations."""
    # Add provider augmentation
    provider_toml = temp_data_dir / "augmentations" / "providers" / "test-provider.toml"
    provider_toml.write_text("""
[profile]
image_url_inputs = true
pdf_inputs = false
""")

    # Add model augmentation that overrides
    model_toml = (
        temp_data_dir / "augmentations" / "models" / "test-provider" / "test-model.toml"
    )
    model_toml.write_text("""
[profile]
pdf_inputs = true
reasoning_output = true
""")

    loader = _DataLoader()
    loader._data_dir = temp_data_dir
    result = loader.get_profile_data("test-provider", "test-model")

    assert result is not None
    # From provider
    assert result["image_url_inputs"] is True
    # Overridden by model
    assert result["pdf_inputs"] is True
    # From model only
    assert result["reasoning_output"] is True
    # From base
    assert result["tool_call"] is True


def test_missing_provider(temp_data_dir: Path) -> None:
    """Test returns None for missing provider."""
    loader = _DataLoader()
    loader._data_dir = temp_data_dir
    result = loader.get_profile_data("nonexistent-provider", "test-model")

    assert result is None


def test_missing_model(temp_data_dir: Path) -> None:
    """Test returns None for missing model."""
    loader = _DataLoader()
    loader._data_dir = temp_data_dir
    result = loader.get_profile_data("test-provider", "nonexistent-model")

    assert result is None


def test_merged_data_is_cached(temp_data_dir: Path) -> None:
    """Test that merged data is cached after first access."""
    loader = _DataLoader()
    loader._data_dir = temp_data_dir
    # First access
    result1 = loader.get_profile_data("test-provider", "test-model")
    # Second access should use cached data
    result2 = loader.get_profile_data("test-provider", "test-model")

    assert result1 == result2
    # Verify it's using the cached property by checking _merged_data was accessed
    assert hasattr(loader, "_merged_data")
