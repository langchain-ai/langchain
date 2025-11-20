"""Tests for data loader with augmentation support."""

import json
from pathlib import Path

import pytest

from langchain_core.language_models.profile._data_loader import _DataLoader


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
                },
                "test-model-2": {
                    "id": "test-model-2",
                    "name": "Test Model 2",
                    "tool_call": True,
                    "limit": {"context": 16000, "output": 8000},
                    "modalities": {"input": ["text"], "output": ["text"]},
                },
            },
        }
    }

    with (data_dir / "models.json").open("w") as f:
        json.dump(base_data, f)

    return data_dir


def test_load_base_data_only(temp_data_dir: Path) -> None:
    """Test loading base data without augmentations."""
    loader = _DataLoader(temp_data_dir)
    result = loader.get_profile_data("test-provider")

    assert result is not None
    assert len(result) == 2
    model = result["test-model"]
    assert model["id"] == "test-model"
    assert model["name"] == "Test Model"
    assert model["tool_call"] is True


def test_provider_level_augmentation(temp_data_dir: Path) -> None:
    """Test provider-level augmentations are applied."""
    # Add provider augmentation using new format
    aug_file = temp_data_dir / "profile_augmentations.toml"
    aug_file.write_text("""
provider = "test-provider"

[overrides]
image_url_inputs = true
pdf_inputs = true
""")

    loader = _DataLoader(temp_data_dir)
    result = loader.get_profile_data("test-provider")

    assert result is not None
    model = result["test-model"]
    assert model["image_url_inputs"] is True
    assert model["pdf_inputs"] is True
    # Base data should still be present
    assert model["tool_call"] is True


def test_model_level_augmentation_overrides_provider(temp_data_dir: Path) -> None:
    """Test model-level augmentations override provider augmentations."""
    # Add both provider and model augmentations using new format
    aug_file = temp_data_dir / "profile_augmentations.toml"
    aug_file.write_text("""
provider = "test-provider"

[overrides]
image_url_inputs = true
pdf_inputs = false

[overrides."test-model"]
pdf_inputs = true
reasoning_output = true
""")

    loader = _DataLoader(temp_data_dir)
    result = loader.get_profile_data("test-provider")

    assert result is not None
    model = result["test-model"]
    # From provider
    assert model["image_url_inputs"] is True
    # Overridden by model
    assert model["pdf_inputs"] is True
    # From model only
    assert model["reasoning_output"] is True
    # From base
    assert model["tool_call"] is True

    # Check that model-2 only has provider-level augmentations
    model2 = result["test-model-2"]
    assert model2["image_url_inputs"] is True
    assert model2["pdf_inputs"] is False
    assert "reasoning_output" not in model2


def test_missing_provider(temp_data_dir: Path) -> None:
    """Test returns None for missing provider."""
    loader = _DataLoader(temp_data_dir)
    result = loader.get_profile_data("nonexistent-provider")

    assert result is None


def test_merged_data_is_cached(temp_data_dir: Path) -> None:
    """Test that merged data is cached after first access."""
    loader = _DataLoader(temp_data_dir)
    # First access
    result1 = loader.get_profile_data("test-provider")
    # Second access should use cached data
    result2 = loader.get_profile_data("test-provider")

    assert result1 == result2
    # Verify it's using the cached property by checking _merged_data was accessed
    assert hasattr(loader, "_merged_data")
