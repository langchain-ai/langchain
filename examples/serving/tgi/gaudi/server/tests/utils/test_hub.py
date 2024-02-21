import pytest

from text_generation_server.utils.hub import (
    weight_hub_files,
    download_weights,
    weight_files,
    EntryNotFoundError,
    LocalEntryNotFoundError,
    RevisionNotFoundError,
)


def test_weight_hub_files():
    filenames = weight_hub_files("bigscience/bloom-560m")
    assert filenames == ["model.safetensors"]


def test_weight_hub_files_llm():
    filenames = weight_hub_files("bigscience/bloom")
    assert filenames == [f"model_{i:05d}-of-00072.safetensors" for i in range(1, 73)]


def test_weight_hub_files_empty():
    with pytest.raises(EntryNotFoundError):
        weight_hub_files("bigscience/bloom", extension=".errors")


def test_download_weights():
    model_id = "bigscience/bloom-560m"
    filenames = weight_hub_files(model_id)
    files = download_weights(filenames, model_id)
    local_files = weight_files("bigscience/bloom-560m")
    assert files == local_files


def test_weight_files_error():
    with pytest.raises(RevisionNotFoundError):
        weight_files("bigscience/bloom-560m", revision="error")
    with pytest.raises(LocalEntryNotFoundError):
        weight_files("bert-base-uncased")
