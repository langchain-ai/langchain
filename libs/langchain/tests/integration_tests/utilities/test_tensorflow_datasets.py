"""Integration tests for the TensorFlow Dataset client."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from langchain.pydantic_v1 import ValidationError
from langchain.schema.document import Document
from langchain.utilities.tensorflow_datasets import TensorflowDatasets

if TYPE_CHECKING:
    import tensorflow as tf  # noqa: E402


def decode_to_str(item: tf.Tensor) -> str:
    return item.numpy().decode("utf-8")


def mlqaen_example_to_document(example: dict) -> Document:
    return Document(
        page_content=decode_to_str(example["context"]),
        metadata={
            "id": decode_to_str(example["id"]),
            "title": decode_to_str(example["title"]),
            "question": decode_to_str(example["question"]),
            "answer": decode_to_str(example["answers"]["text"][0]),
        },
    )


MAX_DOCS = 10


@pytest.fixture
def tfds_client() -> TensorflowDatasets:
    return TensorflowDatasets(
        dataset_name="mlqa/en",
        split_name="test",
        load_max_docs=MAX_DOCS,
        sample_to_document_function=mlqaen_example_to_document,
    )


def test_load_success(tfds_client: TensorflowDatasets) -> None:
    """Test that returns the correct answer"""

    output = tfds_client.load()
    assert isinstance(output, list)
    assert len(output) == MAX_DOCS

    assert isinstance(output[0], Document)
    assert len(output[0].page_content) > 0
    assert isinstance(output[0].page_content, str)
    assert isinstance(output[0].metadata, dict)


def test_load_fail_wrong_dataset_name() -> None:
    """Test that fails to load"""
    with pytest.raises(ValidationError) as exc_info:
        TensorflowDatasets(
            dataset_name="wrong_dataset_name",
            split_name="test",
            load_max_docs=MAX_DOCS,
            sample_to_document_function=mlqaen_example_to_document,
        )
    assert "the dataset name is spelled correctly" in str(exc_info.value)


def test_load_fail_wrong_split_name() -> None:
    """Test that fails to load"""
    with pytest.raises(ValidationError) as exc_info:
        TensorflowDatasets(
            dataset_name="mlqa/en",
            split_name="wrong_split_name",
            load_max_docs=MAX_DOCS,
            sample_to_document_function=mlqaen_example_to_document,
        )
    assert "Unknown split" in str(exc_info.value)


def test_load_fail_no_func() -> None:
    """Test that fails to load"""
    with pytest.raises(ValidationError) as exc_info:
        TensorflowDatasets(
            dataset_name="mlqa/en",
            split_name="test",
            load_max_docs=MAX_DOCS,
        )
    assert "Please provide a function" in str(exc_info.value)
