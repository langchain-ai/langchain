from pathlib import Path

import pytest

from langchain_community.document_loaders import HuggingFaceDatasetLoader

HUGGING_FACE_EXAMPLE_DATASET = str(
    Path(__file__).parent / "sample_documents" / "sample_hugging_face_dataset.py"
)


@pytest.mark.requires("datasets")
@pytest.fixture
def test_load_string() -> None:
    """Loads page_content of type string"""
    page_content_column = "text"
    name = "v1"

    loader = HuggingFaceDatasetLoader(
        HUGGING_FACE_EXAMPLE_DATASET, page_content_column, name
    )
    docs = loader.load()

    # Length should be number of splits for specified `name`
    assert len(docs) == 2
    doc = docs[0]
    assert doc.page_content == '"This is text in version 1"'
    assert doc.metadata.keys() == {
        "split",
        "list",
        "dict",
    }


@pytest.mark.requires("datasets")
@pytest.fixture
def test_load_list() -> None:
    """Loads page_content of type List"""
    page_content_column = "list"
    name = "v1"

    loader = HuggingFaceDatasetLoader(
        HUGGING_FACE_EXAMPLE_DATASET, page_content_column, name
    )
    doc = loader.load()[0]
    assert doc.page_content == '["List item 1", "List item 2", "List item 3"]'
    assert doc.metadata.keys() == {
        "split",
        "text",
        "dict",
    }


@pytest.mark.requires("datasets")
@pytest.fixture
def test_load_object() -> None:
    """Loads page_content of type Object"""
    page_content_column = "dict"
    name = "v2"

    loader = HuggingFaceDatasetLoader(
        HUGGING_FACE_EXAMPLE_DATASET, page_content_column, name
    )
    doc = loader.load()[0]
    assert (
        doc.page_content
        == '{"dict_text": ["Hello world!", "langchain is cool"], "dict_int": [2, 123]}'
    )
    assert doc.metadata.keys() == {
        "split",
        "text",
        "list",
    }


@pytest.mark.requires("datasets")
@pytest.fixture
def test_load_nonexistent_dataset() -> None:
    """Tests that ValueError is thrown for nonexistent dataset name"""
    page_content_column = "text"
    name = "v3"

    loader = HuggingFaceDatasetLoader(
        HUGGING_FACE_EXAMPLE_DATASET, page_content_column, name
    )
    with pytest.raises(ValueError):
        loader.load()


@pytest.mark.requires("datasets")
@pytest.fixture
def test_load_nonexistent_feature() -> None:
    """Tests that KeyError is thrown for nonexistent feature/key in dataset"""
    page_content_column = "langchain"
    name = "v2"

    loader = HuggingFaceDatasetLoader(
        HUGGING_FACE_EXAMPLE_DATASET, page_content_column, name
    )
    with pytest.raises(KeyError):
        loader.load()
