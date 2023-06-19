from pathlib import Path

import pytest

from langchain.document_loaders.rst import UnstructuredRSTLoader

PARENT_DIR = Path(__file__).parent / "test_docs" / "unstructured_rst_loader"


@pytest.mark.requires("docutils")
def test_load_valid_file() -> None:
    loader = UnstructuredRSTLoader((PARENT_DIR / "valid.rst").as_posix())
    documents = loader.load()
    assert len(documents) == 1
    assert len(documents[0].page_content) != 0


@pytest.mark.requires("docutils")
def test_empty_file() -> None:
    loader = UnstructuredRSTLoader((PARENT_DIR / "empty.rst").as_posix())
    documents = loader.load()
    assert len(documents) == 1
    assert len(documents[0].page_content) == 0
