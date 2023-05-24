from pathlib import Path

import pytest
from pytest import raises
from pytest_mock import MockerFixture

from langchain.document_loaders.readthedocs import ReadTheDocsLoader


@pytest.mark.requires("bs4")
def test_main_id_main_content() -> None:
    loader = ReadTheDocsLoader(
        Path(__file__).parent / "test_docs" / "readthedocs" / "main_id_main_content"
    )
    documents = loader.load()
    assert len(documents[0].page_content) != 0


@pytest.mark.requires("bs4")
def test_div_role_main() -> None:
    loader = ReadTheDocsLoader(
        Path(__file__).parent / "test_docs" / "readthedocs" / "div_role_main"
    )
    documents = loader.load()
    assert len(documents[0].page_content) != 0


@pytest.mark.requires("bs4")
def test_custom() -> None:
    loader = ReadTheDocsLoader(
        Path(__file__).parent / "test_docs" / "readthedocs" / "custom",
        custom_html_tag=("article", {"role": "main"}),
    )
    documents = loader.load()
    assert len(documents[0].page_content) != 0


@pytest.mark.requires("bs4")
def test_empty() -> None:
    loader = ReadTheDocsLoader(
        Path(__file__).parent / "test_docs" / "readthedocs" / "custom",
    )
    documents = loader.load()
    assert len(documents[0].page_content) == 0
