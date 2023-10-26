from pathlib import Path

import pytest

from langchain.document_loaders.readthedocs import ReadTheDocsLoader

PARENT_DIR = Path(__file__).parent / "test_docs" / "readthedocs"


@pytest.mark.requires("bs4")
def test_main_id_main_content() -> None:
    loader = ReadTheDocsLoader(PARENT_DIR / "main_id_main_content")
    documents = loader.load()
    assert len(documents[0].page_content) != 0


@pytest.mark.requires("bs4")
def test_div_role_main() -> None:
    loader = ReadTheDocsLoader(PARENT_DIR / "div_role_main")
    documents = loader.load()
    assert len(documents[0].page_content) != 0


@pytest.mark.requires("bs4")
def test_custom() -> None:
    loader = ReadTheDocsLoader(
        PARENT_DIR / "custom",
        custom_html_tag=("article", {"role": "main"}),
    )
    documents = loader.load()
    assert len(documents[0].page_content) != 0


@pytest.mark.requires("bs4")
def test_nested_html_structure() -> None:
    loader = ReadTheDocsLoader(PARENT_DIR / "nested_html_structure")
    documents = loader.load()
    assert "\n" not in documents[0].page_content


@pytest.mark.requires("bs4")
def test_index_page() -> None:
    loader = ReadTheDocsLoader(
        PARENT_DIR / "index_page", exclude_index_pages=True
    )
    documents = loader.load()
    assert len(documents[0].page_content) == 0


@pytest.mark.requires("bs4")
def test_empty() -> None:
    loader = ReadTheDocsLoader(
        PARENT_DIR / "custom",
    )
    documents = loader.load()
    assert len(documents[0].page_content) == 0
