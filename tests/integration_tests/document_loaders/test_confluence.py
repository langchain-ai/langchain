import pytest

from langchain.document_loaders.confluence import ConfluenceLoader

try:
    from atlassian import Confluence  # noqa: F401

    confluence_installed = True
except ImportError:
    confluence_installed = False


@pytest.mark.skipif(not confluence_installed, reason="Atlassian package not installed")
def test_load_single_confluence_page() -> None:
    loader = ConfluenceLoader(url="https://templates.atlassian.net/wiki/")
    docs = loader.load(page_ids=["33189"])

    assert len(docs) == 1
    assert docs[0].page_content is not None
    assert docs[0].metadata["id"] == "33189"
    assert docs[0].metadata["title"] == "An easy intro to using Confluence"


@pytest.mark.skipif(not confluence_installed, reason="Atlassian package not installed")
def test_load_full_confluence_space() -> None:
    loader = ConfluenceLoader(url="https://templates.atlassian.net/wiki/")
    docs = loader.load(space_key="RD")

    assert len(docs) == 14
    assert docs[0].page_content is not None


@pytest.mark.skipif(not confluence_installed, reason="Atlassian package not installed")
def test_confluence_pagination() -> None:
    loader = ConfluenceLoader(url="https://templates.atlassian.net/wiki/")
    docs = loader.load(space_key="RD", limit=5)

    assert len(docs) == 5
    assert docs[0].page_content is not None
