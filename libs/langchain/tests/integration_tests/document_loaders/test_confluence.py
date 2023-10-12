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
    assert docs[0].metadata["source"] == (
        "https://templates.atlassian.net/wiki/"
        "spaces/RD/pages/33189/An+easy+intro+to+using+Confluence"
    )


@pytest.mark.skipif(not confluence_installed, reason="Atlassian package not installed")
def test_load_full_confluence_space() -> None:
    loader = ConfluenceLoader(url="https://templates.atlassian.net/wiki/")
    docs = loader.load(space_key="RD")

    assert len(docs) == 14
    assert docs[0].page_content is not None


@pytest.mark.skipif(not confluence_installed, reason="Atlassian package not installed")
def test_confluence_pagination() -> None:
    loader = ConfluenceLoader(url="https://templates.atlassian.net/wiki/")
    # this will issue 2 requests; each with a limit of 3 until the max_pages of 5 is met
    docs = loader.load(space_key="RD", limit=3, max_pages=5)

    assert len(docs) == 5
    assert docs[0].page_content is not None


@pytest.mark.skipif(not confluence_installed, reason="Atlassian package not installed")
def test_pass_confluence_kwargs() -> None:
    loader = ConfluenceLoader(
        url="https://templates.atlassian.net/wiki/",
        confluence_kwargs={"verify_ssl": False},
    )

    assert loader.confluence.verify_ssl is False
