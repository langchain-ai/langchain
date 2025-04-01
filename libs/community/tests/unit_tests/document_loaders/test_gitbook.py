from typing import Any, Tuple
from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup

from langchain_community.document_loaders.gitbook import GitbookLoader


@pytest.fixture
def mock_soups() -> Tuple[BeautifulSoup, BeautifulSoup]:
    # Create mock soup with loc elements for sitemap testing
    sitemap_content = """
    <urlset>
        <url><loc>https://example.com/page1</loc></url>
        <url><loc>https://example.com/page2</loc></url>
        <url><loc>https://example.com/page3</loc></url>
    </urlset>
    """
    mock_sitemap_soup = BeautifulSoup(sitemap_content, "html.parser")

    # Create mock soup for page content
    page_content = """
    <html>
        <body>
            <main>
                <h1>Test Page</h1>
                <p>This is test content.</p>
            </main>
        </body>
    </html>
    """
    mock_page_soup = BeautifulSoup(page_content, "html.parser")
    return mock_sitemap_soup, mock_page_soup


@patch("langchain_community.document_loaders.web_base.requests.get")
def test_init_with_default_sitemap(mock_get: MagicMock) -> None:
    # Test that the loader uses the default sitemap URL when load_all_paths=True
    loader = GitbookLoader(web_page="https://example.com", load_all_paths=True)

    # Check that the web_path was set to the default sitemap URL
    assert loader.web_paths[0] == "https://example.com/sitemap.xml"


@patch("langchain_community.document_loaders.web_base.requests.get")
def test_init_with_custom_sitemap(mock_get: MagicMock) -> None:
    # Test that the loader uses the provided sitemap URL when specified
    custom_sitemap = "https://example.com/sitemap-pages.xml"
    loader = GitbookLoader(
        web_page="https://example.com",
        load_all_paths=True,
        sitemap_url=custom_sitemap,
    )

    # Check that the web_path was set to the custom sitemap URL
    assert loader.web_paths[0] == custom_sitemap


@patch("langchain_community.document_loaders.gitbook.GitbookLoader.scrape")
@patch("langchain_community.document_loaders.gitbook.GitbookLoader.scrape_all")
def test_lazy_load_with_custom_sitemap(
    mock_scrape_all: MagicMock,
    mock_scrape: MagicMock,
    mock_soups: Tuple[BeautifulSoup, BeautifulSoup],
) -> None:
    # Setup the mocks
    mock_sitemap_soup, mock_page_soup = mock_soups
    mock_scrape.return_value = mock_sitemap_soup
    mock_scrape_all.return_value = [
        mock_page_soup,
        mock_page_soup,
        mock_page_soup,
    ]

    # Create loader with custom sitemap URL
    loader = GitbookLoader(
        web_page="https://example.com",
        load_all_paths=True,
        sitemap_url="https://example.com/sitemap-pages.xml",
    )

    # Get the documents
    docs = list(loader.lazy_load())

    # Check that we got docs for each path in the sitemap
    assert len(docs) == 3
    for doc in docs:
        assert doc.metadata["title"] == "Test Page"
        assert "This is test content." in doc.page_content


@patch("langchain_community.document_loaders.web_base.requests.get")
def test_with_single_page(mock_get: MagicMock) -> None:
    # Test loading a single page (load_all_paths=False)
    loader = GitbookLoader(web_page="https://example.com/page", load_all_paths=False)

    # Check that sitemap URL logic was not applied
    assert loader.web_paths[0] == "https://example.com/page"


@patch("langchain_community.document_loaders.gitbook.GitbookLoader.scrape")
def test_get_paths_extraction(
    mock_scrape: MagicMock, mock_soups: Tuple[BeautifulSoup, BeautifulSoup]
) -> None:
    # Test that _get_paths correctly extracts paths from sitemap
    mock_sitemap_soup, _ = mock_soups
    mock_scrape.return_value = mock_sitemap_soup

    loader = GitbookLoader(web_page="https://example.com", load_all_paths=True)

    soup_info = loader.scrape()
    paths = loader._get_paths(soup_info)

    # Check that paths were extracted correctly
    assert len(paths) == 3
    assert paths == ["/page1", "/page2", "/page3"]


@patch("requests.get")
def test_integration_with_different_sitemaps(mock_get: MagicMock) -> None:
    # This test simulates the reported issue with different sitemap formats

    # Mock response for default sitemap (empty content)
    empty_resp = MagicMock()
    empty_resp.text = "<urlset></urlset>"
    empty_resp.status_code = 200

    # Mock response for custom sitemap (with content)
    custom_resp = MagicMock()
    custom_resp.text = """
    <urlset>
        <url><loc>https://docs.gitbook.com/page1</loc></url>
        <url><loc>https://docs.gitbook.com/page2</loc></url>
    </urlset>
    """
    custom_resp.status_code = 200

    # Mock response for the actual pages
    page_resp = MagicMock()
    page_resp.text = """
    <html><body><main><h1>Page</h1><p>Content</p></main></body></html>
    """
    page_resp.status_code = 200

    # Define side effect to return different responses based on URL
    def side_effect(url: str, *args: Any, **kwargs: Any) -> MagicMock:
        if url == "https://docs.gitbook.com/sitemap.xml":
            return empty_resp
        elif url == "https://docs.gitbook.com/sitemap-pages.xml":
            return custom_resp
        else:
            return page_resp

    mock_get.side_effect = side_effect

    # Test with default sitemap (should result in no docs)
    with patch(
        "langchain_community.document_loaders.web_base.requests.get",
        side_effect=side_effect,
    ):
        with patch(
            "langchain_community.document_loaders.gitbook.GitbookLoader.scrape"
        ) as mock_scrape:
            with patch(
                "langchain_community.document_loaders.gitbook.GitbookLoader.scrape_all"
            ) as mock_scrape_all:
                mock_scrape.return_value = BeautifulSoup(
                    "<urlset></urlset>", "html.parser"
                )
                mock_scrape_all.return_value = []

                loader1 = GitbookLoader(
                    web_page="https://docs.gitbook.com/", load_all_paths=True
                )
                docs1 = list(loader1.lazy_load())
                assert len(docs1) == 0

    # Test with custom sitemap (should result in docs)
    with patch(
        "langchain_community.document_loaders.web_base.requests.get",
        side_effect=side_effect,
    ):
        with patch(
            "langchain_community.document_loaders.gitbook.GitbookLoader.scrape"
        ) as mock_scrape:
            with patch(
                "langchain_community.document_loaders.gitbook.GitbookLoader.scrape_all"
            ) as mock_scrape_all:
                mock_scrape.return_value = BeautifulSoup(
                    custom_resp.text, "html.parser"
                )
                mock_scrape_all.return_value = [
                    BeautifulSoup(page_resp.text, "html.parser"),
                    BeautifulSoup(page_resp.text, "html.parser"),
                ]

                loader2 = GitbookLoader(
                    web_page="https://docs.gitbook.com/",
                    load_all_paths=True,
                    sitemap_url="https://docs.gitbook.com/sitemap-pages.xml",
                )
                docs2 = list(loader2.lazy_load())
                assert len(docs2) == 2
