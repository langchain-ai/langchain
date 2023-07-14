from typing import Any, Callable
from unittest.mock import MagicMock, Mock

import pytest
from pytest import MonkeyPatch

from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader


@pytest.fixture
def url_loader() -> RecursiveUrlLoader:
    url = "http://test.com"
    exclude_dir = "/exclude"  # Note: Changed from list to single string
    return RecursiveUrlLoader(url, exclude_dir)


@pytest.fixture
def mock_requests_get(monkeypatch: MonkeyPatch) -> None:
    """Mock requests.get"""

    # Mocking HTML content with 2 links, one absolute, one relative.
    html_content = """
    <html>
        <body>
            <a href="/relative">relative link</a>
            <a href="http://test.com/absolute">absolute link</a>
        </body>
    </html>
    """

    # Mock Response object for main URL
    mock_response_main = MagicMock()
    mock_response_main.text = html_content

    # Mock Response object for relative URL
    mock_response_relative = MagicMock()
    mock_response_relative.text = "Relative page"

    # Mock Response object for absolute URL
    mock_response_absolute = MagicMock()
    mock_response_absolute.text = "Absolute page"

    # Mock Response object for default
    mock_response_default = MagicMock()
    mock_response_default.text = "Default page"

    def mock_get(url: str, *args: Any, **kwargs: Any) -> Mock:
        if url.startswith("http://test.com"):
            if "/absolute" in url:
                return mock_response_absolute
            elif "/relative" in url:
                return mock_response_relative
            else:
                return mock_response_main
        return mock_response_default

    monkeypatch.setattr(
        "langchain.document_loaders.recursive_url_loader.requests.get", mock_get
    )


def test_get_child_links_recursive(
    url_loader: RecursiveUrlLoader, mock_requests_get: Callable[[], None]
) -> None:
    # Testing for both relative and absolute URL
    child_links = url_loader.get_child_links_recursive("http://test.com")

    assert child_links == {
        "http://test.com/relative",
        "http://test.com/absolute",
    }
