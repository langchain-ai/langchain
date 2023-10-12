import pytest

from langchain.document_loaders import RSSFeedLoader


@pytest.mark.requires("feedparser", "newspaper")
def test_continue_on_failure_true() -> None:
    """Test exception is not raised when continue_on_failure=True."""
    loader = RSSFeedLoader(["badurl.foobar"])
    loader.load()


@pytest.mark.requires("feedparser", "newspaper")
def test_continue_on_failure_false() -> None:
    """Test exception is raised when continue_on_failure=False."""
    loader = RSSFeedLoader(["badurl.foobar"], continue_on_failure=False)
    with pytest.raises(Exception):
        loader.load()
