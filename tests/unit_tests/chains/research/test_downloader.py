"""Tests for the downloader."""
from langchain.chains.research.fetch import (
    AutoDownloadHandler,
    PlaywrightDownloadHandler,
    RequestsDownloadHandler,
    _is_javascript_required,
)


def test_is_javascript_required():
    """Check whether a given page should be re-downloaded with javascript executed."""
    assert not _is_javascript_required(
        """
    <html>
    <body>
    <p>Check whether javascript is required.</p>
    </body>
    </html>
    """
    )

    assert _is_javascript_required(
        """
    <html>
    <body>
    <script>
    console.log("Javascript is required.");
    </script>
    </body>
    </html>
    """
    )


def test_requests_handler():
    """Test that the requests handler is working."""
    handler = RequestsDownloadHandler()
    fetch = handler.download(["https://www.google.com"])
