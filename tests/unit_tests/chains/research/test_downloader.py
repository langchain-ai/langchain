"""Tests for the downloader."""
import pytest

from langchain.chains.research.download import (
    _is_javascript_required,
)


@pytest.mark.requires("lxml")
def test_is_javascript_required() -> None:
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
    <script>
    console.log("Javascript is required.");
    </script>
    <body>
    </body>
    </html>
    """
    )
