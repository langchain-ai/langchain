import pytest

from langchain.document_loaders import UnstructuredURLLoader


def test_continue_on_failure_true() -> None:
    """Test exception is not raised when continue_on_failure=True."""
    loader = UnstructuredURLLoader(["badurl.foobar"])
    loader.load()


def test_continue_on_failure_false() -> None:
    """Test exception is raised when continue_on_failure=False."""
    loader = UnstructuredURLLoader(["badurl.foobar"], continue_on_failure=False)
    with pytest.raises(Exception):
        loader.load()
