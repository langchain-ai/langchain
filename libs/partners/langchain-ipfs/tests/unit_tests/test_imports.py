"""Unit tests for langchain-ipfs package imports."""

from langchain_ipfs import IPFSPinTool, __version__


def test_version():
    """The version string is set."""
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_import_ipfs_pin_tool():
    """IPFSPinTool can be imported from the package."""
    assert IPFSPinTool is not None
    assert hasattr(IPFSPinTool, "name")
    assert hasattr(IPFSPinTool, "description")
    assert hasattr(IPFSPinTool, "_run")
