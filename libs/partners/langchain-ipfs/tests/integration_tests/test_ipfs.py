"""Integration tests for langchain-ipfs package."""

import pytest


def test_import():
    """The package can be imported (SDK dependency may not be installed)."""
    from langchain_ipfs import IPFSPinTool

    assert IPFSPinTool is not None


def test_tool_schema():
    """The tool has expected attributes."""
    from langchain_ipfs import IPFSPinTool

    tool = IPFSPinTool()
    assert tool.name == "ipfs_pin_tool"
    assert "IPFS" in tool.description.lower() or "ipfs" in tool.description.lower()
