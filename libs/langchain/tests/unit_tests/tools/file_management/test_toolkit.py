"""Test the FileManagementToolkit."""

from tempfile import TemporaryDirectory

import pytest

from langchain.agents.agent_toolkits.file_management.toolkit import (
    FileManagementToolkit,
)
from langchain.tools.base import BaseTool


def test_file_toolkit_get_tools() -> None:
    """Test the get_tools method of FileManagementToolkit."""
    with TemporaryDirectory() as temp_dir:
        toolkit = FileManagementToolkit(root_dir=temp_dir)
        tools = toolkit.get_tools()
        assert len(tools) > 0
        assert all(isinstance(tool, BaseTool) for tool in tools)


def test_file_toolkit_get_tools_with_selection() -> None:
    """Test the get_tools method of FileManagementToolkit with selected_tools."""
    with TemporaryDirectory() as temp_dir:
        toolkit = FileManagementToolkit(
            root_dir=temp_dir, selected_tools=["read_file", "write_file"]
        )
        tools = toolkit.get_tools()
        assert len(tools) == 2
        tool_names = [tool.name for tool in tools]
        assert "read_file" in tool_names
        assert "write_file" in tool_names


def test_file_toolkit_invalid_tool() -> None:
    """Test the FileManagementToolkit with an invalid tool."""
    with TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError):
            FileManagementToolkit(root_dir=temp_dir, selected_tools=["invalid_tool"])


def test_file_toolkit_root_dir() -> None:
    """Test the FileManagementToolkit root_dir handling."""
    with TemporaryDirectory() as temp_dir:
        toolkit = FileManagementToolkit(root_dir=temp_dir)
        tools = toolkit.get_tools()
        root_dirs = [tool.root_dir for tool in tools if hasattr(tool, "root_dir")]
        assert all(root_dir == temp_dir for root_dir in root_dirs)
