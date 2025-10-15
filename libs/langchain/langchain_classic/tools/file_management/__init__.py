"""File Management Tools."""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import (
        CopyFileTool,
        DeleteFileTool,
        FileSearchTool,
        ListDirectoryTool,
        MoveFileTool,
        ReadFileTool,
        WriteFileTool,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "CopyFileTool": "langchain_community.tools",
    "DeleteFileTool": "langchain_community.tools",
    "FileSearchTool": "langchain_community.tools",
    "MoveFileTool": "langchain_community.tools",
    "ReadFileTool": "langchain_community.tools",
    "WriteFileTool": "langchain_community.tools",
    "ListDirectoryTool": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "CopyFileTool",
    "DeleteFileTool",
    "FileSearchTool",
    "ListDirectoryTool",
    "MoveFileTool",
    "ReadFileTool",
    "WriteFileTool",
]
