from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import (
        BaseRequestsTool,
        RequestsDeleteTool,
        RequestsGetTool,
        RequestsPatchTool,
        RequestsPostTool,
        RequestsPutTool,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "BaseRequestsTool": "langchain_community.tools",
    "RequestsGetTool": "langchain_community.tools",
    "RequestsPostTool": "langchain_community.tools",
    "RequestsPatchTool": "langchain_community.tools",
    "RequestsPutTool": "langchain_community.tools",
    "RequestsDeleteTool": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "BaseRequestsTool",
    "RequestsGetTool",
    "RequestsPostTool",
    "RequestsPatchTool",
    "RequestsPutTool",
    "RequestsDeleteTool",
]
