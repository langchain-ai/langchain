"""MutliOn Client API tools."""

from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools.multion.close_session import MultionCloseSession
    from langchain_community.tools.multion.create_session import MultionCreateSession
    from langchain_community.tools.multion.update_session import MultionUpdateSession

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "MultionCreateSession": "langchain_community.tools.multion.create_session",
    "MultionUpdateSession": "langchain_community.tools.multion.update_session",
    "MultionCloseSession": "langchain_community.tools.multion.close_session",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "MultionCreateSession",
    "MultionUpdateSession",
    "MultionCloseSession",
]
