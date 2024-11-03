from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import O365SendEvent
    from langchain_community.tools.office365.send_event import SendEventSchema

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "SendEventSchema": "langchain_community.tools.office365.send_event",
    "O365SendEvent": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "SendEventSchema",
    "O365SendEvent",
]
