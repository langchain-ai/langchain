from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.graphs.graph_document import (
        GraphDocument,
        Node,
        Relationship,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "Node": "langchain_community.graphs.graph_document",
    "Relationship": "langchain_community.graphs.graph_document",
    "GraphDocument": "langchain_community.graphs.graph_document",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "GraphDocument",
    "Node",
    "Relationship",
]
