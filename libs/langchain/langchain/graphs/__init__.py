"""**Graphs** provide a natural language interface to graph databases."""

from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.graphs import (
        ArangoGraph,
        FalkorDBGraph,
        HugeGraph,
        KuzuGraph,
        MemgraphGraph,
        NebulaGraph,
        Neo4jGraph,
        NeptuneGraph,
        NetworkxEntityGraph,
        RdfGraph,
    )


# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "MemgraphGraph": "langchain_community.graphs",
    "NetworkxEntityGraph": "langchain_community.graphs",
    "Neo4jGraph": "langchain_community.graphs",
    "NebulaGraph": "langchain_community.graphs",
    "NeptuneGraph": "langchain_community.graphs",
    "KuzuGraph": "langchain_community.graphs",
    "HugeGraph": "langchain_community.graphs",
    "RdfGraph": "langchain_community.graphs",
    "ArangoGraph": "langchain_community.graphs",
    "FalkorDBGraph": "langchain_community.graphs",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "MemgraphGraph",
    "NetworkxEntityGraph",
    "Neo4jGraph",
    "NebulaGraph",
    "NeptuneGraph",
    "KuzuGraph",
    "HugeGraph",
    "RdfGraph",
    "ArangoGraph",
    "FalkorDBGraph",
]
