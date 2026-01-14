"""**Graphs** provide a natural language interface to graph databases."""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.graphs.index_creator import GraphIndexCreator
    from langchain_community.graphs.networkx_graph import NetworkxEntityGraph


# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "GraphIndexCreator": "langchain_community.graphs.index_creator",
    "NetworkxEntityGraph": "langchain_community.graphs.networkx_graph",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = ["GraphIndexCreator", "NetworkxEntityGraph"]
