"""**Graphs** provide a natural language interface to graph databases."""
from typing import Any

from langchain._api import create_importer

importer = create_importer(__package__, fallback_module="langchain_community.graphs")


def __getattr__(name: str) -> Any:
    return importer(name)


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
