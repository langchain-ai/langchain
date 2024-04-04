"""**Graphs** provide a natural language interface to graph databases."""

import importlib
from typing import Any

_module_lookup = {
    "ArangoGraph": "langchain_community.graphs.arangodb_graph",
    "FalkorDBGraph": "langchain_community.graphs.falkordb_graph",
    "GremlinGraph": "langchain_community.graphs.gremlin_graph",
    "HugeGraph": "langchain_community.graphs.hugegraph",
    "KuzuGraph": "langchain_community.graphs.kuzu_graph",
    "MemgraphGraph": "langchain_community.graphs.memgraph_graph",
    "NebulaGraph": "langchain_community.graphs.nebula_graph",
    "Neo4jGraph": "langchain_community.graphs.neo4j_graph",
    "BaseNeptuneGraph": "langchain_community.graphs.neptune_graph",
    "NeptuneAnalyticsGraph": "langchain_community.graphs.neptune_graph",
    "NeptuneGraph": "langchain_community.graphs.neptune_graph",
    "NeptuneRdfGraph": "langchain_community.graphs.neptune_rdf_graph",
    "NetworkxEntityGraph": "langchain_community.graphs.networkx_graph",
    "OntotextGraphDBGraph": "langchain_community.graphs.ontotext_graphdb_graph",
    "RdfGraph": "langchain_community.graphs.rdf_graph",
    "TigerGraph": "langchain_community.graphs.tigergraph_graph",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_module_lookup.keys())
