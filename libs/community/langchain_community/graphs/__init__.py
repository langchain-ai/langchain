"""**Graphs** provide a natural language interface to graph databases."""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_community.graphs.arangodb_graph import (
        ArangoGraph,
    )
    from langchain_community.graphs.falkordb_graph import (
        FalkorDBGraph,
    )
    from langchain_community.graphs.gremlin_graph import (
        GremlinGraph,
    )
    from langchain_community.graphs.hugegraph import (
        HugeGraph,
    )
    from langchain_community.graphs.kuzu_graph import (
        KuzuGraph,
    )
    from langchain_community.graphs.memgraph_graph import (
        MemgraphGraph,
    )
    from langchain_community.graphs.nebula_graph import (
        NebulaGraph,
    )
    from langchain_community.graphs.neo4j_graph import (
        Neo4jGraph,
    )
    from langchain_community.graphs.neptune_graph import (
        NeptuneGraph,
    )
    from langchain_community.graphs.neptune_rdf_graph import (
        NeptuneRdfGraph,
    )
    from langchain_community.graphs.networkx_graph import (
        NetworkxEntityGraph,
    )
    from langchain_community.graphs.ontotext_graphdb_graph import (
        OntotextGraphDBGraph,
    )
    from langchain_community.graphs.rdf_graph import (
        RdfGraph,
    )
    from langchain_community.graphs.tigergraph_graph import (
        TigerGraph,
    )

__all__ = [
    "ArangoGraph",
    "FalkorDBGraph",
    "GremlinGraph",
    "HugeGraph",
    "KuzuGraph",
    "MemgraphGraph",
    "NebulaGraph",
    "Neo4jGraph",
    "NeptuneGraph",
    "NeptuneRdfGraph",
    "NetworkxEntityGraph",
    "OntotextGraphDBGraph",
    "RdfGraph",
    "TigerGraph",
]

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
