"""**Graphs** provide a natural language interface to graph databases."""

from langchain_community.graphs.arangodb_graph import ArangoGraph
from langchain_community.graphs.falkordb_graph import FalkorDBGraph
from langchain_community.graphs.hugegraph import HugeGraph
from langchain_community.graphs.kuzu_graph import KuzuGraph
from langchain_community.graphs.memgraph_graph import MemgraphGraph
from langchain_community.graphs.nebula_graph import NebulaGraph
from langchain_community.graphs.neo4j_graph import Neo4jGraph
from langchain_community.graphs.neptune_graph import NeptuneGraph
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain_community.graphs.rdf_graph import RdfGraph

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
