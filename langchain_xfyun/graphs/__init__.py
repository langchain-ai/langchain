"""**Graphs** provide a natural language interface to graph databases."""

from langchain_xfyun.graphs.arangodb_graph import ArangoGraph
from langchain_xfyun.graphs.hugegraph import HugeGraph
from langchain_xfyun.graphs.kuzu_graph import KuzuGraph
from langchain_xfyun.graphs.memgraph_graph import MemgraphGraph
from langchain_xfyun.graphs.nebula_graph import NebulaGraph
from langchain_xfyun.graphs.neo4j_graph import Neo4jGraph
from langchain_xfyun.graphs.neptune_graph import NeptuneGraph
from langchain_xfyun.graphs.networkx_graph import NetworkxEntityGraph
from langchain_xfyun.graphs.rdf_graph import RdfGraph

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
]
