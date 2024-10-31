from importlib import metadata

from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_neo4j.chat_message_histories.neo4j import Neo4jChatMessageHistory
from langchain_neo4j.graphs.neo4j_graph import Neo4jGraph
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "GraphCypherQAChain",
    "Neo4jChatMessageHistory",
    "Neo4jGraph",
    "Neo4jVector",
    "__version__",
]
