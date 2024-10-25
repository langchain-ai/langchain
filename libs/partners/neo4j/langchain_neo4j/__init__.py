from importlib import metadata

from langchain_neo4j.embeddings import Neo4jEmbeddings
from langchain_neo4j.vectorstores import Neo4jVectorStore
from langchain_neo4j.graphs.neo4j_graph import Neo4jGraph

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "Neo4jVectorStore",
    "Neo4jEmbeddings",
    "Neo4jGraph",
    "__version__",
]
