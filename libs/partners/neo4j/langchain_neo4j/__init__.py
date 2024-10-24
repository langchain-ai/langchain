from importlib import metadata

from langchain_neo4j.chat_models import ChatNeo4j
from langchain_neo4j.embeddings import Neo4jEmbeddings
from langchain_neo4j.llms import Neo4jLLM
from langchain_neo4j.vectorstores import Neo4jVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatNeo4j",
    "Neo4jLLM",
    "Neo4jVectorStore",
    "Neo4jEmbeddings",
    "__version__",
]
