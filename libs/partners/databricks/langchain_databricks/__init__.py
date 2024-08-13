from importlib import metadata

from langchain_databricks.chat_models import ChatDatabricks
from langchain_databricks.embeddings import DatabricksEmbeddings
from langchain_databricks.toolkits import UCFunctionToolkit
from langchain_databricks.vectorstores import DatabricksVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatDatabricks",
    "DatabricksVectorStore",
    "DatabricksEmbeddings",
    "UCFunctionToolkit",
    "__version__",
]
