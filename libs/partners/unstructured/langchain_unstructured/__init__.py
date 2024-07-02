from importlib import metadata

from langchain_unstructured.chat_models import ChatUnstructured
from langchain_unstructured.embeddings import UnstructuredEmbeddings
from langchain_unstructured.llms import UnstructuredLLM
from langchain_unstructured.vectorstores import UnstructuredVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatUnstructured",
    "UnstructuredLLM",
    "UnstructuredVectorStore",
    "UnstructuredEmbeddings",
    "__version__",
]
