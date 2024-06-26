from importlib import metadata

from langchain_naver.chat_models import ChatNaver
from langchain_naver.embeddings import NaverEmbeddings
from langchain_naver.llms import NaverLLM
from langchain_naver.vectorstores import NaverVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatNaver",
    "NaverEmbeddings",
    "__version__",
]
