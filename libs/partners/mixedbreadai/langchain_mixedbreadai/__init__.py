from importlib import metadata

from langchain_mixedbreadai.embeddings import MixedbreadAIEmbeddings
from langchain_mixedbreadai.rerank import MixedbreadAIRerank

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "MixedbreadAIEmbeddings",
    "MixedbreadAIRerank",
    "__version__",
]
