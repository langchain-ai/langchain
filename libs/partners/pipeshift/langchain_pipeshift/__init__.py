"""This package is Pipeshift's LangChain integration."""

from importlib import metadata

from langchain_pipeshift.chat_models import ChatPipeshift

# from langchain_pipeshift.embeddings import PipeshiftEmbeddings
from langchain_pipeshift.llms import Pipeshift

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatPipeshift",
    "Pipeshift",
    # "PipeshiftEmbeddings",
    "__version__",
]
