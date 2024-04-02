from importlib import metadata

from langchain_cloudflare.chat_models import ChatCloudflare
from langchain_cloudflare.embeddings import CloudflareEmbeddings
from langchain_cloudflare.llms import CloudflareLLM
from langchain_cloudflare.vectorstores import CloudflareVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatCloudflare",
    "CloudflareLLM",
    "CloudflareVectorStore",
    "CloudflareEmbeddings",
    "__version__",
]
