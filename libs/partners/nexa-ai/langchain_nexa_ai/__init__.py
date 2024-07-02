from importlib import metadata

from langchain_nexa_ai.llms import NexaAILLM

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "NexaAILLM",
    "__version__",
]
