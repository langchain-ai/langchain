"""This is the langchain_ollama package.

Provides infrastructure for interacting with the `Ollama <https://ollama.com/>`__
service.

.. note::
    **Newly added in 0.3.4:** ``validate_model_on_init`` param on all models.
    This parameter allows you to validate the model exists in Ollama locally on
    initialization. If set to ``True``, it will raise an error if the model does not
    exist locally. This is useful for ensuring that the model is available before
    attempting to use it, especially in environments where models may not be
    pre-downloaded.
"""

from importlib import metadata

from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatOllama",
    "OllamaEmbeddings",
    "OllamaLLM",
    "__version__",
]
