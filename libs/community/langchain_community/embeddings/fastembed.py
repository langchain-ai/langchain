import importlib
import importlib.metadata
from typing import Any, Dict, List, Literal, Optional, cast

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.utils import pre_init
from pydantic import BaseModel, ConfigDict

MIN_VERSION = "0.2.0"


class FastEmbedEmbeddings(BaseModel, Embeddings):
    """Qdrant FastEmbedding models.

    FastEmbed is a lightweight, fast, Python library built for embedding generation.
    See more documentation at:
    * https://github.com/qdrant/fastembed/
    * https://qdrant.github.io/fastembed/

    To use this class, you must install the `fastembed` Python package.

    `pip install fastembed`
    Example:
        from langchain_community.embeddings import FastEmbedEmbeddings
        fastembed = FastEmbedEmbeddings()
    """

    model_name: str = "BAAI/bge-small-en-v1.5"
    """Name of the FastEmbedding model to use
    Defaults to "BAAI/bge-small-en-v1.5"
    Find the list of supported models at
    https://qdrant.github.io/fastembed/examples/Supported_Models/
    """

    max_length: int = 512
    """The maximum number of tokens. Defaults to 512.
    Unknown behavior for values > 512.
    """

    cache_dir: Optional[str] = None
    """The path to the cache directory.
    Defaults to `local_cache` in the parent directory
    """

    threads: Optional[int] = None
    """The number of threads single onnxruntime session can use.
    Defaults to None
    """

    doc_embed_type: Literal["default", "passage"] = "default"
    """Type of embedding to use for documents
    The available options are: "default" and "passage"
    """

    batch_size: int = 256
    """Batch size for encoding. Higher values will use more memory, but be faster.
    Defaults to 256.
    """

    parallel: Optional[int] = None
    """If `>1`, parallel encoding is used, recommended for encoding of large datasets.
    If `0`, use all available cores.
    If `None`, don't use data-parallel processing, use default onnxruntime threading.
    Defaults to `None`.
    """

    model: Any = None  # : :meta private:

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that FastEmbed has been installed."""
        model_name = values.get("model_name")
        max_length = values.get("max_length")
        cache_dir = values.get("cache_dir")
        threads = values.get("threads")

        try:
            fastembed = importlib.import_module("fastembed")

        except ModuleNotFoundError:
            raise ImportError(
                "Could not import 'fastembed' Python package. "
                "Please install it with `pip install fastembed`."
            )

        if importlib.metadata.version("fastembed") < MIN_VERSION:
            raise ImportError(
                'FastEmbedEmbeddings requires `pip install -U "fastembed>=0.2.0"`.'
            )

        values["model"] = fastembed.TextEmbedding(
            model_name=model_name,
            max_length=max_length,
            cache_dir=cache_dir,
            threads=threads,
        )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents using FastEmbed.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings: List[np.ndarray]
        if self.doc_embed_type == "passage":
            embeddings = self.model.passage_embed(
                texts, batch_size=self.batch_size, parallel=self.parallel
            )
        else:
            embeddings = self.model.embed(
                texts, batch_size=self.batch_size, parallel=self.parallel
            )
        return [cast(List[float], e.tolist()) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Generate query embeddings using FastEmbed.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        query_embeddings: np.ndarray = next(
            self.model.query_embed(
                text, batch_size=self.batch_size, parallel=self.parallel
            )
        )
        return cast(List[float], query_embeddings.tolist())
