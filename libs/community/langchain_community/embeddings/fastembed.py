from typing import Any, Dict, List, Literal, Optional

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra
from langchain_core.utils import pre_init


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

    cache_dir: Optional[str]
    """The path to the cache directory.
    Defaults to `local_cache` in the parent directory
    """

    threads: Optional[int]
    """The number of threads single onnxruntime session can use.
    Defaults to None
    """

    doc_embed_type: Literal["default", "passage"] = "default"
    """Type of embedding to use for documents
    The available options are: "default" and "passage"
    """

    _model: Any  # : :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that FastEmbed has been installed."""
        model_name = values.get("model_name")
        max_length = values.get("max_length")
        cache_dir = values.get("cache_dir")
        threads = values.get("threads")

        try:
            # >= v0.2.0
            from fastembed import TextEmbedding

            values["_model"] = TextEmbedding(
                model_name=model_name,
                max_length=max_length,
                cache_dir=cache_dir,
                threads=threads,
            )
        except ImportError as ie:
            try:
                # < v0.2.0
                from fastembed.embedding import FlagEmbedding

                values["_model"] = FlagEmbedding(
                    model_name=model_name,
                    max_length=max_length,
                    cache_dir=cache_dir,
                    threads=threads,
                )
            except ImportError:
                raise ImportError(
                    "Could not import 'fastembed' Python package. "
                    "Please install it with `pip install fastembed`."
                ) from ie
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
            embeddings = self._model.passage_embed(texts)
        else:
            embeddings = self._model.embed(texts)
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Generate query embeddings using FastEmbed.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        query_embeddings: np.ndarray = next(self._model.query_embed(text))
        return query_embeddings.tolist()
