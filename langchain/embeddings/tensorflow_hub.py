"""Wrapper around TensorflowHub embedding models."""
from typing import Any, List

from pydantic import BaseModel, Extra

from langchain.embeddings.base import Embeddings

DEFAULT_MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"


class TensorflowHubEmbeddings(BaseModel, Embeddings):
    embed: Any  #: :meta private:
    model_url: str = DEFAULT_MODEL_URL
    """Model name to use."""

    def __init__(self, **kwargs: Any):
        """Initialize the tensorflow_hub and tensorflow_text."""
        super().__init__(**kwargs)
        try:
            import tensorflow_hub
            import tensorflow_text

            self.embed = tensorflow_hub.load(self.model_url)
        except ImportError:
            raise ValueError(
                "Could not import tensorflow_hub and/or tensorflow_text python package. "
                "Please install it with `pip install tensorflow_hub` and `pip install tensorflow_text`."
            )

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a TensorflowHub embedding model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.embed(texts).numpy()
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a TensorflowHub embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        embedding = self.embed(text).numpy()[0]
        return embedding.tolist()
