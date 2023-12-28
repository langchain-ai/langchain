from typing import Any, List, Sequence

from langchain_core.callbacks.manager import (
    CallbackManagerForEmbeddingRun,
)
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra

DEFAULT_MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"


class TensorflowHubEmbeddings(BaseModel, Embeddings):
    """TensorflowHub embedding models.

    To use, you should have the ``tensorflow_text`` python package installed.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import TensorflowHubEmbeddings
            url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
            tf = TensorflowHubEmbeddings(model_url=url)
    """

    embed: Any  #: :meta private:
    model_url: str = DEFAULT_MODEL_URL
    """Model name to use."""

    def __init__(self, **kwargs: Any):
        """Initialize the tensorflow_hub and tensorflow_text."""
        super().__init__(**kwargs)
        try:
            import tensorflow_hub
        except ImportError:
            raise ImportError(
                "Could not import tensorflow-hub python package. "
                "Please install it with `pip install tensorflow-hub``."
            )
        try:
            import tensorflow_text  # noqa
        except ImportError:
            raise ImportError(
                "Could not import tensorflow_text python package. "
                "Please install it with `pip install tensorflow_text``."
            )

        self.embed = tensorflow_hub.load(self.model_url)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def _embed_documents(
        self,
        texts: List[str],
        *,
        run_managers: Sequence[CallbackManagerForEmbeddingRun],
    ) -> List[List[float]]:
        """Compute doc embeddings using a TensorflowHub embedding model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.embed(texts).numpy()
        return embeddings.tolist()

    def _embed_query(
        self,
        text: str,
        *,
        run_manager: CallbackManagerForEmbeddingRun,
    ) -> List[float]:
        """Compute query embeddings using a TensorflowHub embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        embedding = self.embed([text]).numpy()[0]
        return embedding.tolist()
