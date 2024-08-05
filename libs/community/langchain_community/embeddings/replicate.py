import json
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field

DEFAULT_MODEL_NAME = "replicate/all-mpnet-base-v2:b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305"


class ReplicateEmbeddings(BaseModel, Embeddings):
    """Replicate embedding models.

    To use, you should have the ``replicate-python`` python package installed.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import ReplicateEmbeddings

            model_name = "replicate/all-mpnet-base-v2:b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305"
            re = ReplicateEmbeddings(
                model_name=model_name
            )
    """

    client: Any  #: :meta private:
    model_name: str = DEFAULT_MODEL_NAME
    """Model name to use."""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
        try:
            import replicate  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Could not import replicate python package. "
                "Please install it with `pip install replicate-python`."
            ) from exc

        self.client = replicate

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using replicate client and model transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = [text.replace("\n", " ") for text in texts]
        response = self.client.run(self.model_name, input={"text_batch": json.dumps(texts)})

        embeddings = [item['embedding'] for item in response]

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a  replicate client and model transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]
