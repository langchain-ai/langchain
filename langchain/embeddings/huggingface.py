"""Wrapper around OpenAI embedding models."""
from typing import Any, Dict, List

from pydantic import BaseModel, Extra, root_validator

from langchain.embeddings.base import Embeddings


class HuggingFaceEmbeddings(BaseModel, Embeddings):
    """Wrapper around sentence_transformers embedding models.

    To use, you should have the ``sentence_transformers`` python package installed.

    Example:
        .. code-block:: python

            from langchain.embeddings import HuggingFaceEmbeddings
            huggingface = HuggingFaceEmbeddings(model_name="")
    """

    client: Any  #: :meta private:
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    """Model name to use."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""
        try:
            import sentence_transformers

            values["client"] = sentence_transformers.SentenceTransformer
        except ImportError:
            raise ValueError(
                "Could not import sentence_transformers python package. "
                "Please it install it with `pip install sentence_transformers`."
            )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Computes doc embeddings using a HuggingFace transformer model

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.client(self.model_name).encode(texts)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Computes query embeddings using a HuggingFace transformer model

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        embedding = self.client(self.model_name).encode(text)
        return embedding
