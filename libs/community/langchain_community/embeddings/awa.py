from typing import Any, Dict, List

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, model_validator


class AwaEmbeddings(BaseModel, Embeddings):
    """Embedding documents and queries with Awa DB.

    Attributes:
        client: The AwaEmbedding client.
        model: The name of the model used for embedding.
         Default is "all-mpnet-base-v2".
    """

    client: Any  #: :meta private:
    model: str = "all-mpnet-base-v2"

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that awadb library is installed."""

        try:
            from awadb import AwaEmbedding
        except ImportError as exc:
            raise ImportError(
                "Could not import awadb library. "
                "Please install it with `pip install awadb`"
            ) from exc
        values["client"] = AwaEmbedding()
        return values

    def set_model(self, model_name: str) -> None:
        """Set the model used for embedding.
        The default model used is all-mpnet-base-v2

        Args:
            model_name: A string which represents the name of model.
        """
        self.model = model_name
        self.client.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using AwaEmbedding.

        Args:
            texts: The list of texts need to be embedded

        Returns:
            List of embeddings, one for each text.
        """
        return self.client.EmbeddingBatch(texts)

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using AwaEmbedding.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.client.Embedding(text)
