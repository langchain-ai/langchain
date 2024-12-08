"""Wrapper around model2vec embedding models."""

from typing import Any, List, Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel


class Model2vecEmbeddings(Embeddings, BaseModel):
    """model2v embedding models.

    Install model2vec first, run 'pip install -U model2vec'.
    The github repository for model2vec is : https://github.com/MinishLab/model2vec

    Example:
        .. code-block:: python

            from langchain_community.embeddings.model2v import Model2vecEmbeddings

            embedding = Model2vecEmbeddings()
            embedding.embed_documents([
                "It's dangerous to go alone!",
                "It's a secret to everybody.",
            ])
            embedding.embed_query(
                "Take this with you."
            )
    """

    model_name: Optional[str] = None

    def __init__(
        self,
        *,
        model_name: Optional[str] = "minishlab/potion-base-8M",
        **kwargs: Any,
    ):
        try:
            from model2vec import StaticModel
        except ImportError as e:
            raise ImportError(
                "Unable to import model2vec, please install with "
                "`pip install -U model2vec`."
            ) from e
        self.model = StaticModel.from_pretrained(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using the model2vec embeddings model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        return self.model.encode_as_sequence(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the model2vec embeddings model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """

        return self.model.encode(text)
