"""Wrapper around text2vec embedding models."""

from typing import Any, List

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel
from text2vec import SentenceModel


class Text2vecEmbeddings(BaseModel, Embeddings):
    """text2vec embedding models.

    Install text2vec first, run 'pip install -U text2vec'.
    Example:
        .. code-block:: python

            from langchain_community.embeddings.text2vec import Text2vecEmbeddings

            embedding = Text2vecEmbeddings()
            bookend.embed_documents([
                "This is a CoSENT(Cosine Sentence) model.",
                "It maps sentences to a 768 dimensional dense vector space.",
            ])
            bookend.embed_query(
                "It can be used for text matching or semantic search."
            )
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using the text2vec embeddings model.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        
        m = SentenceModel()
        return m.encode(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the text2vec embeddings model.
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        
        m = SentenceModel()        
        return m.encode(text)
