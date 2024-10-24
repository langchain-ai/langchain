"""Wrapper around text2vec embedding models."""

from typing import Any, List, Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel


class Text2vecEmbeddings(Embeddings, BaseModel):
    """text2vec embedding models.

    Install text2vec first, run 'pip install -U text2vec'.
    The github repository for text2vec is : https://github.com/shibing624/text2vec

    Example:
        .. code-block:: python

            from langchain_community.embeddings.text2vec import Text2vecEmbeddings

            embedding = Text2vecEmbeddings()
            embedding.embed_documents([
                "This is a CoSENT(Cosine Sentence) model.",
                "It maps sentences to a 768 dimensional dense vector space.",
            ])
            embedding.embed_query(
                "It can be used for text matching or semantic search."
            )
    """

    model_name_or_path: Optional[str] = None
    encoder_type: Any = "MEAN"
    max_seq_length: int = 256
    device: Optional[str] = None
    model: Any = None

    def __init__(
        self,
        *,
        model: Any = None,
        model_name_or_path: Optional[str] = None,
        **kwargs: Any,
    ):
        try:
            from text2vec import SentenceModel
        except ImportError as e:
            raise ImportError(
                "Unable to import text2vec, please install with "
                "`pip install -U text2vec`."
            ) from e

        model_kwargs = {}
        if model_name_or_path is not None:
            model_kwargs["model_name_or_path"] = model_name_or_path
        model = model or SentenceModel(**model_kwargs, **kwargs)
        super().__init__(model=model, model_name_or_path=model_name_or_path, **kwargs)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using the text2vec embeddings model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        return self.model.encode(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the text2vec embeddings model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """

        return self.model.encode(text)
