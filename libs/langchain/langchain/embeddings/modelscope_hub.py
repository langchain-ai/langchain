from typing import Any, List, Optional

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, Extra


class ModelScopeEmbeddings(BaseModel, Embeddings):
    """ModelScopeHub embedding models.

    To use, you should have the ``modelscope`` python package installed.

    Example:
        .. code-block:: python

            from langchain.embeddings import ModelScopeEmbeddings
            model_id = "damo/nlp_corom_sentence-embedding_english-base"
            embed = ModelScopeEmbeddings(model_id=model_id, model_revision="v1.0.0")
    """

    embed: Any
    model_id: str = "damo/nlp_corom_sentence-embedding_english-base"
    """Model name to use."""
    model_revision: Optional[str] = None

    def __init__(self, **kwargs: Any):
        """Initialize the modelscope"""
        super().__init__(**kwargs)
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
        except ImportError as e:
            raise ImportError(
                "Could not import some python packages."
                "Please install it with `pip install modelscope`."
            ) from e
        self.embed = pipeline(
            Tasks.sentence_embedding,
            model=self.model_id,
            model_revision=self.model_revision,
        )

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a modelscope embedding model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        inputs = {"source_sentence": texts}
        embeddings = self.embed(input=inputs)["text_embedding"]
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a modelscope embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        inputs = {"source_sentence": [text]}
        embedding = self.embed(input=inputs)["text_embedding"][0]
        return embedding.tolist()
