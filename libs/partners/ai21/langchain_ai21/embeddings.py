from typing import Any, List

from ai21.models import EmbedType
from langchain_core.embeddings import Embeddings

from langchain_ai21.ai21_base import AI21Base


class AI21Embeddings(Embeddings, AI21Base):
    """AI21 Embeddings embedding model.
    To use, you should have the 'AI21_API_KEY' environment variable set
    or pass as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_ai21 import AI21Embeddings

            embeddings = AI21Embeddings()
            query_result = embeddings.embed_query("Hello embeddings world!")
    """

    def embed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """Embed search docs."""
        response = self.client.embed.create(
            texts=texts,
            type=EmbedType.SEGMENT,
            **kwargs,
        )

        return [result.embedding for result in response.results]

    def embed_query(self, text: str, **kwargs: Any) -> List[float]:
        """Embed query text."""
        response = self.client.embed.create(
            texts=[text],
            type=EmbedType.QUERY,
            **kwargs,
        )

        return [result.embedding for result in response.results][0]
