from typing import Any, List, Optional

from ai21.models import EmbedType
from langchain_core.embeddings import Embeddings

from langchain_ai21.ai21_base import AI21Base

_DEFAULT_CHUNK_SIZE = 128


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

    chunk_size: int = _DEFAULT_CHUNK_SIZE
    """Maximum number of texts to embed in each batch"""

    def embed_documents(
        self,
        texts: List[str],
        chunk_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Embed search docs."""
        return self._send_embeddings(
            texts=texts,
            chunk_size=chunk_size or self.chunk_size,
            embed_type=EmbedType.SEGMENT,
            **kwargs,
        )

    def embed_query(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[float]:
        """Embed query text."""
        return self._send_embeddings(
            texts=[text],
            chunk_size=chunk_size or self.chunk_size,
            embed_type=EmbedType.QUERY,
            **kwargs,
        )[0]

    def _send_embeddings(
        self, texts: List[str], chunk_size: int, embed_type: EmbedType, **kwargs: Any
    ) -> List[List[float]]:
        chunks = self._get_len_safe_embeddings(texts, chunk_size)
        responses = [
            self.client.embed.create(
                texts=chunk,
                type=embed_type,
                **kwargs,
            )
            for chunk in chunks
        ]

        return [
            result.embedding for response in responses for result in response.results
        ]

    def _get_len_safe_embeddings(
        self, texts: List[str], chunk_size: int
    ) -> List[List[str]]:
        result = []

        for i in range(0, len(texts), chunk_size):
            result.append(texts[i : i + chunk_size])

        return result
