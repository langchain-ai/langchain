from itertools import islice
from typing import Any, Iterator, List, Optional

from ai21.models import EmbedType
from langchain_core.embeddings import Embeddings

from langchain_ai21.ai21_base import AI21Base

_DEFAULT_BATCH_SIZE = 128


def _split_texts_into_batches(texts: List[str], batch_size: int) -> Iterator[List[str]]:
    texts_itr = iter(texts)
    return iter(lambda: list(islice(texts_itr, batch_size)), [])


class AI21Embeddings(Embeddings, AI21Base):
    """AI21 embedding model.

    To use, you should have the 'AI21_API_KEY' environment variable set
    or pass as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_ai21 import AI21Embeddings

            embeddings = AI21Embeddings()
            query_result = embeddings.embed_query("Hello embeddings world!")
    """

    batch_size: int = _DEFAULT_BATCH_SIZE
    """Maximum number of texts to embed in each batch"""

    def embed_documents(
        self,
        texts: List[str],
        *,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Embed search docs."""
        return self._send_embeddings(
            texts=texts,
            batch_size=batch_size or self.batch_size,
            embed_type=EmbedType.SEGMENT,
            **kwargs,
        )

    def embed_query(
        self,
        text: str,
        *,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[float]:
        """Embed query text."""
        return self._send_embeddings(
            texts=[text],
            batch_size=batch_size or self.batch_size,
            embed_type=EmbedType.QUERY,
            **kwargs,
        )[0]

    def _send_embeddings(
        self, texts: List[str], *, batch_size: int, embed_type: EmbedType, **kwargs: Any
    ) -> List[List[float]]:
        chunks = _split_texts_into_batches(texts, batch_size)
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
