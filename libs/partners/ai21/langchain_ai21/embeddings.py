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
    """AI21 embedding model integration.

        Install ``langchain_ai21`` and set environment variable ``AI21_API_KEY``.

        .. code-block:: bash

            pip install -U langchain_ai21
            export AI21_API_KEY="your-api-key"

    Key init args — client params:
      api_key: Optional[SecretStr]
      batch_size: int
        The number of texts that will be sent to the API in each batch.
        Use larger batch sizes if working with many short texts. This will reduce
        the number of API calls made, and can improve the time it takes to embed
        a large number of texts.
      num_retries: Optional[int]
       Maximum number of retries for API requests before giving up.
      timeout_sec: Optional[float]
        Timeout in seconds for API requests. If not set, it will default to the
        value of the environment variable `AI21_TIMEOUT_SEC` or 300 seconds.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_ai21 import AI21Embeddings

            embed = AI21Embeddings(
                # api_key="...",
                # batch_size=128,
            )

    Embed single text:
        .. code-block:: python

            input_text = "The meaning of life is 42"
            vector = embed.embed_query(input_text)
            print(vector[:3])

        .. code-block:: python

            [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]

    Embed multiple texts:
        .. code-block:: python

             input_texts = ["Document 1...", "Document 2..."]
            vectors = embed.embed_documents(input_texts)
            print(len(vectors))
            # The first 3 coordinates for the first vector
            print(vectors[0][:3])

        .. code-block:: python

            2
            [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]
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
