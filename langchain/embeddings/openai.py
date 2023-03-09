"""Wrapper around OpenAI embedding models."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Extra, root_validator
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


def _create_retry_decorator(embeddings: OpenAIEmbeddings) -> Callable[[Any], Any]:
    import openai

    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(embeddings.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def embed_with_retry(embeddings: OpenAIEmbeddings, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(embeddings)

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Any:
        return embeddings.client.create(**kwargs)

    return _completion_with_retry(**kwargs)


class OpenAIEmbeddings(BaseModel, Embeddings):
    """Wrapper around OpenAI embedding models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key or pass it
    as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.embeddings import OpenAIEmbeddings
            openai = OpenAIEmbeddings(openai_api_key="my-api-key")
    """

    client: Any  #: :meta private:
    document_model_name: str = "text-embedding-ada-002"
    query_model_name: str = "text-embedding-ada-002"
    embedding_ctx_length: int = -1
    openai_api_key: Optional[str] = None
    chunk_size: int = 1000
    """Maximum number of texts to embed in each batch"""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    # TODO: deprecate this
    @root_validator(pre=True)
    def get_model_names(cls, values: Dict) -> Dict:
        """Get model names from just old model name."""
        if "model_name" in values:
            if "document_model_name" in values:
                raise ValueError(
                    "Both `model_name` and `document_model_name` were provided, "
                    "but only one should be."
                )
            if "query_model_name" in values:
                raise ValueError(
                    "Both `model_name` and `query_model_name` were provided, "
                    "but only one should be."
                )
            model_name = values.pop("model_name")
            values["document_model_name"] = f"text-search-{model_name}-doc-001"
            values["query_model_name"] = f"text-search-{model_name}-query-001"
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        try:
            import openai

            openai.api_key = openai_api_key
            values["client"] = openai.Embedding
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please it install it with `pip install openai`."
            )
        return values

    # please refer to
    # https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    def _get_len_safe_embeddings(
        self, texts: List[str], *, engine: str, chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        embeddings: List[List[float]] = [[] for i in range(len(texts))]
        try:
            import tiktoken

            tokens = []
            indices = []
            encoding = tiktoken.model.encoding_for_model(self.document_model_name)
            for i, text in enumerate(texts):
                # replace newlines, which can negatively affect performance.
                text = text.replace("\n", " ")
                token = encoding.encode(text)
                for j in range(0, len(token), self.embedding_ctx_length):
                    tokens += [token[j : j + self.embedding_ctx_length]]
                    indices += [i]

            batched_embeddings = []
            _chunk_size = chunk_size or self.chunk_size
            for i in range(0, len(tokens), _chunk_size):
                response = embed_with_retry(
                    self,
                    input=tokens[i : i + _chunk_size],
                    engine=self.document_model_name,
                )
                batched_embeddings += [r["embedding"] for r in response["data"]]

            results: List[List[List[float]]] = [[] for i in range(len(texts))]
            lens: List[List[int]] = [[] for i in range(len(texts))]
            for i in range(len(indices)):
                results[indices[i]].append(batched_embeddings[i])
                lens[indices[i]].append(len(batched_embeddings[i]))

            for i in range(len(texts)):
                average = np.average(results[i], axis=0, weights=lens[i])
                embeddings[i] = (average / np.linalg.norm(average)).tolist()

            return embeddings

        except ImportError:
            raise ValueError(
                "Could not import tiktoken python package. "
                "This is needed in order to for OpenAIEmbeddings. "
                "Please it install it with `pip install tiktoken`."
            )

    def _embedding_func(self, text: str, *, engine: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint."""
        # replace newlines, which can negatively affect performance.
        if self.embedding_ctx_length > 0:
            return self._get_len_safe_embeddings([text], engine=engine)[0]
        else:
            text = text.replace("\n", " ")
            return embed_with_retry(self, input=[text], engine=engine)["data"][0][
                "embedding"
            ]

    def embed_documents(
        self, texts: List[str], chunk_size: Optional[int] = 0
    ) -> List[List[float]]:
        """Call out to OpenAI's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size of embeddings. If None, will use the chunk size
                specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        # handle large batches of texts
        if self.embedding_ctx_length > 0:
            return self._get_len_safe_embeddings(texts, engine=self.document_model_name)
        else:
            results = []
            _chunk_size = chunk_size or self.chunk_size
            for i in range(0, len(texts), _chunk_size):
                response = embed_with_retry(
                    self,
                    input=texts[i : i + _chunk_size],
                    engine=self.document_model_name,
                )
                results += [r["embedding"] for r in response["data"]]
            return results

    def embed_query(self, text: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embedding = self._embedding_func(text, engine=self.query_model_name)
        return embedding
