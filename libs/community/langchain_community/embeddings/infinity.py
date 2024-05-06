"""written under MIT Licence, Michael Feil 2023."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env

__all__ = ["InfinityEmbeddings"]


class InfinityEmbeddings(BaseModel, Embeddings):
    """Self-hosted embedding models for `infinity` package.

    See https://github.com/michaelfeil/infinity
    This also works for text-embeddings-inference and other
    self-hosted openai-compatible servers.

    Infinity is a package to interact with Embedding Models on https://github.com/michaelfeil/infinity


    Example:
        .. code-block:: python

            from langchain_community.embeddings import InfinityEmbeddings
            InfinityEmbeddings(
                model="BAAI/bge-small",
                infinity_api_url="http://localhost:7997",
            )
    """

    model: str
    "Underlying Infinity model id."

    infinity_api_url: str = "http://localhost:7997"
    """Endpoint URL to use."""

    client: Any = None  #: :meta private:
    """Infinity client."""

    # LLM call kwargs
    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""

        values["infinity_api_url"] = get_from_dict_or_env(
            values, "infinity_api_url", "INFINITY_API_URL"
        )

        values["client"] = TinyAsyncOpenAIInfinityEmbeddingClient(
            host=values["infinity_api_url"],
        )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Infinity's embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = self.client.embed(
            model=self.model,
            texts=texts,
        )
        return embeddings

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async call out to Infinity's embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = await self.client.aembed(
            model=self.model,
            texts=texts,
        )
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Call out to Infinity's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]

    async def aembed_query(self, text: str) -> List[float]:
        """Async call out to Infinity's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embeddings = await self.aembed_documents([text])
        return embeddings[0]


class TinyAsyncOpenAIInfinityEmbeddingClient:  #: :meta private:
    """Helper tool to embed Infinity.

    It is not a part of Langchain's stable API,
    direct use discouraged.

    Example:
        .. code-block:: python


            mini_client = TinyAsyncInfinityEmbeddingClient(
            )
            embeds = mini_client.embed(
                model="BAAI/bge-small",
                text=["doc1", "doc2"]
            )
            # or
            embeds = await mini_client.aembed(
                model="BAAI/bge-small",
                text=["doc1", "doc2"]
            )

    """

    def __init__(
        self,
        host: str = "http://localhost:7797/v1",
        aiosession: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self.host = host
        self.aiosession = aiosession

        if self.host is None or len(self.host) < 3:
            raise ValueError(" param `host` must be set to a valid url")
        self._batch_size = 128

    @staticmethod
    def _permute(
        texts: List[str], sorter: Callable = len
    ) -> Tuple[List[str], Callable]:
        """Sort texts in ascending order, and
        delivers a lambda expr, which can sort a same length list
        https://github.com/UKPLab/sentence-transformers/blob/
        c5f93f70eca933c78695c5bc686ceda59651ae3b/sentence_transformers/SentenceTransformer.py#L156

        Args:
            texts (List[str]): _description_
            sorter (Callable, optional): _description_. Defaults to len.

        Returns:
            Tuple[List[str], Callable]: _description_

        Example:
            ```
            texts = ["one","three","four"]
            perm_texts, undo = self._permute(texts)
            texts == undo(perm_texts)
            ```
        """

        if len(texts) == 1:
            # special case query
            return texts, lambda t: t
        length_sorted_idx = np.argsort([-sorter(sen) for sen in texts])
        texts_sorted = [texts[idx] for idx in length_sorted_idx]

        return texts_sorted, lambda unsorted_embeddings: [  # noqa E731
            unsorted_embeddings[idx] for idx in np.argsort(length_sorted_idx)
        ]

    def _batch(self, texts: List[str]) -> List[List[str]]:
        """
        splits Lists of text parts into batches of size max `self._batch_size`
        When encoding vector database,

        Args:
            texts (List[str]): List of sentences
            self._batch_size (int, optional): max batch size of one request.

        Returns:
            List[List[str]]: Batches of List of sentences
        """
        if len(texts) == 1:
            # special case query
            return [texts]
        batches = []
        for start_index in range(0, len(texts), self._batch_size):
            batches.append(texts[start_index : start_index + self._batch_size])
        return batches

    @staticmethod
    def _unbatch(batch_of_texts: List[List[Any]]) -> List[Any]:
        if len(batch_of_texts) == 1 and len(batch_of_texts[0]) == 1:
            # special case query
            return batch_of_texts[0]
        texts = []
        for sublist in batch_of_texts:
            texts.extend(sublist)
        return texts

    def _kwargs_post_request(self, model: str, texts: List[str]) -> Dict[str, Any]:
        """Build the kwargs for the Post request, used by sync

        Args:
            model (str): _description_
            texts (List[str]): _description_

        Returns:
            Dict[str, Collection[str]]: _description_
        """
        return dict(
            url=f"{self.host}/embeddings",
            headers={
                # "accept": "application/json",
                "content-type": "application/json",
            },
            json=dict(
                input=texts,
                model=model,
            ),
        )

    def _sync_request_embed(
        self, model: str, batch_texts: List[str]
    ) -> List[List[float]]:
        response = requests.post(
            **self._kwargs_post_request(model=model, texts=batch_texts)
        )
        if response.status_code != 200:
            raise Exception(
                f"Infinity returned an unexpected response with status "
                f"{response.status_code}: {response.text}"
            )
        return [e["embedding"] for e in response.json()["data"]]

    def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        """call the embedding of model

        Args:
            model (str): to embedding model
            texts (List[str]): List of sentences to embed.

        Returns:
            List[List[float]]: List of vectors for each sentence
        """
        perm_texts, unpermute_func = self._permute(texts)
        perm_texts_batched = self._batch(perm_texts)

        # Request
        map_args = (
            self._sync_request_embed,
            [model] * len(perm_texts_batched),
            perm_texts_batched,
        )
        if len(perm_texts_batched) == 1:
            embeddings_batch_perm = list(map(*map_args))
        else:
            with ThreadPoolExecutor(32) as p:
                embeddings_batch_perm = list(p.map(*map_args))

        embeddings_perm = self._unbatch(embeddings_batch_perm)
        embeddings = unpermute_func(embeddings_perm)
        return embeddings

    async def _async_request(
        self, session: aiohttp.ClientSession, kwargs: Dict[str, Any]
    ) -> List[List[float]]:
        async with session.post(**kwargs) as response:
            if response.status != 200:
                raise Exception(
                    f"Infinity returned an unexpected response with status "
                    f"{response.status}: {response.text}"
                )
            embedding = (await response.json())["embeddings"]
            return [e["embedding"] for e in embedding]

    async def aembed(self, model: str, texts: List[str]) -> List[List[float]]:
        """call the embedding of model, async method

        Args:
            model (str): to embedding model
            texts (List[str]): List of sentences to embed.

        Returns:
            List[List[float]]: List of vectors for each sentence
        """
        perm_texts, unpermute_func = self._permute(texts)
        perm_texts_batched = self._batch(perm_texts)

        # Request
        if self.aiosession is None:
            self.aiosession = aiohttp.ClientSession(
                trust_env=True, connector=aiohttp.TCPConnector(limit=32)
            )
        async with self.aiosession as session:
            embeddings_batch_perm = await asyncio.gather(
                *[
                    self._async_request(
                        session=session,
                        **self._kwargs_post_request(model=model, texts=t),
                    )
                    for t in perm_texts_batched
                ]
            )

        embeddings_perm = self._unbatch(embeddings_batch_perm)
        embeddings = unpermute_func(embeddings_perm)
        return embeddings
