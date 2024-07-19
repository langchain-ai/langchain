"""
TextEmbed: Embedding Inference Server

TextEmbed provides a high-throughput, low-latency solution for serving embeddings.
It supports various sentence-transformer models.
Now, it includes the ability to deploy image embedding models.
TextEmbed offers flexibility and scalability for diverse applications.

TextEmbed is maintained by Keval Dekivadiya and is licensed under the Apache-2.0 license.
"""  # noqa: E501

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env

__all__ = ["TextEmbedEmbeddings"]


class TextEmbedEmbeddings(BaseModel, Embeddings):
    """
    A class to handle embedding requests to the TextEmbed API.

    Attributes:
        model : The TextEmbed model ID to use for embeddings.
        api_url : The base URL for the TextEmbed API.
        api_key : The API key for authenticating with the TextEmbed API.
        client : The TextEmbed client instance.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import TextEmbedEmbeddings

            embeddings = TextEmbedEmbeddings(
                model="sentence-transformers/clip-ViT-B-32",
                api_url="http://localhost:8000/v1",
                api_key="<API_KEY>"
            )

    For more information: https://github.com/kevaldekivadiya2415/textembed/blob/main/docs/setup.md
    """  # noqa: E501

    model: str
    """Underlying TextEmbed model id."""

    api_url: str = "http://localhost:8000/v1"
    """Endpoint URL to use."""

    api_key: str = "None"
    """API Key for authentication"""

    client: Any = None
    """TextEmbed client."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator(pre=False, skip_on_failure=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and URL exist in the environment.

        Args:
            values (Dict): Dictionary of values to validate.

        Returns:
            Dict: Validated values.
        """
        values["api_url"] = get_from_dict_or_env(values, "api_url", "API_URL")
        values["api_key"] = get_from_dict_or_env(values, "api_key", "API_KEY")

        values["client"] = AsyncOpenAITextEmbedEmbeddingClient(
            host=values["api_url"], api_key=values["api_key"]
        )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to TextEmbed's embedding endpoint.

        Args:
            texts (List[str]): The list of texts to embed.

        Returns:
            List[List[float]]: List of embeddings, one for each text.
        """
        embeddings = self.client.embed(
            model=self.model,
            texts=texts,
        )
        return embeddings

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async call out to TextEmbed's embedding endpoint.

        Args:
            texts (List[str]): The list of texts to embed.

        Returns:
            List[List[float]]: List of embeddings, one for each text.
        """
        embeddings = await self.client.aembed(
            model=self.model,
            texts=texts,
        )
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Call out to TextEmbed's embedding endpoint for a single query.

        Args:
            text (str): The text to embed.

        Returns:
            List[float]: Embeddings for the text.
        """
        return self.embed_documents([text])[0]

    async def aembed_query(self, text: str) -> List[float]:
        """Async call out to TextEmbed's embedding endpoint for a single query.

        Args:
            text (str): The text to embed.

        Returns:
            List[float]: Embeddings for the text.
        """
        embeddings = await self.aembed_documents([text])
        return embeddings[0]


class AsyncOpenAITextEmbedEmbeddingClient:
    """
    A client to handle synchronous and asynchronous requests to the TextEmbed API.

    Attributes:
        host (str): The base URL for the TextEmbed API.
        api_key (str): The API key for authenticating with the TextEmbed API.
        aiosession (Optional[aiohttp.ClientSession]): The aiohttp session for async requests.
        _batch_size (int): Maximum batch size for a single request.
    """  # noqa: E501

    def __init__(
        self,
        host: str = "http://localhost:8000/v1",
        api_key: Union[str, None] = None,
        aiosession: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self.host = host
        self.api_key = api_key
        self.aiosession = aiosession

        if self.host is None or len(self.host) < 3:
            raise ValueError("Parameter `host` must be set to a valid URL")
        self._batch_size = 256

    @staticmethod
    def _permute(
        texts: List[str], sorter: Callable = len
    ) -> Tuple[List[str], Callable]:
        """
        Sorts texts in ascending order and provides a function to restore the original order.

        Args:
            texts (List[str]): List of texts to sort.
            sorter (Callable, optional): Sorting function, defaults to length.

        Returns:
            Tuple[List[str], Callable]: Sorted texts and a function to restore original order.
        """  # noqa: E501
        if len(texts) == 1:
            return texts, lambda t: t
        length_sorted_idx = np.argsort([-sorter(sen) for sen in texts])
        texts_sorted = [texts[idx] for idx in length_sorted_idx]

        return texts_sorted, lambda unsorted_embeddings: [
            unsorted_embeddings[idx] for idx in np.argsort(length_sorted_idx)
        ]

    def _batch(self, texts: List[str]) -> List[List[str]]:
        """
        Splits a list of texts into batches of size max `self._batch_size`.

        Args:
            texts (List[str]): List of texts to split.

        Returns:
            List[List[str]]: List of batches of texts.
        """
        if len(texts) == 1:
            return [texts]
        batches = []
        for start_index in range(0, len(texts), self._batch_size):
            batches.append(texts[start_index : start_index + self._batch_size])
        return batches

    @staticmethod
    def _unbatch(batch_of_texts: List[List[Any]]) -> List[Any]:
        """
        Merges batches of texts into a single list.

        Args:
            batch_of_texts (List[List[Any]]): List of batches of texts.

        Returns:
            List[Any]: Merged list of texts.
        """
        if len(batch_of_texts) == 1 and len(batch_of_texts[0]) == 1:
            return batch_of_texts[0]
        texts = []
        for sublist in batch_of_texts:
            texts.extend(sublist)
        return texts

    def _kwargs_post_request(self, model: str, texts: List[str]) -> Dict[str, Any]:
        """
        Builds the kwargs for the POST request, used by sync method.

        Args:
            model (str): The model to use for embedding.
            texts (List[str]): List of texts to embed.

        Returns:
            Dict[str, Any]: Dictionary of POST request parameters.
        """
        return dict(
            url=f"{self.host}/embedding",
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json=dict(
                input=texts,
                model=model,
            ),
        )

    def _sync_request_embed(
        self, model: str, batch_texts: List[str]
    ) -> List[List[float]]:
        """
        Sends a synchronous request to the embedding endpoint.

        Args:
            model (str): The model to use for embedding.
            batch_texts (List[str]): Batch of texts to embed.

        Returns:
            List[List[float]]: List of embeddings for the batch.

        Raises:
            Exception: If the response status is not 200.
        """
        response = requests.post(
            **self._kwargs_post_request(model=model, texts=batch_texts)
        )
        if response.status_code != 200:
            raise Exception(
                f"TextEmbed responded with an unexpected status message "
                f"{response.status_code}: {response.text}"
            )
        return [e["embedding"] for e in response.json()["data"]]

    def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts synchronously.

        Args:
            model (str): The model to use for embedding.
            texts (List[str]): List of texts to embed.

        Returns:
            List[List[float]]: List of embeddings for the texts.
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
        self, session: aiohttp.ClientSession, **kwargs: Dict[str, Any]
    ) -> List[List[float]]:
        """
        Sends an asynchronous request to the embedding endpoint.

        Args:
            session (aiohttp.ClientSession): The aiohttp session for the request.
            kwargs (Dict[str, Any]): Dictionary of POST request parameters.

        Returns:
            List[List[float]]: List of embeddings for the request.

        Raises:
            Exception: If the response status is not 200.
        """
        async with session.post(**kwargs) as response:  # type: ignore
            if response.status != 200:
                raise Exception(
                    f"TextEmbed responded with an unexpected status message "
                    f"{response.status}: {response.text}"
                )
            embedding = (await response.json())["data"]
            return [e["embedding"] for e in embedding]

    async def aembed(self, model: str, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts asynchronously.

        Args:
            model (str): The model to use for embedding.
            texts (List[str]): List of texts to embed.

        Returns:
            List[List[float]]: List of embeddings for the texts.
        """
        perm_texts, unpermute_func = self._permute(texts)
        perm_texts_batched = self._batch(perm_texts)

        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=32)
        ) as session:
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
