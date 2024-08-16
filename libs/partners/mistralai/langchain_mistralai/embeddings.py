import asyncio
import logging
import warnings
from typing import Dict, Iterable, List, Optional

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import (
    BaseModel,
    Field,
    SecretStr,
    root_validator,
)
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from tokenizers import Tokenizer  # type: ignore

logger = logging.getLogger(__name__)

MAX_TOKENS = 16_000
"""A batching parameter for the Mistral API. This is NOT the maximum number of tokens
accepted by the embedding model for each document/chunk, but rather the maximum number 
of tokens that can be sent in a single request to the Mistral API (across multiple
documents/chunks)"""


class DummyTokenizer:
    """Dummy tokenizer for when tokenizer cannot be accessed (e.g., via Huggingface)"""

    def encode_batch(self, texts: List[str]) -> List[List[str]]:
        return [list(text) for text in texts]


class MistralAIEmbeddings(BaseModel, Embeddings):
    """MistralAI embedding model integration.

    Setup:
        Install ``langchain_mistralai`` and set environment variable
        ``MISTRAL_API_KEY``.

        .. code-block:: bash

            pip install -U langchain_mistralai
            export MISTRAL_API_KEY="your-api-key"

    Key init args — completion params:
        model: str
            Name of MistralAI model to use.

    Key init args — client params:
      api_key: Optional[SecretStr]
        The API key for the MistralAI API. If not provided, it will be read from the
        environment variable `MISTRAL_API_KEY`.
      max_retries: int
        The number of times to retry a request if it fails.
      timeout: int
        The number of seconds to wait for a response before timing out.
      max_concurrent_requests: int
        The maximum number of concurrent requests to make to the Mistral API.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from __module_name__ import MistralAIEmbeddings

            embed = MistralAIEmbeddings(
                model="mistral-embed",
                # api_key="...",
                # other params...
            )

    Embed single text:
        .. code-block:: python

            input_text = "The meaning of life is 42"
            vector = embed.embed_query(input_text)
            print(vector[:3])

        .. code-block:: python

            [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]

    Embed multiple text:
        .. code-block:: python

             input_texts = ["Document 1...", "Document 2..."]
            vectors = embed.embed_documents(input_texts)
            print(len(vectors))
            # The first 3 coordinates for the first vector
            print(vectors[0][:3])

        .. code-block:: python

            2
            [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]

    Async:
        .. code-block:: python

            vector = await embed.aembed_query(input_text)
           print(vector[:3])

            # multiple:
            # await embed.aembed_documents(input_texts)

        .. code-block:: python

            [-0.009100092574954033, 0.005071679595857859, -0.0029193938244134188]
    """

    client: httpx.Client = Field(default=None)  #: :meta private:
    async_client: httpx.AsyncClient = Field(default=None)  #: :meta private:
    mistral_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    endpoint: str = "https://api.mistral.ai/v1/"
    max_retries: int = 5
    timeout: int = 120
    max_concurrent_requests: int = 64
    tokenizer: Tokenizer = Field(default=None)

    model: str = "mistral-embed"

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate configuration."""

        values["mistral_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values, "mistral_api_key", "MISTRAL_API_KEY", default=""
            )
        )
        api_key_str = values["mistral_api_key"].get_secret_value()
        # todo: handle retries
        if not values.get("client"):
            values["client"] = httpx.Client(
                base_url=values["endpoint"],
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_key_str}",
                },
                timeout=values["timeout"],
            )
        # todo: handle retries and max_concurrency
        if not values.get("async_client"):
            values["async_client"] = httpx.AsyncClient(
                base_url=values["endpoint"],
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_key_str}",
                },
                timeout=values["timeout"],
            )
        if values["tokenizer"] is None:
            try:
                values["tokenizer"] = Tokenizer.from_pretrained(
                    "mistralai/Mixtral-8x7B-v0.1"
                )
            except IOError:  # huggingface_hub GatedRepoError
                warnings.warn(
                    "Could not download mistral tokenizer from Huggingface for "
                    "calculating batch sizes. Set a Huggingface token via the "
                    "HF_TOKEN environment variable to download the real tokenizer. "
                    "Falling back to a dummy tokenizer that uses `len()`."
                )
                values["tokenizer"] = DummyTokenizer()
        return values

    def _get_batches(self, texts: List[str]) -> Iterable[List[str]]:
        """Split a list of texts into batches of less than 16k tokens
        for Mistral API."""
        batch: List[str] = []
        batch_tokens = 0

        text_token_lengths = [
            len(encoded) for encoded in self.tokenizer.encode_batch(texts)
        ]

        for text, text_tokens in zip(texts, text_token_lengths):
            if batch_tokens + text_tokens > MAX_TOKENS:
                if len(batch) > 0:
                    # edge case where first batch exceeds max tokens
                    # should not yield an empty batch.
                    yield batch
                batch = [text]
                batch_tokens = text_tokens
            else:
                batch.append(text)
                batch_tokens += text_tokens
        if batch:
            yield batch

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        try:
            batch_responses = (
                self.client.post(
                    url="/embeddings",
                    json=dict(
                        model=self.model,
                        input=batch,
                    ),
                )
                for batch in self._get_batches(texts)
            )
            return [
                list(map(float, embedding_obj["embedding"]))
                for response in batch_responses
                for embedding_obj in response.json()["data"]
            ]
        except Exception as e:
            logger.error(f"An error occurred with MistralAI: {e}")
            raise

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        try:
            batch_responses = await asyncio.gather(
                *[
                    self.async_client.post(
                        url="/embeddings",
                        json=dict(
                            model=self.model,
                            input=batch,
                        ),
                    )
                    for batch in self._get_batches(texts)
                ]
            )
            return [
                list(map(float, embedding_obj["embedding"]))
                for response in batch_responses
                for embedding_obj in response.json()["data"]
            ]
        except Exception as e:
            logger.error(f"An error occurred with MistralAI: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        return self.embed_documents([text])[0]

    async def aembed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        return (await self.aembed_documents([text]))[0]
