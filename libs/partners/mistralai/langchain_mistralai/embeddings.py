import logging
from typing import Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import (
    BaseModel,
    Extra,
    Field,
    SecretStr,
    root_validator,
)
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from mistralai.async_client import MistralAsyncClient
from mistralai.client import MistralClient
from mistralai.constants import (
    ENDPOINT as DEFAULT_MISTRAL_ENDPOINT,
)
from mistralai.exceptions import MistralException

logger = logging.getLogger(__name__)


class MistralAIEmbeddings(BaseModel, Embeddings):
    """MistralAI embedding models.

    To use, set the environment variable `MISTRAL_API_KEY` is set with your API key or
    pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_mistralai import MistralAIEmbeddings
            mistral = MistralAIEmbeddings(
                model="mistral-embed",
                mistral_api_key="my-api-key"
            )
    """

    client: MistralClient = Field(default=None)  #: :meta private:
    async_client: MistralAsyncClient = Field(default=None)  #: :meta private:
    mistral_api_key: Optional[SecretStr] = None
    endpoint: str = DEFAULT_MISTRAL_ENDPOINT
    max_retries: int = 5
    timeout: int = 120
    max_concurrent_requests: int = 64

    model: str = "mistral-embed"

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate configuration."""

        values["mistral_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values, "mistral_api_key", "MISTRAL_API_KEY", default=""
            )
        )
        values["client"] = MistralClient(
            api_key=values["mistral_api_key"].get_secret_value(),
            endpoint=values["endpoint"],
            max_retries=values["max_retries"],
            timeout=values["timeout"],
        )
        values["async_client"] = MistralAsyncClient(
            api_key=values["mistral_api_key"].get_secret_value(),
            endpoint=values["endpoint"],
            max_retries=values["max_retries"],
            timeout=values["timeout"],
            max_concurrent_requests=values["max_concurrent_requests"],
        )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        try:
            embeddings_batch_response = self.client.embeddings(
                model=self.model,
                input=texts,
            )
            return [
                list(map(float, embedding_obj.embedding))
                for embedding_obj in embeddings_batch_response.data
            ]
        except MistralException as e:
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
            embeddings_batch_response = await self.async_client.embeddings(
                model=self.model,
                input=texts,
            )
            return [
                list(map(float, embedding_obj.embedding))
                for embedding_obj in embeddings_batch_response.data
            ]
        except MistralException as e:
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
