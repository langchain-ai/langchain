from typing import Any, Dict, List, Optional

from langchain_core._api.deprecation import deprecated
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, model_validator

from langchain_community.llms.cohere import _create_retry_decorator


@deprecated(
    since="0.0.30",
    removal="1.0",
    alternative_import="langchain_cohere.CohereEmbeddings",
)
class CohereEmbeddings(BaseModel, Embeddings):
    """Cohere embedding models.

    To use, you should have the ``cohere`` python package installed, and the
    environment variable ``COHERE_API_KEY`` set with your API key or pass it
    as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import CohereEmbeddings
            cohere = CohereEmbeddings(
                model="embed-english-light-v3.0",
                cohere_api_key="my-api-key"
            )
    """

    client: Any = None  #: :meta private:
    """Cohere client."""
    async_client: Any = None  #: :meta private:
    """Cohere async client."""
    model: str = "embed-english-v2.0"
    """Model name to use."""

    truncate: Optional[str] = None
    """Truncate embeddings that are too long from start or end ("NONE"|"START"|"END")"""

    cohere_api_key: Optional[str] = None

    max_retries: int = 3
    """Maximum number of retries to make when generating."""
    request_timeout: Optional[float] = None
    """Timeout in seconds for the Cohere API request."""
    user_agent: str = "langchain"
    """Identifier for the application making the request."""

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and python package exists in environment."""
        cohere_api_key = get_from_dict_or_env(
            values, "cohere_api_key", "COHERE_API_KEY"
        )
        request_timeout = values.get("request_timeout")

        try:
            import cohere

            client_name = values["user_agent"]
            values["client"] = cohere.Client(
                cohere_api_key,
                timeout=request_timeout,
                client_name=client_name,
            )
            values["async_client"] = cohere.AsyncClient(
                cohere_api_key,
                timeout=request_timeout,
                client_name=client_name,
            )
        except ImportError:
            raise ImportError(
                "Could not import cohere python package. "
                "Please install it with `pip install cohere`."
            )
        return values

    def embed_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the embed call."""
        retry_decorator = _create_retry_decorator(self.max_retries)

        @retry_decorator
        def _embed_with_retry(**kwargs: Any) -> Any:
            return self.client.embed(**kwargs)

        return _embed_with_retry(**kwargs)

    def aembed_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the embed call."""
        retry_decorator = _create_retry_decorator(self.max_retries)

        @retry_decorator
        async def _embed_with_retry(**kwargs: Any) -> Any:
            return await self.async_client.embed(**kwargs)

        return _embed_with_retry(**kwargs)

    def embed(
        self, texts: List[str], *, input_type: Optional[str] = None
    ) -> List[List[float]]:
        embeddings = self.embed_with_retry(
            model=self.model,
            texts=texts,
            input_type=input_type,
            truncate=self.truncate,
        ).embeddings
        return [list(map(float, e)) for e in embeddings]

    async def aembed(
        self, texts: List[str], *, input_type: Optional[str] = None
    ) -> List[List[float]]:
        embeddings = (
            await self.aembed_with_retry(
                model=self.model,
                texts=texts,
                input_type=input_type,
                truncate=self.truncate,
            )
        ).embeddings
        return [list(map(float, e)) for e in embeddings]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        return self.embed(texts, input_type="search_document")

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async call out to Cohere's embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        return await self.aembed(texts, input_type="search_document")

    def embed_query(self, text: str) -> List[float]:
        """Call out to Cohere's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed([text], input_type="search_query")[0]

    async def aembed_query(self, text: str) -> List[float]:
        """Async call out to Cohere's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return (await self.aembed([text], input_type="search_query"))[0]
