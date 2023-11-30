from typing import Any, Dict, List, Optional

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.schema.embeddings import Embeddings
from langchain.utils import get_from_dict_or_env


class CohereEmbeddings(BaseModel, Embeddings):
    """Cohere embedding models.

    To use, you should have the ``cohere`` python package installed, and the
    environment variable ``COHERE_API_KEY`` set with your API key or pass it
    as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.embeddings import CohereEmbeddings
            cohere = CohereEmbeddings(
                model="embed-english-light-v3.0",
                input_type="search_document",
                cohere_api_key="my-api-key"
            )
    """

    client: Any  #: :meta private:
    """Cohere client."""
    async_client: Any  #: :meta private:
    """Cohere async client."""
    model: str = "embed-english-v2.0"
    """Model name to use."""
    input_type: Optional[str] = None
    """
    This applies to embed v3 models only.

    input_type="search_document": Use this for texts (documents) you want to
    store in your vector database
    input_type="search_query": Use this for search queries to find the most
    relevant documents in your vector database
    input_type="classification": Use this if you use the embeddings as an input
    for a classification system
    input_type="clustering": Use this if you use the embeddings for text
    clustering

    Using these input types ensures the highest possible quality for the
    respective tasks. If you want to use the embeddings for multiple use
    cases, we recommend using input_type="search_document".
    """

    truncate: Optional[str] = None
    """Truncate embeddings that are too long from start or end ("NONE"|"START"|"END")"""

    cohere_api_key: Optional[str] = None

    max_retries: Optional[int] = None
    """Maximum number of retries to make when generating."""
    request_timeout: Optional[float] = None
    """Timeout in seconds for the Cohere API request."""
    user_agent: str = "langchain"
    """Identifier for the application making the request."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        cohere_api_key = get_from_dict_or_env(
            values, "cohere_api_key", "COHERE_API_KEY"
        )
        max_retries = values.get("max_retries")
        request_timeout = values.get("request_timeout")

        try:
            import cohere

            client_name = values["user_agent"]
            values["client"] = cohere.Client(
                cohere_api_key,
                max_retries=max_retries,
                timeout=request_timeout,
                client_name=client_name,
            )
            values["async_client"] = cohere.AsyncClient(
                cohere_api_key,
                max_retries=max_retries,
                timeout=request_timeout,
                client_name=client_name,
            )
        except ImportError:
            raise ValueError(
                "Could not import cohere python package. "
                "Please install it with `pip install cohere`."
            )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Cohere's embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = self.client.embed(
            model=self.model,
            texts=texts,
            input_type=self.input_type or "search_document",
            truncate=self.truncate,
        ).embeddings
        return [list(map(float, e)) for e in embeddings]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async call out to Cohere's embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = await self.async_client.embed(
            model=self.model,
            texts=texts,
            input_type=self.input_type or "search_document",
            truncate=self.truncate,
        )
        return [list(map(float, e)) for e in embeddings.embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Call out to Cohere's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embeddings = self.client.embed(
            model=self.model,
            texts=[text],
            input_type=self.input_type or "search_query",
            truncate=self.truncate,
        ).embeddings
        return [list(map(float, e)) for e in embeddings][0]

    async def aembed_query(self, text: str) -> List[float]:
        """Async call out to Cohere's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embeddings = await self.async_client.embed(
            model=self.model,
            texts=[text],
            input_type=self.input_type or "search_query",
            truncate=self.truncate,
        )
        return [list(map(float, e)) for e in embeddings.embeddings][0]