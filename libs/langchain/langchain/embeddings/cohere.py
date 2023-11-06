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
                model="embed-english-light-v2.0", cohere_api_key="my-api-key"
            )
    """

    client: Any  #: :meta private:
    """Cohere client."""
    async_client: Any  #: :meta private:
    """Cohere async client."""
    model: str = "embed-english-v2.0"
    """Model name to use."""

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
    
    def _embed(self, texts: List[str], input_type: str) -> List[List[float]]:
        """Call Cohere's embed endpoint synchronously.

        Args:
            texts: The list of texts to embed.
            input_type: The type of input for embedding.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = self.client.embed(
            model=self.model,
            texts=texts,
            truncate=self.truncate,
            input_type=input_type
        ).embeddings
        return [list(map(float, e)) for e in embeddings]

    async def _aembed(self, texts: List[str], input_type: str) -> List[List[float]]:
        """Call Cohere's embed endpoint asynchronously.

        Args:
            texts: The list of texts to embed.
            input_type: The type of input for embedding.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = await self.async_client.embed(
            model=self.model,
            texts=texts,
            truncate=self.truncate,
            input_type=input_type
        )
        return [list(map(float, e)) for e in embeddings.embeddings]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self._embed(texts, "search_document")

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously embed a list of documents."""
        return await self._aembed(texts, "search_document")

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self._embed([text], "search_query")[0]

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronously embed a single query."""
        return (await self._aembed([text], "search_query"))[0]