from typing import Any, Dict, List, Optional, ClassVar, Type
import logging

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)

class MistralAIEmbeddings(BaseModel, Embeddings):
    """MistralAI embedding models.

    To use, ensure the `mistralai` python package is installed, and the
    environment variable `MISTRAL_API_KEY` is set with your API key or pass it
    as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_mistralai import MistralAIEmbeddings
            mistral = MistralAIEmbeddings(
                model="mistral-embed",
                mistral_api_key="my-api-key"
            )
    """

    client: Any  #: :meta private:
    model: str = "mistral-embed"
    mistral_api_key: Optional[str] = None
    MistralException: ClassVar[Type[BaseException]]

    class Config:
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that mistralai library is installed."""

        try:
            from mistralai.client import MistralClient
            from mistralai.exceptions import MistralException
        except ImportError as exc:
            raise ImportError(
                "Could not import mistralai library. "
                "Please install it with `pip install mistralai`"
            ) from exc

        mistral_api_key = values.get("mistral_api_key")
        values["client"] = MistralClient(api_key=mistral_api_key)
        cls.MistralException = MistralException
        return values

    def _embed(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings_batch_response = self.client.embeddings(
                model=self.model,
                input=texts,
            )
            return [list(map(float, embedding_obj.embedding)) for embedding_obj in embeddings_batch_response.data]
        except self.MistralException as e:
            logger.error(f"An error occurred with MistralAI: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        return self._embed([text])[0]