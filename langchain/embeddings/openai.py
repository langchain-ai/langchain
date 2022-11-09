"""Wrapper around OpenAI embedding models."""
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.embeddings.base import Embeddings


class OpenAIEmbeddings(BaseModel, Embeddings):
    """Wrapper around OpenAI embedding models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key or pass it
    as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.embeddings import OpenAIEmbeddings
            openai = OpenAIEmbeddings(model_name="davinci", openai_api_key="my-api-key")
    """

    client: Any  #: :meta private:
    model_name: str = "babbage"
    """Model name to use."""

    openai_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY")

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = values.get("openai_api_key")

        if openai_api_key is None or openai_api_key == "":
            raise ValueError(
                "Did not find OpenAI API key, please add an environment variable"
                " `OPENAI_API_KEY` which contains it, or pass `openai_api_key` as a"
                " named parameter."
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

    def _embedding_func(self, text: str, *, engine: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint."""
        # replace newlines, which can negatively affect performance.
        text = text.replace("\n", " ")
        return self.client.create(input=[text], engine=engine)["data"][0]["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to OpenAI's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        responses = [
            self._embedding_func(text, engine=f"text-search-{self.model_name}-doc-001")
            for text in texts
        ]
        return responses

    def embed_query(self, text: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embedding = self._embedding_func(
            text, engine=f"text-search-{self.model_name}-query-001"
        )
        return embedding
