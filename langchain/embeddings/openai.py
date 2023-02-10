"""Wrapper around OpenAI embedding models."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env


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
    openai_api_key: Optional[str] = None

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

    def _embedding_func(self, text: str, *, engine: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint."""
        # replace newlines, which can negatively affect performance.
        text = text.replace("\n", " ")
        return self.client.create(input=[text], engine=engine)["data"][0]["embedding"]

    def embed_documents(
        self, texts: List[str], chunk_size: int = 1000
    ) -> List[List[float]]:
        """Call out to OpenAI's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.
            chunk_size: The maximum number of texts to send to OpenAI at once
                (max 1000).

        Returns:
            List of embeddings, one for each text.
        """
        # handle large batches of texts
        results = []
        for i in range(0, len(texts), chunk_size):
            response = self.client.create(
                input=texts[i : i + chunk_size], engine=self.document_model_name
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
