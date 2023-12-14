from typing import Dict, List, Optional

# TODO: remove ignore once the google package is published with types
import google.generativeai as genai  # type: ignore[import]
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.utils import get_from_dict_or_env

from langchain_google_genai._common import GoogleGenerativeAIError


class GoogleGenerativeAIEmbeddings(BaseModel, Embeddings):
    """`Google Generative AI Embeddings`.

    To use, you must have either:

        1. The ``GOOGLE_API_KEY``` environment variable set with your API key, or
        2. Pass your API key using the google_api_key kwarg to the ChatGoogle
           constructor.

    Example:
        .. code-block:: python

            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            embeddings.embed_query("What's our Q1 revenue?")
    """

    model: str = Field(
        ...,
        description="The name of the embedding model to use. "
        "Example: models/embedding-001",
    )
    task_type: Optional[str] = Field(
        None,
        description="The task type. Valid options include: "
        "task_type_unspecified, retrieval_query, retrieval_document, "
        "semantic_similarity, classification, and clustering",
    )
    google_api_key: Optional[SecretStr] = Field(
        None,
        description="The Google API key to use. If not provided, "
        "the GOOGLE_API_KEY environment variable will be used.",
    )

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validates that the python package exists in environment."""
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_API_KEY"
        )
        if isinstance(google_api_key, SecretStr):
            google_api_key = google_api_key.get_secret_value()
        genai.configure(api_key=google_api_key)
        return values

    def _embed(
        self, texts: List[str], task_type: str, title: Optional[str] = None
    ) -> List[List[float]]:
        task_type = self.task_type or "retrieval_document"
        try:
            result = genai.embed_content(
                model=self.model,
                content=texts,
                task_type=task_type,
                title=title,
            )
        except Exception as e:
            raise GoogleGenerativeAIError(f"Error embedding content: {e}") from e
        return result["embedding"]

    def embed_documents(
        self, texts: List[str], batch_size: int = 5
    ) -> List[List[float]]:
        """Embed a list of strings. Vertex AI currently
        sets a max batch size of 5 strings.

        Args:
            texts: List[str] The list of strings to embed.
            batch_size: [int] The batch size of embeddings to send to the model

        Returns:
            List of embeddings, one for each text.
        """
        task_type = self.task_type or "retrieval_document"
        return self._embed(texts, task_type=task_type)

    def embed_query(self, text: str) -> List[float]:
        """Embed a text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        task_type = self.task_type or "retrieval_query"
        return self._embed([text], task_type=task_type)[0]
