from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def _create_retry_decorator() -> Callable[[Any], Any]:
    """Returns a tenacity retry decorator."""

    multiplier = 1
    min_seconds = 1
    max_seconds = 4
    max_retries = 6

    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=multiplier, min=min_seconds, max=max_seconds),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def embed_with_retry(embeddings: MiniMaxEmbeddings, *args: Any, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator()

    @retry_decorator
    def _embed_with_retry(*args: Any, **kwargs: Any) -> Any:
        return embeddings.embed(*args, **kwargs)

    return _embed_with_retry(*args, **kwargs)


class MiniMaxEmbeddings(BaseModel, Embeddings):
    """MiniMax's embedding service.

    To use, you should have the environment variable ``MINIMAX_GROUP_ID`` and
    ``MINIMAX_API_KEY`` set with your API token, or pass it as a named parameter to
    the constructor.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import MiniMaxEmbeddings
            embeddings = MiniMaxEmbeddings()

            query_text = "This is a test query."
            query_result = embeddings.embed_query(query_text)

            document_text = "This is a test document."
            document_result = embeddings.embed_documents([document_text])

    """

    endpoint_url: str = "https://api.minimax.chat/v1/embeddings"
    """Endpoint URL to use."""
    model: str = "embo-01"
    """Embeddings model name to use."""
    embed_type_db: str = "db"
    """For embed_documents"""
    embed_type_query: str = "query"
    """For embed_query"""

    minimax_group_id: Optional[str] = None
    """Group ID for MiniMax API."""
    minimax_api_key: Optional[SecretStr] = None
    """API Key for MiniMax API."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that group id and api key exists in environment."""
        minimax_group_id = get_from_dict_or_env(
            values, "minimax_group_id", "MINIMAX_GROUP_ID"
        )
        minimax_api_key = convert_to_secret_str(
            get_from_dict_or_env(values, "minimax_api_key", "MINIMAX_API_KEY")
        )
        values["minimax_group_id"] = minimax_group_id
        values["minimax_api_key"] = minimax_api_key
        return values

    def embed(
        self,
        texts: List[str],
        embed_type: str,
    ) -> List[List[float]]:
        payload = {
            "model": self.model,
            "type": embed_type,
            "texts": texts,
        }

        # HTTP headers for authorization
        headers = {
            "Authorization": f"Bearer {self.minimax_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

        params = {
            "GroupId": self.minimax_group_id,
        }

        # send request
        response = requests.post(
            self.endpoint_url, params=params, headers=headers, json=payload
        )
        parsed_response = response.json()

        # check for errors
        if parsed_response["base_resp"]["status_code"] != 0:
            raise ValueError(
                f"MiniMax API returned an error: {parsed_response['base_resp']}"
            )

        embeddings = parsed_response["vectors"]

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using a MiniMax embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = embed_with_retry(self, texts=texts, embed_type=self.embed_type_db)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using a MiniMax embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embeddings = embed_with_retry(
            self, texts=[text], embed_type=self.embed_type_query
        )
        return embeddings[0]
