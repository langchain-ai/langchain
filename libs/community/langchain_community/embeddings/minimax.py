from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init
from pydantic import BaseModel, ConfigDict, Field, SecretStr
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
    """MiniMax embedding model integration.

    Setup:
        To use, you should have the environment variable ``MINIMAX_GROUP_ID`` and
        ``MINIMAX_API_KEY`` set with your API token.

        .. code-block:: bash

            export MINIMAX_API_KEY="your-api-key"
            export MINIMAX_GROUP_ID="your-group-id"

    Key init args — completion params:
        model: Optional[str]
            Name of ZhipuAI model to use.
        api_key: Optional[str]
            Automatically inferred from env var `MINIMAX_GROUP_ID` if not provided.
        group_id: Optional[str]
            Automatically inferred from env var `MINIMAX_GROUP_ID` if not provided.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:

        .. code-block:: python

            from langchain_community.embeddings import MiniMaxEmbeddings

            embed = MiniMaxEmbeddings(
                model="embo-01",
                # api_key="...",
                # group_id="...",
                # other
            )

    Embed single text:
        .. code-block:: python

            input_text = "The meaning of life is 42"
            embed.embed_query(input_text)

        .. code-block:: python

            [0.03016241, 0.03617699, 0.0017198119, -0.002061239, -0.00029994643, -0.0061320597, -0.0043635326, ...]

    Embed multiple text:
        .. code-block:: python

            input_texts = ["This is a test query1.", "This is a test query2."]
            embed.embed_documents(input_texts)

        .. code-block:: python

            [
                [-0.0021588828, -0.007608119, 0.029349545, -0.0038194496, 0.008031177, -0.004529633, -0.020150753, ...],
                [ -0.00023150232, -0.011122423, 0.016930554, 0.0083089275, 0.012633711, 0.019683322, -0.005971041, ...]
            ]
    """  # noqa: E501

    endpoint_url: str = "https://api.minimax.chat/v1/embeddings"
    """Endpoint URL to use."""
    model: str = "embo-01"
    """Embeddings model name to use."""
    embed_type_db: str = "db"
    """For embed_documents"""
    embed_type_query: str = "query"
    """For embed_query"""

    minimax_group_id: Optional[str] = Field(default=None, alias="group_id")
    """Group ID for MiniMax API."""
    minimax_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """API Key for MiniMax API."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that group id and api key exists in environment."""
        minimax_group_id = get_from_dict_or_env(
            values, ["minimax_group_id", "group_id"], "MINIMAX_GROUP_ID"
        )
        minimax_api_key = convert_to_secret_str(
            get_from_dict_or_env(
                values, ["minimax_api_key", "api_key"], "MINIMAX_API_KEY"
            )
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
            "Authorization": f"Bearer {self.minimax_api_key.get_secret_value()}",  # type: ignore[union-attr]
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
