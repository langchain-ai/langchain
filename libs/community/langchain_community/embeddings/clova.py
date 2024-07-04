from __future__ import annotations

from typing import Dict, List, Optional, cast

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env


class ClovaEmbeddings(BaseModel, Embeddings):
    """
    Clova's embedding service.

    To use this service,

    you should have the following environment variables
    set with your API tokens and application ID,
    or pass them as named parameters to the constructor:

    - ``CLOVA_EMB_API_KEY``: API key for accessing Clova's embedding service.
    - ``CLOVA_EMB_APIGW_API_KEY``: API gateway key for enhanced security.
    - ``CLOVA_EMB_APP_ID``: Application ID for identifying your application.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import ClovaEmbeddings
            embeddings = ClovaEmbeddings(
                clova_emb_api_key='your_clova_emb_api_key',
                clova_emb_apigw_api_key='your_clova_emb_apigw_api_key',
                app_id='your_app_id'
            )

            query_text = "This is a test query."
            query_result = embeddings.embed_query(query_text)

            document_text = "This is a test document."
            document_result = embeddings.embed_documents([document_text])

    """

    endpoint_url: str = (
        "https://clovastudio.apigw.ntruss.com/testapp/v1/api-tools/embedding"
    )
    """Endpoint URL to use."""
    model: str = "clir-emb-dolphin"
    """Embedding model name to use."""
    clova_emb_api_key: Optional[SecretStr] = None
    """API key for accessing Clova's embedding service."""
    clova_emb_apigw_api_key: Optional[SecretStr] = None
    """API gateway key for enhanced security."""
    app_id: Optional[SecretStr] = None
    """Application ID for identifying your application."""

    class Config:
        extra = Extra.forbid

    @root_validator(pre=True, allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate api key exists in environment."""
        values["clova_emb_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "clova_emb_api_key", "CLOVA_EMB_API_KEY")
        )
        values["clova_emb_apigw_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values, "clova_emb_apigw_api_key", "CLOVA_EMB_APIGW_API_KEY"
            )
        )
        values["app_id"] = convert_to_secret_str(
            get_from_dict_or_env(values, "app_id", "CLOVA_EMB_APP_ID")
        )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts and return their embeddings.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = []
        for text in texts:
            embeddings.append(self._embed_text(text))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text and return its embedding.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self._embed_text(text)

    def _embed_text(self, text: str) -> List[float]:
        """
        Internal method to call the embedding API and handle the response.
        """
        payload = {"text": text}

        # HTTP headers for authorization
        headers = {
            "X-NCP-CLOVASTUDIO-API-KEY": cast(
                SecretStr, self.clova_emb_api_key
            ).get_secret_value(),
            "X-NCP-APIGW-API-KEY": cast(
                SecretStr, self.clova_emb_apigw_api_key
            ).get_secret_value(),
            "Content-Type": "application/json",
        }

        # send request
        app_id = cast(SecretStr, self.app_id).get_secret_value()
        response = requests.post(
            f"{self.endpoint_url}/{self.model}/{app_id}",
            headers=headers,
            json=payload,
        )

        # check for errors
        if response.status_code == 200:
            response_data = response.json()
            if "result" in response_data and "embedding" in response_data["result"]:
                return response_data["result"]["embedding"]
        raise ValueError(
            f"API request failed with status {response.status_code}: {response.text}"
        )
