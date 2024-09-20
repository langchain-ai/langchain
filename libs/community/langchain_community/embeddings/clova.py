from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, cast

import requests
from pydantic import BaseModel, ConfigDict, SecretStr, model_validator

from langchain_core.embeddings import Embeddings
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

class BaseClovaEmbeddings(BaseModel, Embeddings, ABC):
    """
    Base class for Clova's embedding services.

    To use this service, you should have the following environment variables
    set with your API tokens and application ID,
    or pass them as named parameters to the constructor:

    - ``CLOVA_EMB_API_KEY``: API key for accessing Clova's embedding service.
    - ``CLOVA_EMB_APIGW_API_KEY``: API gateway key for enhanced security.
    - ``CLOVA_EMB_APP_ID_V1``: Application ID for V1 models.
    - ``CLOVA_EMB_APP_ID_V2``: Application ID for V2 model. 
    """

    endpoint_url: str
    """Endpoint URL to use."""
    clova_emb_api_key: SecretStr
    """API key for accessing Clova's embedding service."""
    clova_emb_apigw_api_key: SecretStr
    """API gateway key for enhanced security."""
    app_id: SecretStr
    """Application ID for identifying your application."""
    model: Optional[str] = None
    """Embedding model name to use (optional)."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that API keys and app ID are set."""
        values["clova_emb_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "clova_emb_api_key", "CLOVA_EMB_API_KEY")
        )
        values["clova_emb_apigw_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values, "clova_emb_apigw_api_key", "CLOVA_EMB_APIGW_API_KEY"
            )
        )
        values["app_id"] = convert_to_secret_str(
            get_from_dict_or_env(values, "app_id", cls.get_app_id_env_var())
        )
        if cls.requires_model():
            values["model"] = get_from_dict_or_env(
                values, "model", cls.get_model_env_var(), default="clir-emb-dolphin"
            )
        return values

    @classmethod
    @abstractmethod
    def requires_model(cls) -> bool:
        """Indicates whether the model parameter is required."""
        pass

    @classmethod
    @abstractmethod
    def get_app_id_env_var(cls) -> str:
        """Provides the environment variable name for the app ID."""
        pass

    @classmethod
    def get_model_env_var(cls) -> str:
        """Provides the environment variable name for the model."""
        return "CLOVA_EMB_MODEL"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts and return their embeddings.

        Args:
            texts: List of texts to embed.

        Returns:
            A list of embeddings, one for each text.

        Raises:
            ValueError: If embeddings cannot be retrieved.
        """
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text and return its embedding.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the text embedding.

        Raises:
            ValueError: If embedding cannot be retrieved.
        """
        return self._embed_text(text)

    def _embed_text(self, text: str) -> List[float]:
        """
        Internal method to call the embedding API and handle the response.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding.

        Raises:
            ValueError: If the API request fails or the response is invalid.
        """
        payload = {"text": text}
        headers = {
            "X-NCP-CLOVASTUDIO-API-KEY": self.clova_emb_api_key.get_secret_value(),
            "X-NCP-APIGW-API-KEY": self.clova_emb_apigw_api_key.get_secret_value(),
            "Content-Type": "application/json",
        }

        response_data = self._send_request(headers, payload)
        try:
            return response_data["result"]["embedding"]
        except KeyError:
            raise ValueError(
                "Invalid response format: 'embedding' not found in response."
            )

    @abstractmethod
    def _send_request(self, headers: Dict[str, str], payload: Dict[str, str]) -> Dict:
        """
        Send the request to the embedding API.

        Args:
            headers: HTTP headers for authentication.
            payload: JSON payload for the request.

        Returns:
            The API response as a dictionary.

        Raises:
            ConnectionError: If the API request fails.
        """
        pass

class ClovaEmbeddingsV1(BaseClovaEmbeddings):
    """
    Clova's embedding service V1.
    """

    endpoint_url: str = (
        "https://clovastudio.apigw.ntruss.com/testapp/v1/api-tools/embedding"
    )
    """Endpoint URL to use."""

    @classmethod
    def get_app_id_env_var(cls) -> str:
        """Provides the environment variable name for the app ID."""
        return "CLOVA_EMB_APP_ID_V1"

    @classmethod
    def requires_model(cls) -> bool:
        """Indicates whether the model parameter is required."""
        return True

    def _send_request(self, headers: Dict[str, str], payload: Dict[str, str]) -> Dict:
        request_url = f"{self.endpoint_url}/{self.model}/{self.app_id.get_secret_value()}"
        try:
            response = requests.post(
                request_url, headers=headers, json=payload, timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Error connecting to Clova API: {e}") from e

class ClovaEmbeddingsV2(BaseClovaEmbeddings):
    """
    Clova's embedding service V2.
    """

    endpoint_url: str = (
        "https://clovastudio.apigw.ntruss.com/testapp/v1/api-tools/embedding/v2"
    )
    """Endpoint URL to use."""

    @classmethod
    def get_app_id_env_var(cls) -> str:
        """Provides the environment variable name for the app ID."""
        return "CLOVA_EMB_APP_ID_V2"

    @classmethod
    def requires_model(cls) -> bool:
        """Indicates whether the model parameter is required."""
        return False

    def _send_request(self, headers: Dict[str, str], payload: Dict[str, str]) -> Dict:
        request_url = f"{self.endpoint_url}/{self.app_id.get_secret_value()}"
        try:
            response = requests.post(
                request_url, headers=headers, json=payload, timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Error connecting to Clova API: {e}") from e