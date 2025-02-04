"""AiBrary embeddings wrapper."""

from __future__ import annotations

from typing import Dict, Optional

from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init
from pydantic import Field, SecretStr

from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.utils.openai import is_openai_v1
from typing import Any, Dict, List, Mapping, Optional

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env, pre_init
from pydantic import BaseModel, ConfigDict


DEFAULT_API_BASE = "https://api.aibrary.dev/v0"
DEFAULT_MODEL = "text-embedding-ada-002"


class AiBraryEmbeddings(OpenAIEmbeddings):
    """AiBrary Compute Service embedding models.

    See https://www.aibrary.dev/ for information about AiBrary.

    To use, you should have the ``openai`` python package installed and the
    environment variable ``AIBRARY_API_KEY`` set with your API token.
    Alternatively, you can use the aibrary_api_key keyword argument.
    Example:
        .. code-block:: python

            from langchain_community.embeddings import AiBraryEmbeddings
            aibrary_emb = AiBraryEmbeddings(
                model="sentence-transformers/clip-ViT-B-32",
                aibrary_api_key="my-api-key"
            )
            r1 = aibrary_emb.embed_documents(
                [
                    "Alpha is the first letter of Greek alphabet",
                    "Beta is the second letter of Greek alphabet",
                ]
            )
            r2 = aibrary_emb.embed_query(
                "What is the second letter of Greek alphabet"
            )

    """

    model: str = Field(default=DEFAULT_MODEL)
    """Other model keyword args"""
    aibrary_api_key: Optional[str] = None
    """API token for Deep Infra. If not provided, the token is 
    fetched from the environment variable 'AIBRARY_API_KEY'."""
    aibrary_api_base: str = Field(default=DEFAULT_API_BASE)

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        aibrary_api_key = get_from_dict_or_env(
            values, "aibrary_api_key", "AIBRARY_API_KEY"
        )

        aibrary_api_base = get_from_dict_or_env(
            values,
            "aibrary_api_base",
            "AIBRARY_API_BASE",
            default=DEFAULT_API_BASE,
        )
        values["aibrary_api_base"] = aibrary_api_base
        values["aibrary_api_key"] = aibrary_api_key
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model": self.model}

    def _embed(self, input: List[str]) -> List[List[float]]:
        _model_kwargs = self.model_kwargs or {}
        # HTTP headers for authorization
        headers = {
            "Authorization": f"bearer {self.aibrary_api_key}",
            "Content-Type": "application/json",
        }
        # send request
        try:
            res = requests.post(
                f"{self.aibrary_api_base}/embeddings",
                headers=headers,
                json={
                    "input": input,
                    "model": self.model,
                    "encoding_format": "float",
                },
            )
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        if res.status_code != 200:
            raise ValueError(
                "Error raised by inference API HTTP code: %s, %s"
                % (res.status_code, res.text)
            )
        try:
            t = res.json()
            embeddings = [item["embedding"] for item in t["data"]]
        except requests.exceptions.JSONDecodeError as e:
            raise ValueError(
                f"Error raised by inference API: {e}.\nResponse: {res.text}"
            )

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using a Deep Infra deployed embedding model.
        For larger batches, the input list of texts is chunked into smaller
        batches to avoid exceeding the maximum request size.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        embeddings = []
        for text in texts:
            embeddings += self._embed(text)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using a Deep Infra deployed embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embedding = self._embed([text])[0]
        return embedding
