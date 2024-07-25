from typing import Any, Dict, List, Optional

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from requests import RequestException

BAICHUAN_API_URL: str = "http://api.baichuan-ai.com/v1/embeddings"

# BaichuanTextEmbeddings is an embedding model provided by Baichuan Inc. (https://www.baichuan-ai.com/home).
# As of today (Jan 25th, 2024) BaichuanTextEmbeddings ranks #1 in C-MTEB
# (Chinese Multi-Task Embedding Benchmark) leaderboard.
# Leaderboard (Under Overall -> Chinese section): https://huggingface.co/spaces/mteb/leaderboard

# Official Website: https://platform.baichuan-ai.com/docs/text-Embedding
# An API-key is required to use this embedding model. You can get one by registering
# at https://platform.baichuan-ai.com/docs/text-Embedding.
# BaichuanTextEmbeddings support 512 token window and preduces vectors with
# 1024 dimensions.


# NOTE!! BaichuanTextEmbeddings only supports Chinese text embedding.
# Multi-language support is coming soon.
class BaichuanTextEmbeddings(BaseModel, Embeddings):
    """Baichuan Text Embedding models.

    Setup:
        To use, you should set the environment variable ``BAICHUAN_API_KEY`` to
        your API key or pass it as a named parameter to the constructor.

        .. code-block:: bash

            export BAICHUAN_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python

            from langchain_community.embeddings import BaichuanTextEmbeddings

            embeddings = BaichuanTextEmbeddings()

    Embed:
        .. code-block:: python

            # embed the documents
            vectors = embeddings.embed_documents([text1, text2, ...])

            # embed the query
            vectors = embeddings.embed_query(text)
    """  # noqa: E501

    session: Any  #: :meta private:
    model_name: str = Field(default="Baichuan-Text-Embedding", alias="model")
    """The model used to embed the documents."""
    baichuan_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """Automatically inferred from env var `BAICHUAN_API_KEY` if not provided."""
    chunk_size: int = 16
    """Chunk size when multiple texts are input"""

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that auth token exists in environment."""
        try:
            baichuan_api_key = convert_to_secret_str(
                get_from_dict_or_env(values, "baichuan_api_key", "BAICHUAN_API_KEY")
            )
        except ValueError as original_exc:
            try:
                baichuan_api_key = convert_to_secret_str(
                    get_from_dict_or_env(
                        values, "baichuan_auth_token", "BAICHUAN_AUTH_TOKEN"
                    )
                )
            except ValueError:
                raise original_exc
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {baichuan_api_key.get_secret_value()}",
                "Accept-Encoding": "identity",
                "Content-type": "application/json",
            }
        )
        values["session"] = session
        return values

    def _embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Internal method to call Baichuan Embedding API and return embeddings.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of list of floats representing the embeddings, or None if an
            error occurs.
        """
        chunk_texts = [
            texts[i : i + self.chunk_size]
            for i in range(0, len(texts), self.chunk_size)
        ]
        embed_results = []
        for chunk in chunk_texts:
            response = self.session.post(
                BAICHUAN_API_URL, json={"input": chunk, "model": self.model_name}
            )
            # Raise exception if response status code from 400 to 600
            response.raise_for_status()
            # Check if the response status code indicates success
            if response.status_code == 200:
                resp = response.json()
                embeddings = resp.get("data", [])
                # Sort resulting embeddings by index
                sorted_embeddings = sorted(embeddings, key=lambda e: e.get("index", 0))
                # Return just the embeddings
                embed_results.extend(
                    [result.get("embedding", []) for result in sorted_embeddings]
                )
            else:
                # Log error or handle unsuccessful response appropriately
                # Handle 100 <= status_code < 400, not include 200
                raise RequestException(
                    f"Error: Received status code {response.status_code} from "
                    "`BaichuanEmbedding` API"
                )
        return embed_results

    def embed_documents(self, texts: List[str]) -> Optional[List[List[float]]]:  # type: ignore[override]
        """Public method to get embeddings for a list of documents.

        Args:
            texts: The list of texts to embed.

        Returns:
            A list of embeddings, one for each text, or None if an error occurs.
        """
        return self._embed(texts)

    def embed_query(self, text: str) -> Optional[List[float]]:  # type: ignore[override]
        """Public method to get embedding for a single query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text, or None if an error occurs.
        """
        result = self._embed([text])
        return result[0] if result is not None else None
