from typing import Any, Dict, List, Optional

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

JINA_API_URL: str = "https://api.jina.ai/v1/embeddings"


class JinaEmbeddings(BaseModel, Embeddings):
    """Jina embedding models."""

    session: Any  #: :meta private:
    model_name: str = "jina-embeddings-v2-base-en"
    jina_api_key: Optional[SecretStr] = None

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that auth token exists in environment."""
        try:
            jina_api_key = convert_to_secret_str(
                get_from_dict_or_env(values, "jina_api_key", "JINA_API_KEY")
            )
        except ValueError as original_exc:
            try:
                jina_api_key = convert_to_secret_str(
                    get_from_dict_or_env(values, "jina_auth_token", "JINA_AUTH_TOKEN")
                )
            except ValueError:
                raise original_exc
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {jina_api_key.get_secret_value()}",
                "Accept-Encoding": "identity",
                "Content-type": "application/json",
            }
        )
        values["session"] = session
        return values

    def _embed(self, texts: List[str]) -> List[List[float]]:
        # Call Jina AI Embedding API
        resp = self.session.post(  # type: ignore
            JINA_API_URL, json={"input": texts, "model": self.model_name}
        ).json()
        if "data" not in resp:
            raise RuntimeError(resp["detail"])

        embeddings = resp["data"]

        # Sort resulting embeddings by index
        sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])  # type: ignore

        # Return just the embeddings
        return [result["embedding"] for result in sorted_embeddings]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Jina's embedding endpoint.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Call out to Jina's embedding endpoint.
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        return self._embed([text])[0]
