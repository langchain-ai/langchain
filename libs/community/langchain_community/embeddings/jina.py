import base64
from os.path import exists
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

JINA_API_URL: str = "https://api.jina.ai/v1/embeddings"


def is_local(url: str) -> bool:
    url_parsed = urlparse(url)
    if url_parsed.scheme in ("file", ""):  # Possibly a local file
        return exists(url_parsed.path)
    return False


def get_bytes_str(file_path: str) -> str:
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


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

    def _embed(self, input: Any) -> List[List[float]]:
        # Call Jina AI Embedding API
        resp = self.session.post(  # type: ignore
            JINA_API_URL, json={"input": input, "model": self.model_name}
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

    def embed_images(self, uris: List[str]) -> List[List[float]]:
        """Call out to Jina's image embedding endpoint.
        Args:
            uris: The list of uris to embed.
        Returns:
            List of embeddings, one for each text.
        """
        input = []
        for uri in uris:
            if is_local(uri):
                input.append({"bytes": get_bytes_str(uri)})
            else:
                input.append({"url": uri})
        return self._embed(input)
