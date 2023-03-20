"""Wrapper around Jina embedding models."""

from typing import Any, Dict, List, Optional

import jina
from pydantic import BaseModel, root_validator

from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env


class JinaEmbeddings(BaseModel, Embeddings):
    client: Any  #: :meta private:

    """Model name to use."""
    model_name: str = "ViT-B-32::openai"

    jina_auth_token: Optional[str] = None
    jina_api_url: str = "https://api.clip.jina.ai/api/v1/models/"
    request_headers: Optional[dict] = None

    def __init__(self, **kwargs: Any):
        """Initialize the jina embeddings."""
        super().__init__(**kwargs)

        import os

        import requests

        jina_api_url = os.environ.get("JINA_API_URL", self.jina_api_url)

        try:
            resp = requests.get(
                jina_api_url + f"?model_name={self.model_name}",
                headers={"Authorization": self.jina_auth_token},
            )

            if resp.status_code == 401:
                raise ValueError(
                    "The given Jina auth token is invalid. Please check your Jina auth token."
                )
            elif resp.status_code == 404:
                raise ValueError(
                    f"The given model name `{self.model_name}` is not valid. "
                    f"Please go to https://cloud.jina.ai/user/inference "
                    f"and create a model with the given model name."
                )
            resp.raise_for_status()

            endpoint = resp.json()["endpoints"]["grpc"]
            self.client = jina.Client(host=endpoint)
        except requests.exceptions.HTTPError as err:
            raise ValueError(f"Error: {err!r}")

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that auth token exists in environment."""
        jina_auth_token = get_from_dict_or_env(
            values, "jina_auth_token", "JINA_AUTH_TOKEN"
        )
        try:
            import jina

            values["jina_auth_token"] = jina_auth_token
            values["request_headers"] = (("authorization", jina_auth_token),)
        except ImportError:
            raise ValueError(
                "Could not import `jina` python package. "
                "Please it install it with `pip install jina`."
            )
        return values

    def _post(self, docs, **kwargs):
        payload = dict(inputs=docs, metadata=self.request_headers, **kwargs)
        return self.client.post(on="/encode", **payload)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Jina's embedding endpoint.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        from docarray import Document, DocumentArray

        embeddings = self._post(
            docs=DocumentArray([Document(text=t) for t in texts])
        ).embeddings
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Call out to Jina's embedding endpoint.
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        from docarray import Document, DocumentArray

        embedding = self._post(docs=DocumentArray([Document(text=text)])).embeddings[0]
        return list(map(float, embedding))
