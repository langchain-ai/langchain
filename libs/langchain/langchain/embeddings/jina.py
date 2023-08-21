import os
from typing import Any, Dict, List, Optional

import requests

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.utils import get_from_dict_or_env


class JinaEmbeddings(BaseModel, Embeddings):
    """Jina embedding models."""

    client: Any  #: :meta private:

    model_name: str = "ViT-B-32::openai"
    """Model name to use."""

    jina_auth_token: Optional[str] = None
    jina_api_url: str = "https://api.clip.jina.ai/api/v1/models/"
    request_headers: Optional[dict] = None

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that auth token exists in environment."""
        # Set Auth
        jina_auth_token = get_from_dict_or_env(
            values, "jina_auth_token", "JINA_AUTH_TOKEN"
        )
        values["jina_auth_token"] = jina_auth_token
        values["request_headers"] = (("authorization", jina_auth_token),)

        # Test that package is installed
        try:
            import jina
        except ImportError:
            raise ImportError(
                "Could not import `jina` python package. "
                "Please install it with `pip install jina`."
            )

        # Setup client
        jina_api_url = os.environ.get("JINA_API_URL", values["jina_api_url"])
        model_name = values["model_name"]
        try:
            resp = requests.get(
                jina_api_url + f"?model_name={model_name}",
                headers={"Authorization": jina_auth_token},
            )

            if resp.status_code == 401:
                raise ValueError(
                    "The given Jina auth token is invalid. "
                    "Please check your Jina auth token."
                )
            elif resp.status_code == 404:
                raise ValueError(
                    f"The given model name `{model_name}` is not valid. "
                    f"Please go to https://cloud.jina.ai/user/inference "
                    f"and create a model with the given model name."
                )
            resp.raise_for_status()

            endpoint = resp.json()["endpoints"]["grpc"]
            values["client"] = jina.Client(host=endpoint)
        except requests.exceptions.HTTPError as err:
            raise ValueError(f"Error: {err!r}")
        return values

    def _post(self, docs: List[Any], **kwargs: Any) -> Any:
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
