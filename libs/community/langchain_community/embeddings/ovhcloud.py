import json
import logging
import time
from typing import Any, List

import requests
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class OVHCloudEmbeddings(BaseModel, Embeddings):
    """
    OVHcloud AI Endpoints Embeddings.
    """

    """ OVHcloud AI Endpoints Access Token"""
    access_token: str = ""

    """ OVHcloud AI Endpoints model name for embeddings generation"""
    model_name: str = ""

    """ OVHcloud AI Endpoints region"""
    region: str = "kepler"

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if self.access_token == "":
            raise ValueError("Access token is required for OVHCloud embeddings.")
        if self.model_name == "":
            raise ValueError("Model name is required for OVHCloud embeddings.")
        if self.region == "":
            raise ValueError("Region is required for OVHCloud embeddings.")

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings from OVHCLOUD AIE.
        Args:
            text (str): The text to embed.
        Returns:
            List[float]: Embeddings for the text.
        """

        return self._send_request_to_ai_endpoints("text/plain", text, "text2vec")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.
        Args:
           texts (List[str]): The list of texts to embed.

        Returns:
           List[List[float]]: List of embeddings, one for each input text.

        """

        return self._send_request_to_ai_endpoints(
            "application/json", json.dumps(texts), "batch_text2vec"
        )

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.
        Args:
            text (str): The text to embed.
        Returns:
            List[float]: Embeddings for the text.
        """
        return self._generate_embedding(text)

    def _send_request_to_ai_endpoints(
        self, contentType: str, payload: str, route: str
    ) -> Any:
        """Send a HTTPS request to OVHcloud AI Endpoints
        Args:
            contentType (str): The content type of the request, application/json or text/plain.
            payload (str): The payload of the request.
            route (str): The route of the request, batch_text2vec or text2vec.
        """  # noqa: E501
        headers = {
            "content-type": contentType,
            "Authorization": f"Bearer {self.access_token}",
        }

        session = requests.session()
        while True:
            response = session.post(
                (
                    f"https://{self.model_name}.endpoints.{self.region}"
                    f".ai.cloud.ovh.net/api/{route}"
                ),
                headers=headers,
                data=payload,
            )
            if response.status_code != 200:
                if response.status_code == 429:
                    """Rate limit exceeded, wait for reset"""
                    reset_time = int(response.headers.get("RateLimit-Reset", 0))
                    logger.info("Rate limit exceeded. Waiting %d seconds.", reset_time)
                    if reset_time > 0:
                        time.sleep(reset_time)
                        continue
                    else:
                        """Rate limit reset time has passed, retry immediately"""
                        continue
                if response.status_code == 401:
                    """ Unauthorized, retry with new token """
                    raise ValueError("Unauthorized, retry with new token")
                """ Handle other non-200 status codes """
                raise ValueError(
                    "Request failed with status code: {status_code}, {text}".format(
                        status_code=response.status_code, text=response.text
                    )
                )
            return response.json()
