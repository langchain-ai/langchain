import logging
import time
from typing import Any, List, Optional

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra

logger = logging.getLogger(__name__)


class OVHCloudEmbeddings(BaseModel, Embeddings):
    """
    Usage:
        OVH_AI_ENDPOINTS_ACCESS_TOKEN="your-token" python3 langchain_embedding.py
    NB: Make sure you are using a valid token.
    In the contrary, document indexing will be long due to rate-limiting.
    """

    """ OVHcloud AI Endpoints Access Token"""
    access_token: Optional[str] = None

    """ OVHcloud AI Endpoints model name for embeddings generation"""
    model_name: str = ""

    """ OVHcloud AI Endpoints region"""
    region: str = "kepler"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if self.access_token is None:
            logger.warning(
                "No access token provided indexing will be slow due to rate limiting."
            )
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
        headers = {
            "content-type": "text/plain",
            "Authorization": f"Bearer {self.access_token}",
        }

        session = requests.session()
        while True:
            response = session.post(
                f"https://{self.model_name}.endpoints.{self.region}.ai.cloud.ovh.net/api/text2vec",
                headers=headers,
                data=text,
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

                """ Handle other non-200 status codes """
                raise ValueError(
                    "Request failed with status code: {status_code}, {text}".format(
                        status_code=response.status_code, text=response.text
                    )
                )
            return response.json()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Create a retry decorator for PremAIEmbeddings.
        Args:
           texts (List[str]): The list of texts to embed.

        Returns:
           List[List[float]]: List of embeddings, one for each input text.
        """
        return [self._generate_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.
        Args:
            text (str): The text to embed.
        Returns:
            List[float]: Embeddings for the text.
        """
        return self._generate_embedding(text)
