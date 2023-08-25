from typing import Dict, List, Optional

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain.requests import Requests
from langchain.utils import get_from_dict_or_env


class EdenAiEmbeddings(BaseModel, Embeddings):
    """EdenAI embedding.
    environment variable ``EDENAI_API_KEY`` set with your API key, or pass
    it as a named parameter.
    """

    edenai_api_key: Optional[str] = Field(None, description="EdenAI API Token")

    provider: Optional[str] = "openai"
    """embedding provider to use (eg: openai,google etc.)"""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        values["edenai_api_key"] = get_from_dict_or_env(
            values, "edenai_api_key", "EDENAI_API_KEY"
        )
        return values

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings using EdenAi api."""
        url = "https://api.edenai.run/v2/text/embeddings"

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.edenai_api_key}",
        }

        payload = {"texts": texts, "providers": self.provider}
        request = Requests(headers=headers)
        response = request.post(url=url, data=payload)
        if response.status_code >= 500:
            raise Exception(f"EdenAI Server: Error {response.status_code}")
        elif response.status_code >= 400:
            raise ValueError(f"EdenAI received an invalid payload: {response.text}")
        elif response.status_code != 200:
            raise Exception(
                f"EdenAI returned an unexpected response with status "
                f"{response.status_code}: {response.text}"
            )

        temp = response.json()

        embeddings = []
        for embed_item in temp[self.provider]["items"]:
            embedding = embed_item["embedding"]

            embeddings.append(embedding)

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using EdenAI.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        return self._generate_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using EdenAI.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self._generate_embeddings([text])[0]
