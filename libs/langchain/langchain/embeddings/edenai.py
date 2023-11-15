from typing import Any, Dict, List, Optional

from langchain.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain.schema.embeddings import Embeddings
from langchain.utilities.requests import Requests
from langchain.utils import get_from_dict_or_env


class EdenAiEmbeddings(BaseModel, Embeddings):
    """EdenAI embedding.
    environment variable ``EDENAI_API_KEY`` set with your API key, or pass
    it as a named parameter.
    """

    edenai_api_key: Optional[str] = Field(None, description="EdenAI API Token")

    provider: str = "openai"
    """embedding provider to use (eg: openai,google etc.)"""

    model: Optional[str] = None
    """
    model name for above provider (eg: 'text-davinci-003' for openai)
    available models are shown on https://docs.edenai.co/ under 'available providers'
    """

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

    @staticmethod
    def get_user_agent() -> str:
        from langchain import __version__

        return f"langchain/{__version__}"

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings using EdenAi api."""
        url = "https://api.edenai.run/v2/text/embeddings"

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.edenai_api_key}",
            "User-Agent": self.get_user_agent(),
        }

        payload: Dict[str, Any] = {"texts": texts, "providers": self.provider}

        if self.model is not None:
            payload["settings"] = {self.provider: self.model}

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

        provider_response = temp[self.provider]
        if provider_response.get("status") == "fail":
            err_msg = provider_response.get("error", {}).get("message")
            raise Exception(err_msg)

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
