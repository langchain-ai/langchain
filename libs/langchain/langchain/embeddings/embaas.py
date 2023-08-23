from typing import Any, Dict, List, Mapping, Optional

import requests
from typing_extensions import NotRequired, TypedDict

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.utils import get_from_dict_or_env

# Currently supported maximum batch size for embedding requests
MAX_BATCH_SIZE = 256
EMBAAS_API_URL = "https://api.embaas.io/v1/embeddings/"


class EmbaasEmbeddingsPayload(TypedDict):
    """Payload for the Embaas embeddings API."""

    model: str
    texts: List[str]
    instruction: NotRequired[str]


class EmbaasEmbeddings(BaseModel, Embeddings):
    """Embaas's embedding service.

    To use, you should have the
    environment variable ``EMBAAS_API_KEY`` set with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            # Initialise with default model and instruction
            from langchain.embeddings import EmbaasEmbeddings
            emb = EmbaasEmbeddings()

            # Initialise with custom model and instruction
            from langchain.embeddings import EmbaasEmbeddings
            emb_model = "instructor-large"
            emb_inst = "Represent the Wikipedia document for retrieval"
            emb = EmbaasEmbeddings(
                model=emb_model,
                instruction=emb_inst
            )
    """

    model: str = "e5-large-v2"
    """The model used for embeddings."""
    instruction: Optional[str] = None
    """Instruction used for domain-specific embeddings."""
    api_url: str = EMBAAS_API_URL
    """The URL for the embaas embeddings API."""
    embaas_api_key: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        embaas_api_key = get_from_dict_or_env(
            values, "embaas_api_key", "EMBAAS_API_KEY"
        )
        values["embaas_api_key"] = embaas_api_key
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying params."""
        return {"model": self.model, "instruction": self.instruction}

    def _generate_payload(self, texts: List[str]) -> EmbaasEmbeddingsPayload:
        """Generates payload for the API request."""
        payload = EmbaasEmbeddingsPayload(texts=texts, model=self.model)
        if self.instruction:
            payload["instruction"] = self.instruction
        return payload

    def _handle_request(self, payload: EmbaasEmbeddingsPayload) -> List[List[float]]:
        """Sends a request to the Embaas API and handles the response."""
        headers = {
            "Authorization": f"Bearer {self.embaas_api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()

        parsed_response = response.json()
        embeddings = [item["embedding"] for item in parsed_response["data"]]

        return embeddings

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using the Embaas API."""
        payload = self._generate_payload(texts)
        try:
            return self._handle_request(payload)
        except requests.exceptions.RequestException as e:
            if e.response is None or not e.response.text:
                raise ValueError(f"Error raised by embaas embeddings API: {e}")

            parsed_response = e.response.json()
            if "message" in parsed_response:
                raise ValueError(
                    "Validation Error raised by embaas embeddings API:"
                    f"{parsed_response['message']}"
                )
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts.

        Args:
            texts: The list of texts to get embeddings for.

        Returns:
            List of embeddings, one for each text.
        """
        batches = [
            texts[i : i + MAX_BATCH_SIZE] for i in range(0, len(texts), MAX_BATCH_SIZE)
        ]
        embeddings = [self._generate_embeddings(batch) for batch in batches]
        # flatten the list of lists into a single list
        return [embedding for batch in embeddings for embedding in batch]

    def embed_query(self, text: str) -> List[float]:
        """Get embeddings for a single text.

        Args:
            text: The text to get embeddings for.

        Returns:
            List of embeddings.
        """
        return self.embed_documents([text])[0]
