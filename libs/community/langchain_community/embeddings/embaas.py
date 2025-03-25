from typing import Any, Dict, List, Mapping, Optional

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init
from pydantic import BaseModel, ConfigDict, SecretStr
from requests.adapters import HTTPAdapter, Retry
from typing_extensions import NotRequired, TypedDict

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

            # initialize with default model and instruction
            from langchain_community.embeddings import EmbaasEmbeddings
            emb = EmbaasEmbeddings()

            # initialize with custom model and instruction
            from langchain_community.embeddings import EmbaasEmbeddings
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
    embaas_api_key: Optional[SecretStr] = None
    """max number of retries for requests"""
    max_retries: Optional[int] = 3
    """request timeout in seconds"""
    timeout: Optional[int] = 30

    model_config = ConfigDict(
        extra="forbid",
    )

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        embaas_api_key = convert_to_secret_str(
            get_from_dict_or_env(values, "embaas_api_key", "EMBAAS_API_KEY")
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
            "Authorization": f"Bearer {self.embaas_api_key.get_secret_value()}",  # type: ignore[union-attr]
            "Content-Type": "application/json",
        }

        session = requests.Session()
        retries = Retry(
            total=self.max_retries,
            backoff_factor=0.5,
            allowed_methods=["POST"],
            raise_on_status=True,
        )

        session.mount("http://", HTTPAdapter(max_retries=retries))
        session.mount("https://", HTTPAdapter(max_retries=retries))
        response = session.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

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
