from typing import Any, Dict, List, Mapping, Optional

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env

DEFAULT_MODEL_ID = "sentence-transformers/clip-ViT-B-32"


class DeepInfraEmbeddings(BaseModel, Embeddings):
    """Deep Infra's embedding inference service.

    To use, you should have the
    environment variable ``DEEPINFRA_API_TOKEN`` set with your API token, or pass
    it as a named parameter to the constructor.
    There are multiple embeddings models available,
    see https://deepinfra.com/models?type=embeddings.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import DeepInfraEmbeddings
            deepinfra_emb = DeepInfraEmbeddings(
                model_id="sentence-transformers/clip-ViT-B-32",
                deepinfra_api_token="my-api-key"
            )
            r1 = deepinfra_emb.embed_documents(
                [
                    "Alpha is the first letter of Greek alphabet",
                    "Beta is the second letter of Greek alphabet",
                ]
            )
            r2 = deepinfra_emb.embed_query(
                "What is the second letter of Greek alphabet"
            )

    """

    model_id: str = DEFAULT_MODEL_ID
    """Embeddings model to use."""
    normalize: bool = False
    """whether to normalize the computed embeddings"""
    embed_instruction: str = "passage: "
    """Instruction used to embed documents."""
    query_instruction: str = "query: "
    """Instruction used to embed the query."""
    model_kwargs: Optional[dict] = None
    """Other model keyword args"""

    deepinfra_api_token: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        deepinfra_api_token = get_from_dict_or_env(
            values, "deepinfra_api_token", "DEEPINFRA_API_TOKEN"
        )
        values["deepinfra_api_token"] = deepinfra_api_token
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_id": self.model_id}

    def _embed(self, input: List[str]) -> List[List[float]]:
        _model_kwargs = self.model_kwargs or {}
        # HTTP headers for authorization
        headers = {
            "Authorization": f"bearer {self.deepinfra_api_token}",
            "Content-Type": "application/json",
        }
        # send request
        try:
            res = requests.post(
                f"https://api.deepinfra.com/v1/inference/{self.model_id}",
                headers=headers,
                json={"inputs": input, "normalize": self.normalize, **_model_kwargs},
            )
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        if res.status_code != 200:
            raise ValueError(
                "Error raised by inference API HTTP code: %s, %s"
                % (res.status_code, res.text)
            )
        try:
            t = res.json()
            embeddings = t["embeddings"]
        except requests.exceptions.JSONDecodeError as e:
            raise ValueError(
                f"Error raised by inference API: {e}.\nResponse: {res.text}"
            )

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using a Deep Infra deployed embedding model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        instruction_pairs = [f"{self.embed_instruction}{text}" for text in texts]
        embeddings = self._embed(instruction_pairs)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using a Deep Infra deployed embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        instruction_pair = f"{self.query_instruction}{text}"
        embedding = self._embed([instruction_pair])[0]
        return embedding
