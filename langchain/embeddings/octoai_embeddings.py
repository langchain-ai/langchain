"""Module providing a wrapper around OctoAI Compute Service embedding models."""

from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, Extra, Field, root_validator

from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env

DEFAULT_EMBED_INSTRUCTION = "Represent this input: "
DEFAULT_QUERY_INSTRUCTION = "Represent the question for retrieving similar documents: "


class OctoAIEmbeddings(BaseModel, Embeddings):
    """Wrapper around OctoAI Compute Service embedding models.

    The environment variable ``OCTOAI_API_TOKEN`` should be set
    with your API token, or it can be passed
    as a named parameter to the constructor.
    """

    endpoint_url: Optional[str] = Field(None, description="Endpoint URL to use.")
    model_kwargs: Optional[dict] = Field(
        None, description="Keyword arguments to pass to the model."
    )
    octoai_api_token: Optional[str] = Field(None, description="OCTOAI API Token")
    embed_instruction: str = Field(
        DEFAULT_EMBED_INSTRUCTION,
        description="Instruction to use for embedding documents.",
    )
    query_instruction: str = Field(
        DEFAULT_QUERY_INSTRUCTION, description="Instruction to use for embedding query."
    )

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Ensure that the API key and python package exist in environment."""
        values["octoai_api_token"] = get_from_dict_or_env(
            values, "octoai_api_token", "OCTOAI_API_TOKEN"
        )
        values["endpoint_url"] = get_from_dict_or_env(
            values, "endpoint_url", "ENDPOINT_URL"
        )
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Return the identifying parameters."""
        return {
            "endpoint_url": self.endpoint_url,
            "model_kwargs": self.model_kwargs or {},
        }

    def _compute_embeddings(
        self, texts: List[str], instruction: str
    ) -> List[List[float]]:
        """Compute embeddings using an OctoAI instruct model."""
        from octoai import client

        embeddings = []
        octoai_client = client.Client(token=self.octoai_api_token)

        for text in texts:
            parameter_payload = {
                "sentence": str([text]),  # for item in text]),
                "instruction": str([instruction]),  # for item in text]),
                "parameters": self.model_kwargs or {},
            }

            try:
                resp_json = octoai_client.infer(self.endpoint_url, parameter_payload)
                embedding = resp_json["embeddings"]
            except Exception as e:
                raise ValueError(f"Error raised by the inference endpoint: {e}") from e

            embeddings.append(embedding)

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute document embeddings using an OctoAI instruct model."""
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        return self._compute_embeddings(texts, self.embed_instruction)

    def embed_query(self, text: str) -> List[float]:
        """Compute query embedding using an OctoAI instruct model."""
        text = text.replace("\n", " ")
        return self._compute_embeddings([text], self.embed_instruction)[0]
