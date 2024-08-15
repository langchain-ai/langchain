from typing import Any, Dict, List

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.utils import secret_from_env
from openai import OpenAI  # type: ignore


class FireworksEmbeddings(BaseModel, Embeddings):
    """Fireworks embedding model integration.

     Setup:
         Install ``langchain_fireworks`` and set environment variable
         ``FIREWORKS_API_KEY``.

         .. code-block:: bash

             pip install -U langchain_fireworks
             export FIREWORKS_API_KEY="your-api-key"

     Key init args — completion params:
         model: str
             Name of Fireworks model to use.

    Key init args — client params:
        fireworks_api_key: SecretStr
            Fireworks API key.

     See full list of supported init args and their descriptions in the params section.

     Instantiate:
         .. code-block:: python

             from langchain_fireworks import FireworksEmbeddings

             model = FireworksEmbeddings(
                 model='nomic-ai/nomic-embed-text-v1.5'
                 # Use FIREWORKS_API_KEY env var or pass it in directly
                 # fireworks_api_key="..."
             )

     Embed multiple texts:
         .. code-block:: python

             vectors = embeddings.embed_documents(['hello', 'goodbye'])
             # Showing only the first 3 coordinates
             print(len(vectors))
             print(vectors[0][:3])

         .. code-block:: python

             2
             [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]


     Embed single text:
         .. code-block:: python

             input_text = "The meaning of life is 42"
             vector = embeddings.embed_query('hello')
             print(vector[:3])

         .. code-block:: python

             [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]
    """

    _client: OpenAI = Field(default=None)
    fireworks_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env(
            "FIREWORKS_API_KEY",
            default="",
        ),
    )
    """Fireworks API key.
    
    Automatically read from env variable `FIREWORKS_API_KEY` if not provided.
    """
    model: str = "nomic-ai/nomic-embed-text-v1.5"

    @root_validator(pre=False, skip_on_failure=True)
    def validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate environment variables."""
        values["_client"] = OpenAI(
            api_key=values["fireworks_api_key"].get_secret_value(),
            base_url="https://api.fireworks.ai/inference/v1",
        )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return [
            i.embedding
            for i in self._client.embeddings.create(input=texts, model=self.model).data
        ]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]
