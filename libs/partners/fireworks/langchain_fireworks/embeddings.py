from typing import List

from langchain_core.embeddings import Embeddings
from langchain_core.utils import secret_from_env
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

# type: ignore


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

    client: OpenAI = Field(default=None, exclude=True)  # type: ignore[assignment] # :meta private:
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

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate environment variables."""
        self.client = OpenAI(
            api_key=self.fireworks_api_key.get_secret_value(),
            base_url="https://api.fireworks.ai/inference/v1",
        )
        return self

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return [
            i.embedding
            for i in self.client.embeddings.create(input=texts, model=self.model).data
        ]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]
