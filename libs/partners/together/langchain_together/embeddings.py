import os
from typing import Any, Dict, List

import together  # type: ignore
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str


class TogetherEmbeddings(BaseModel, Embeddings):
    """TogetherEmbeddings embedding model.

    Example:
        .. code-block:: python

            from langchain_together import TogetherEmbeddings

            model = TogetherEmbeddings(
                model='togethercomputer/m2-bert-80M-8k-retrieval'
            )
    """

    _client: together.Together
    together_api_key: SecretStr = convert_to_secret_str("")
    model: str

    @root_validator()
    def validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate environment variables."""
        together_api_key = convert_to_secret_str(
            values.get("together_api_key") or os.getenv("TOGETHER_API_KEY") or ""
        )
        values["together_api_key"] = together_api_key

        # note this sets it globally for module
        # there isn't currently a way to pass it into client
        together.api_key = together_api_key.get_secret_value()
        values["_client"] = together.Together()
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
