"""Wrapper around Google VertexAI embedding models."""
from typing import Dict, List

from pydantic import root_validator

from langchain.embeddings.base import Embeddings
from langchain.llms.vertex import _VertexAICommon


class VertexAIEmbeddings(_VertexAICommon, Embeddings):
    model_name: str = "embedding-gecko-001"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validates that the python package exists in environment."""
        try:
            from google.cloud.aiplatform.private_preview.language_models import (
                TextEmbeddingModel,
            )

        except ImportError:
            raise ValueError("Could not import Vertex AI LLM python package. ")
        try:
            values["client"] = TextEmbeddingModel.from_pretrained(values["model_name"])
        except AttributeError:
            raise ValueError("Could not initialize Vertex AI LLM.")

        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings = self.client.get_embeddings(texts)
        return [el.values for el in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        embeddings = self.client.get_embeddings([text])
        return embeddings[0].values
