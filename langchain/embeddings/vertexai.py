"""Wrapper around Google VertexAI embedding models."""
from typing import Dict, List, Optional

from pydantic import root_validator

from langchain.embeddings.base import Embeddings
from langchain.llms.vertexai import _VertexAICommon


class VertexAIEmbeddings(_VertexAICommon, Embeddings):
    model_name: Optional[str] = "textembedding-gecko@001"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validates that the python package exists in environment."""
        try:
            import vertexai
            from vertexai.preview.language_models import TextEmbeddingModel

            vertexai.init()
        except ImportError:
            cls._raise_import_error()
        values["client"] = TextEmbeddingModel.from_pretrained(values["model_name"])
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings = self.client.get_embeddings(texts)
        return [el.values for el in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        embeddings = self.client.get_embeddings([text])
        return embeddings[0].values
