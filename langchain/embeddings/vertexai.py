"""Wrapper around Google VertexAI embedding models."""
from typing import Dict, List

from pydantic import root_validator

from langchain.embeddings.base import Embeddings
from langchain.llms.vertexai import _VertexAICommon
from langchain.utilities.vertexai import raise_vertex_import_error


class VertexAIEmbeddings(_VertexAICommon, Embeddings):
    model_name: str = "textembedding-gecko"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validates that the python package exists in environment."""
        cls._try_init_vertexai(values)
        try:
            from vertexai.preview.language_models import TextEmbeddingModel
        except ImportError:
            raise_vertex_import_error()
        values["client"] = TextEmbeddingModel.from_pretrained(values["model_name"])
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of strings.

        Args:
            texts: List[str] The list of strings to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = self.client.get_embeddings(texts)
        return [el.values for el in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        embeddings = self.client.get_embeddings([text])
        return embeddings[0].values
