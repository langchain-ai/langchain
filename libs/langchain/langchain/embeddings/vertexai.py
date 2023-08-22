from typing import Dict, List

from langchain.embeddings.base import Embeddings
from langchain.llms.vertexai import _VertexAICommon
from langchain.pydantic_v1 import root_validator
from langchain.utilities.vertexai import raise_vertex_import_error


class VertexAIEmbeddings(_VertexAICommon, Embeddings):
    """Google Cloud VertexAI embedding models."""

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

    def embed_documents(
        self, texts: List[str], batch_size: int = 5
    ) -> List[List[float]]:
        """Embed a list of strings. Vertex AI currently
        sets a max batch size of 5 strings.

        Args:
            texts: List[str] The list of strings to embed.
            batch_size: [int] The batch size of embeddings to send to the model

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = []
        for batch in range(0, len(texts), batch_size):
            text_batch = texts[batch : batch + batch_size]
            embeddings_batch = self.client.get_embeddings(text_batch)
            embeddings.extend([el.values for el in embeddings_batch])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        embeddings = self.client.get_embeddings([text])
        return embeddings[0].values
