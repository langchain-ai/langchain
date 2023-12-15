from typing import Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import root_validator

from langchain_community.llms.vertexai import _VertexAICommon
from langchain_community.utilities.vertexai import raise_vertex_import_error


class VertexAIEmbeddings(_VertexAICommon, Embeddings):
    """Google Cloud VertexAI embedding models."""

    model_name: str = "textembedding-gecko"
    # https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.language_models.TextEmbeddingInput
    task_type: Optional[str] = None
    auto_truncate: bool = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validates that the python package exists in environment."""
        cls._try_init_vertexai(values)
        try:
            from vertexai.language_models import TextEmbeddingModel
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
        try:
            from vertexai.language_models import TextEmbeddingInput
        except ImportError:
            raise_vertex_import_error()

        embeddings = []
        for batch in range(0, len(texts), batch_size):
            text_batch = [
                TextEmbeddingInput(text=text, task_type=self.task_type)
                for text in texts[batch : batch + batch_size]
            ]
            embeddings_batch = self.client.get_embeddings(
                texts=text_batch, auto_truncate=self.auto_truncate
            )
            embeddings.extend([el.values for el in embeddings_batch])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        try:
            from vertexai.language_models import TextEmbeddingInput
        except ImportError:
            raise_vertex_import_error()

        embeddings = self.client.get_embeddings(
            texts=[TextEmbeddingInput(text=text, task_type=self.task_type)],
            auto_truncate=self.auto_truncate,
        )
        return embeddings[0].values
