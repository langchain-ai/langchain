import logging
from typing import Dict, List

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import root_validator

from langchain.llms.vertexai import _VertexAICommon
from langchain.utilities.vertexai import raise_vertex_import_error

logger = logging.getLogger(__name__)

class VertexAIEmbeddings(_VertexAICommon, Embeddings):
    """Google Cloud VertexAI embedding models."""

    model_name: str = "textembedding-gecko"
    show_progress_bar: bool = False

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
        if self.show_progress_bar:
            try:
                from tqdm import tqdm
            except ImportError:
                logger.warning(
                    "Unable to show progress bar because tqdm could not be imported. "
                    "Please install with `pip install tqdm`."
                )
                progress_bar = None
            else:
                progress_bar = tqdm(total=len(texts), desc="VertexAIEmbeddings")
        else:
            progress_bar = None

        embeddings = []
        for batch_start in range(0, len(texts), batch_size):
            text_batch = texts[batch_start: batch_start + batch_size]
            embeddings_batch = self.client.get_embeddings(text_batch)
            embeddings.extend([el.values for el in embeddings_batch])

            if progress_bar:
                progress_bar.update(len(text_batch))

        if progress_bar:
            progress_bar.close()

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
