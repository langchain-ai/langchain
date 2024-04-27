import logging
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra

logger = logging.getLogger(__name__)


class OllamaEmbeddings(BaseModel, Embeddings):
    """Ollama locally runs large language models. Follow the instructions at https://ollama.ai/ for more information.
    Example:
        .. code-block:: python

            from langchain_community.embeddings import OllamaEmbeddings
            ollama_emb = OllamaEmbeddings(
                model="llama:7b",
            )
            r1 = ollama_emb.embed_documents(
                [
                    "Alpha is the first letter of Greek alphabet",
                    "Beta is the second letter of Greek alphabet",
                ]
            )
            r2 = ollama_emb.embed_query(
                "What is the second letter of Greek alphabet"

    """  # noqa: E501

    model: str = "mxbai-embed-large"  # Model name to use.

    show_progress: bool = False
    """Whether to show a tqdm progress bar. Must have `tqdm` installed."""

    @property
    def _default_params(self) -> dict:
        """Assemble default parameters for embeddings."""
        return {
            "model": self.model,
        }

    @property
    def _identifying_params(self) -> dict:
        """Identifying parameters used primarily for caching or debugging purposes."""
        return self._default_params

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def _embed(self, inputs: List[str]) -> List[List[float]]:
        """Embed a list of input texts using the Ollama embeddings API."""
        try:
            import ollama
        except ImportError as e:
            raise ImportError(
                "Unable to import ollama, please install with `pip install -U ollama`."
            ) from e
        if self.show_progress:
            from tqdm import tqdm

            inputs = tqdm(inputs, desc="Embedding Documents")

        embeddings = []
        for input_text in inputs:
            response = ollama.embeddings(model=self.model, prompt=input_text)
            embeddings.append(response["embedding"])
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using an Ollama deployed embedding model."""
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using an Ollama deployed embedding model."""
        return self._embed([text])[0]
