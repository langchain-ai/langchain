"""Wrapper around sentence transformer embedding models."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, Field, root_validator

from langchain.embeddings.base import Embeddings


class SentenceTransformerEmbeddings(BaseModel, Embeddings):
    embedding_function: Any  #: :meta private:

    model: Optional[str] = Field("all-MiniLM-L6-v2", alias="model")
    """Transformer model to use."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that sentence_transformers library is installed."""
        model = values["model"]

        try:
            from sentence_transformers import SentenceTransformer

            values["embedding_function"] = SentenceTransformer(model)
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import sentence_transformers library. "
                "Please install the sentence_transformers library to "
                "use this embedding model: pip install sentence_transformers"
            )
        except Exception:
            raise NameError(f"Could not load SentenceTransformer model {model}.")

        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the SentenceTransformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = self.embedding_function.encode(
            texts, convert_to_numpy=True
        ).tolist()
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the SentenceTransformer model.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        return self.embed_documents([text])[0]
