from typing import Any, Dict, List

from transformers import AutoTokenizer

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, Extra, Field


class OnnxEmbeddings(BaseModel, Embeddings):
    """ONNX Embedding models.

    Example:
        .. code-block:: python

            from your_module import OnnxEmbeddings

            model_name = "BAAI/bge-large-en"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'return_tensors': 'pt'}
            onnx_emb = OnnxEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    """

    client: Any  #: :meta private:
    tokenizer: Any  #: :meta private:
    model_name: str
    """Directory where the onnx model and tokenizer are saved."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass when tokenizing the input text."""

    def __init__(self, **kwargs: Any):
        """Initialize the onnx model."""
        super().__init__(**kwargs)
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction

        except ImportError as exc:
            raise ImportError(
                "Could not import optimum python package. " "Please install it with `pip install optimum onnx`."
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.client = ORTModelForFeatureExtraction.from_pretrained(self.model_name)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using an ONNX model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = [t.replace("\n", " ") for t in texts]
        inputs = self.tokenizer(texts, **self.encode_kwargs)
        embeddings = self.client(**inputs)
        return embeddings.last_hidden_state.mean(dim=1).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using an ONNX model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        inputs = self.tokenizer([text], **self.encode_kwargs)
        embedding = self.client(**inputs)
        return embedding.last_hidden_state.mean(dim=1).squeeze().tolist()
