from typing import Any, Dict, List

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, Extra, Field


class OnnxEmbeddings(BaseModel, Embeddings):
    """ONNX Embedding models.
    Example:
        .. code-block:: python
            from langchain.embeddings import OnnxEmbeddings
            from langchain.embeddings.huggingface import (
                DEFAULT_QUERY_BGE_INSTRUCTION_EN
            )
            model_name = "BAAI/bge-large-en"
            model_kwargs = {'device': 'cpu'}
            query_instruction = DEFAULT_QUERY_BGE_INSTRUCTION_EN
            onnx_emb = OnnxEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                query_instruction=query_instruction,
            )
    """

    client: Any  #: :meta private:
    tokenizer: Any  #: :meta private:
    model_name: str
    """The name of the HuggingFace model to transform to ONNX format."""
    query_instruction: str = Field(default="query:")
    """Instruction to use for embedding query."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass when tokenizing the input text."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        try:
            import onnx
            import onnxruntime
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Please ensure the required packages are installed with "
                "`pip install optimum transformers onnxruntime onnx`."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.client = ORTModelForFeatureExtraction.from_pretrained(
            self.model_name, export=True
        )

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
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
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
        text = self.query_instruction + text if self.query_instruction else text
        inputs = self.tokenizer([text], return_tensors="pt", **self.encode_kwargs)
        embedding = self.client(**inputs)
        return embedding.last_hidden_state.mean(dim=1).squeeze().tolist()
