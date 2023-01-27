"""Wrapper around HuggingFace embedding models."""
from typing import Any, List
from enum import Enum

from pydantic import BaseModel, Extra

from langchain.embeddings.base import Embeddings

DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

class MODEL_TYPE(Enum):
    SENTENCE_TRANSFORMER = 1
    INSTRUCTION_EMBEDDING = 2

class HuggingFaceEmbeddings(BaseModel, Embeddings):
    """Wrapper around sentence_transformers embedding models.

    To use sentence transformers, you should have the ``sentence_transformers`` python package installed. 
    To use Instructor, you should have ``InstructorEmbedding`` python package installed.

    Example:
        .. code-block:: python

            from langchain.embeddings import HuggingFaceEmbeddings
            model_name = "sentence-transformers/all-mpnet-base-v2"
            hf = HuggingFaceEmbeddings(model_name=model_name)
    """

    client: Any  #: :meta private:
    model_name: str = DEFAULT_MODEL_NAME
    model_type: str = MODEL_TYPE.SENTENCE_TRANSFORMER
    """Model name to use."""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
        
        if (self.model_name == DEFAULT_MODEL_NAME):
            try:
                import sentence_transformers

                self.client = sentence_transformers.SentenceTransformer(self.model_name)
            except ImportError:
                raise ValueError(
                    "Could not import sentence_transformers python package. "
                    "Please install it with `pip install sentence_transformers`."
                )
        elif ("instructor" in self.model_name):
            try:
                from InstructorEmbedding import INSTRUCTOR
                self.model_type = MODEL_TYPE.INSTRUCTION_EMBEDDING
                self.client = INSTRUCTOR(self.model_name)
            except ImportError:
                raise ValueError(
                    "Could not import InstructorEmbedding python package. "
                    "Please install it with `pip install InstructorEmbedding`."
                )


    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.client.encode(texts)

        if (self.model_name == DEFAULT_MODEL_NAME):
            return embeddings.tolist()

        return embeddings.tolist()

    ## Embedding instruction-tuned models requires a list of instruction, text pairs.
    def embed_documents(self, texts: List[List[str]]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        if (self.model_type != MODEL_TYPE.INSTRUCTION_EMBEDDING):
            raise ValueError(
                    "Erorr: You passed a list of string pairs but did not instantiate an Instruction embedding model. "
                ) 

        for text_list in texts:
            for text in text_list:
                if isinstance(text, str):
                    text = text.replace("\n", " ")

        embeddings = self.client.encode(texts)

        return embeddings.tolist()

    ## Embedding instruction-tuned model queries requires a list of instruction, text pairs.
    def embed_query(self, texts: List[str]) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        if (self.model_type != MODEL_TYPE.INSTRUCTION_EMBEDDING):
            raise ValueError(
                    "Erorr: You passed a string pair but did not instantiate an Instruction embedding model. "
                ) 

        for text in texts:
            if isinstance(text, str):
                text = text.replace("\n", " ")

        embedding = self.client.encode(texts)
        return embedding.tolist()