import importlib.util
from typing import Any, Dict, List

from pydantic import BaseModel, Extra, root_validator

from langchain.embeddings.base import Embeddings


class SpacyEmbeddings(BaseModel, Embeddings):
    """
    SpacyEmbeddings is a class for generating embeddings using the Spacy library.
    It only supports the 'en_core_web_sm' model.

    Attributes:
        nlp (Any): The Spacy model loaded into memory.

    Methods:
        embed_documents(texts: List[str]) -> List[List[float]]:
            Generates embeddings for a list of documents.
        embed_query(text: str) -> List[float]:
            Generates an embedding for a single piece of text.
    """

    nlp: Any  # The Spacy model loaded into memory

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid  # Forbid extra attributes during model initialization

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """
        Validates that the Spacy package and the 'en_core_web_sm' model are installed.

        Args:
            values (Dict): The values provided to the class constructor.

        Returns:
            The validated values.

        Raises:
            ValueError: If the Spacy package or the 'en_core_web_sm'
            model are not installed.
        """
        # Check if the Spacy package is installed
        if importlib.util.find_spec("spacy") is None:
            raise ValueError(
                "Spacy package not found. "
                "Please install it with `pip install spacy`."
            )
        try:
            # Try to load the 'en_core_web_sm' Spacy model
            import spacy

            values["nlp"] = spacy.load("en_core_web_sm")
        except OSError:
            # If the model is not found, raise a ValueError
            raise ValueError(
                "Spacy model 'en_core_web_sm' not found. "
                "Please install it with"
                " `python -m spacy download en_core_web_sm`."
            )
        return values  # Return the validated values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of documents.

        Args:
            texts (List[str]): The documents to generate embeddings for.

        Returns:
            A list of embeddings, one for each document.
        """
        return [self.nlp(text).vector.tolist() for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        Generates an embedding for a single piece of text.

        Args:
            text (str): The text to generate an embedding for.

        Returns:
            The embedding for the text.
        """
        return self.nlp(text).vector.tolist()

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously generates embeddings for a list of documents.
        This method is not implemented and raises a NotImplementedError.

        Args:
            texts (List[str]): The documents to generate embeddings for.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError("Asynchronous embedding generation is not supported.")

    async def aembed_query(self, text: str) -> List[float]:
        """
        Asynchronously generates an embedding for a single piece of text.
        This method is not implemented and raises a NotImplementedError.

        Args:
            text (str): The text to generate an embedding for.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError("Asynchronous embedding generation is not supported.")
