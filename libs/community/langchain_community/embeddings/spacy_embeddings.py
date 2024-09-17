import importlib.util
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, model_validator


class SpacyEmbeddings(BaseModel, Embeddings):
    """Embeddings by spaCy models.

    Attributes:
        model_name (str): Name of a spaCy model.
        nlp (Any): The spaCy model loaded into memory.

    Methods:
        embed_documents(texts: List[str]) -> List[List[float]]:
            Generates embeddings for a list of documents.
        embed_query(text: str) -> List[float]:
            Generates an embedding for a single piece of text.
    """

    model_name: str = "en_core_web_sm"
    nlp: Optional[Any] = None

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """
        Validates that the spaCy package and the model are installed.

        Args:
            values (Dict): The values provided to the class constructor.

        Returns:
            The validated values.

        Raises:
            ValueError: If the spaCy package or the
            model are not installed.
        """
        if values.get("model_name") is None:
            values["model_name"] = "en_core_web_sm"

        model_name = values.get("model_name")

        # Check if the spaCy package is installed
        if importlib.util.find_spec("spacy") is None:
            raise ValueError(
                "SpaCy package not found. "
                "Please install it with `pip install spacy`."
            )
        try:
            # Try to load the spaCy model
            import spacy

            values["nlp"] = spacy.load(model_name)
        except OSError:
            # If the model is not found, raise a ValueError
            raise ValueError(
                f"SpaCy model '{model_name}' not found. "
                f"Please install it with"
                f" `python -m spacy download {model_name}`"
                "or provide a valid spaCy model name."
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
        return [self.nlp(text).vector.tolist() for text in texts]  # type: ignore[misc]

    def embed_query(self, text: str) -> List[float]:
        """
        Generates an embedding for a single piece of text.

        Args:
            text (str): The text to generate an embedding for.

        Returns:
            The embedding for the text.
        """
        return self.nlp(text).vector.tolist()  # type: ignore[misc]

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
