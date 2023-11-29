from typing import Any, Dict, List

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, root_validator


class GPT4AllEmbeddings(BaseModel, Embeddings):
    """GPT4All embedding models.

    To use, you should have the gpt4all python package installed

    Example:
        .. code-block:: python

            from langchain.embeddings import GPT4AllEmbeddings

            embeddings = GPT4AllEmbeddings()
    """

    client: Any  #: :meta private:

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that GPT4All library is installed."""

        try:
            from gpt4all import Embed4All

            values["client"] = Embed4All()
        except ImportError:
            raise ImportError(
                "Could not import gpt4all library. "
                "Please install the gpt4all library to "
                "use this embedding model: pip install gpt4all"
            )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using GPT4All.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        embeddings = [self.client.embed(text) for text in texts]
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using GPT4All.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]
