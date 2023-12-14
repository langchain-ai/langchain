from __future__ import annotations

import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class VoyageEmbeddings(BaseModel, Embeddings):
    """Voyage embedding models.

    To use, you should have the environment variable ``VOYAGE_API_KEY`` set with
    your API key or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import VoyageEmbeddings

            voyage = VoyageEmbeddings(voyage_api_key="your-api-key")
            text = "This is a test query."
            query_result = voyage.embed_query(text)
    """

    client: Any

    model: str = "voyage-01"
    voyage_api_key: Optional[str] = None
    batch_size: int = 64
    """Maximum number of texts to embed in each request."""
    show_progress_bar: bool = False
    """Whether to show a progress bar when embedding. Must have tqdm installed if set 
        to True."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["voyage_api_key"] = get_from_dict_or_env(
            values, "voyage_api_key", "VOYAGE_API_KEY"
        )

        try:
            import voyageai

            values["client"] = voyageai
            voyageai.api_key = values["voyage_api_key"]
        except ImportError:
            raise ImportError(
                "Could not import voyageai python package. "
                "Please install it with `pip install voyageai`."
            )
        return values

    def _get_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        input_type: Optional[str] = None,
    ) -> List[List[float]]:
        embeddings: List[List[float]] = []

        if batch_size is None:
            batch_size = self.batch_size

        if self.show_progress_bar:
            try:
                from tqdm.auto import tqdm
            except ImportError as e:
                raise ImportError(
                    "Must have tqdm installed if `show_progress_bar` is set to True. "
                    "Please install with `pip install tqdm`."
                ) from e

            _iter = tqdm(range(0, len(texts), batch_size))
        else:
            _iter = range(0, len(texts), batch_size)

        if input_type and input_type not in ["query", "document"]:
            raise ValueError(
                f"input_type {input_type} is invalid. Options: None, 'query', "
                "'document'."
            )

        for i in _iter:
            response = self.client.get_embeddings(
                texts[i : i + batch_size],
                model=self.model,
                input_type=input_type,
            )
            embeddings.extend(response)

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Voyage Embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        return self._get_embeddings(
            texts, batch_size=self.batch_size, input_type="document"
        )

    def embed_query(self, text: str) -> List[float]:
        """Call out to Voyage Embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        return self._get_embeddings([text], input_type="query")[0]

    def embed_general_texts(
        self, texts: List[str], *, input_type: Optional[str] = None
    ) -> List[List[float]]:
        """Call out to Voyage Embedding endpoint for embedding general text.

        Args:
            texts: The list of texts to embed.
            input_type: Type of the input text. Default to None, meaning the type is
                unspecified. Other options: query, document.

        Returns:
            Embedding for the text.
        """
        return self._get_embeddings(
            texts, batch_size=self.batch_size, input_type=input_type
        )
