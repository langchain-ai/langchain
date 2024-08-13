"""**Embeddings** interface."""

import abc
from abc import ABC, abstractmethod
from typing import List, Union, Literal, Optional, Any
from typing import TypedDict

from langchain_core.documents.base import BaseMedia, Document, Blob
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import run_in_executor, RunnableConfig


class Embeddings(ABC):
    """Interface for embedding models.

    This is an interface meant for implementing text embedding models.

    Text embedding models are used to map text to a vector (a point in n-dimensional
    space).

    Texts that are similar will usually be mapped to points that are close to each
    other in this space. The exact details of what's considered "similar" and how
    "distance" is measured in this space are dependent on the specific embedding model.

    This abstraction contains a method for embedding a list of documents and a method
    for embedding a query text. The embedding of a query text is expected to be a single
    vector, while the embedding of a list of documents is expected to be a list of
    vectors.

    Usually the query embedding is identical to the document embedding, but the
    abstraction allows treating them independently.

    In addition to the synchronous methods, this interface also provides asynchronous
    versions of the methods.

    By default, the asynchronous methods are implemented using the synchronous methods;
    however, implementations may choose to override the asynchronous methods with
    an async native implementation for performance reasons.
    """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        return await run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        return await run_in_executor(None, self.embed_query, text)


# An input into the embedding model.
# The input can be a document, a media object, or a string.
# Whether it's supported depends on the specific embedding model.
EmbeddingInput = Union[BaseMedia, Document, str]


class BaseEmbedding(TypedDict):
    """Base embedding."""

    vector: List[float]
    scope: Literal["entire", "slice"]


class TextEmbedding(BaseEmbedding):
    """Text embedding."""

    type: Literal["text"]
    start: int
    limit: int


class ImageEmbedding(TypedDict):
    """Designed for embedding 2-D images."""

    type: Literal["image"]


class AudioEmbedding(TypedDict):
    """Audio embedding."""

    type: Literal["audio"]
    start: float
    limit: float


class VideoEmbedding(TypedDict):
    """Video embedding"""

    type: Literal["video"]
    start: float
    limit: float


Embedding = Union[TextEmbedding, ImageEmbedding, AudioEmbedding, VideoEmbedding]


class EmbeddingOutput(TypedDict):
    """The response of an embedding model."""

    embeddings: List[Embedding]


def _standardize_embedding_input(input_: EmbeddingInput) -> Blob:
    """Convert an embedding input into a standardized Blob."""
    if isinstance(input_, Blob):
        return input_
    elif isinstance(input_, Document):
        return Blob(
            id=input_.id,
            metadata=input_.metadata,
            data=input_.page_content,
            mimetype_type="text/plain",
        )
    elif isinstance(input_, str):  # This is a string of text to embed
        return Blob(
            metadata={},
            data=input_,
            mimetype_type="text/plain",
        )
    else:
        raise NotImplementedError()


class EmbeddingModel(RunnableSerializable[EmbeddingInput, EmbeddingOutput]):
    """An embedding model."""

    @abc.abstractmethod
    def _embed(
        self, input_: Blob, config: Optional[RunnableConfig], **kwargs: Any
    ) -> EmbeddingOutput:
        """Embed input."""

    def invoke(
        self,
        input_: EmbeddingInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> EmbeddingOutput:
        """Embed input."""
        blob = _standardize_embedding_input(input_)
        return self._call_with_config(self._embed, blob, config=config, **kwargs)
