"""Interface for embedding models."""
from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import Field, validator, BaseModel

from langchain.callbacks import get_callback_manager
from langchain.callbacks.base import BaseCallbackManager
from langchain.schema import EmbeddingResult


class Embeddings(BaseModel, ABC):
    """Interface for embedding models."""

    callback_manager: BaseCallbackManager = Field(default_factory=get_callback_manager)

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @validator("callback_manager", pre=True, always=True)
    def set_callback_manager(
        cls, callback_manager: Optional[BaseCallbackManager]
    ) -> BaseCallbackManager:
        """
        If callback manager is None, set it.
        This allows users to pass in None as callback manager, which is a nice UX.
        """
        return callback_manager or get_callback_manager()

    # non-abstract for now to avoid breaking changes
    def _embed_documents(self, texts: List[str]) -> EmbeddingResult:
        pass

    def _embed_query(self, text: str) -> EmbeddingResult:
        pass

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        self.callback_manager.on_embedding_start(
            {"name": self.__class__.__name__}, texts
        )
        try:
            result = self._embed_documents(texts)
        except (Exception, KeyboardInterrupt) as e:
            self.callback_manager.on_embedding_error(e)
            raise e
        self.callback_manager.on_embedding_end(result)
        return result.embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        self.callback_manager.on_embedding_start(
            {"name": self.__class__.__name__}, [text]
        )
        try:
            result = self._embed_query(text)
        except (Exception, KeyboardInterrupt) as e:
            self.callback_manager.on_embedding_error(e)
            raise e
        self.callback_manager.on_embedding_end(result)
        return result.embeddings[0]
