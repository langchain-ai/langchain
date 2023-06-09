"""Schema for a document.""" ""
from __future__ import annotations

from abc import ABC, abstractmethod
from inspect import signature
from typing import (
    Any,
    List,
    Optional,
)

from langchain.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForRetrieverRun,
    CallbackManager,
    CallbackManagerForRetrieverRun,
    Callbacks,
)
from langchain.schema.document import Document


class BaseRetriever(ABC):
    """Base interface for a retriever."""

    _new_arg_supported: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._new_arg_supported = (
            signature(cls.get_relevant_documents).parameters.get("run_manager")
            is not None
        )

    @abstractmethod
    def get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Get documents relevant for a query.
        Args:
            query: string to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """

    @abstractmethod
    async def aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Get documents relevant for a query.
        Args:
            query: string to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """

    def retrieve(
        self, query: str, *, callbacks: Callbacks = None, **kwargs: Any
    ) -> List[Document]:
        """Retrieve documents.
        Args:
            query: string to find relevant documents for
            callbacks: Callback manager or list of callbacks
        Returns:
            List of relevant documents
        """
        callback_manager = CallbackManager.configure(
            callbacks, None, verbose=kwargs.get("verbose", False)
        )
        run_manager = callback_manager.on_retriever_start(
            query,
            **kwargs,
        )
        try:
            # TODO: maybe also pass through run_manager is _run supports kwargs
            if self._new_arg_supported:
                result = self.get_relevant_documents(
                    query, run_manager=run_manager, **kwargs
                )
            else:
                result = self.get_relevant_documents(query)
        except Exception as e:
            run_manager.on_retriever_error(e)
            raise e
        else:
            run_manager.on_retriever_end(
                result,
                **kwargs,
            )
            return result

    async def aretrieve(
        self, query: str, *, callbacks: Callbacks = None, **kwargs: Any
    ) -> List[Document]:
        """Get documents relevant for a query.
        Args:
            query: string to find relevant documents for
            callbacks: Callback manager or list of callbacks
        Returns:
            List of relevant documents
        """
        callback_manager = AsyncCallbackManager.configure(
            callbacks, None, verbose=kwargs.get("verbose", False)
        )
        run_manager = await callback_manager.on_retriever_start(
            query,
            **kwargs,
        )
        try:
            if self._new_arg_supported:
                result = await self.aget_relevant_documents(
                    query, run_manager=run_manager, **kwargs
                )
            else:
                result = await self.aget_relevant_documents(query)
        except Exception as e:
            await run_manager.on_retriever_error(e)
            raise e
        else:
            await run_manager.on_retriever_end(
                result,
                **kwargs,
            )
            return result
