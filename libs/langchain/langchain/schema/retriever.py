from __future__ import annotations

import asyncio
import warnings
from abc import ABC, abstractmethod
from functools import partial
from inspect import signature
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain.load.dump import dumpd
from langchain.schema.document import Document
from langchain.schema.runnable import RunnableConfig, RunnableSerializable

if TYPE_CHECKING:
    from langchain.callbacks.manager import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
        Callbacks,
    )


class BaseRetriever(RunnableSerializable[str, List[Document]], ABC):
    """Abstract base class for a Document retrieval system.

    A retrieval system is defined as something that can take string queries and return
        the most 'relevant' Documents from some source.

    Example:
        .. code-block:: python

            class TFIDFRetriever(BaseRetriever, BaseModel):
                vectorizer: Any
                docs: List[Document]
                tfidf_array: Any
                k: int = 4

                class Config:
                    arbitrary_types_allowed = True

                def get_relevant_documents(self, query: str) -> List[Document]:
                    from sklearn.metrics.pairwise import cosine_similarity

                    # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
                    query_vec = self.vectorizer.transform([query])
                    # Op -- (n_docs,1) -- Cosine Sim with each doc
                    results = cosine_similarity(self.tfidf_array, query_vec).reshape((-1,))
                    return [self.docs[i] for i in results.argsort()[-self.k :][::-1]]
    """  # noqa: E501

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    _new_arg_supported: bool = False
    _expects_other_args: bool = False
    tags: Optional[List[str]] = None
    """Optional list of tags associated with the retriever. Defaults to None
    These tags will be associated with each call to this retriever,
    and passed as arguments to the handlers defined in `callbacks`.
    You can use these to eg identify a specific instance of a retriever with its 
    use case.
    """
    metadata: Optional[Dict[str, Any]] = None
    """Optional metadata associated with the retriever. Defaults to None
    This metadata will be associated with each call to this retriever,
    and passed as arguments to the handlers defined in `callbacks`.
    You can use these to eg identify a specific instance of a retriever with its 
    use case.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Version upgrade for old retrievers that implemented the public
        # methods directly.
        if cls.get_relevant_documents != BaseRetriever.get_relevant_documents:
            warnings.warn(
                "Retrievers must implement abstract `_get_relevant_documents` method"
                " instead of `get_relevant_documents`",
                DeprecationWarning,
            )
            swap = cls.get_relevant_documents
            cls.get_relevant_documents = (  # type: ignore[assignment]
                BaseRetriever.get_relevant_documents
            )
            cls._get_relevant_documents = swap  # type: ignore[assignment]
        if (
            hasattr(cls, "aget_relevant_documents")
            and cls.aget_relevant_documents != BaseRetriever.aget_relevant_documents
        ):
            warnings.warn(
                "Retrievers must implement abstract `_aget_relevant_documents` method"
                " instead of `aget_relevant_documents`",
                DeprecationWarning,
            )
            aswap = cls.aget_relevant_documents
            cls.aget_relevant_documents = (  # type: ignore[assignment]
                BaseRetriever.aget_relevant_documents
            )
            cls._aget_relevant_documents = aswap  # type: ignore[assignment]
        parameters = signature(cls._get_relevant_documents).parameters
        cls._new_arg_supported = parameters.get("run_manager") is not None
        # If a V1 retriever broke the interface and expects additional arguments
        cls._expects_other_args = (
            len(set(parameters.keys()) - {"self", "query", "run_manager"}) > 0
        )

    def invoke(
        self, input: str, config: Optional[RunnableConfig] = None
    ) -> List[Document]:
        config = config or {}
        return self.get_relevant_documents(
            input,
            callbacks=config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
        )

    async def ainvoke(
        self,
        input: str,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> List[Document]:
        config = config or {}
        return await self.aget_relevant_documents(
            input,
            callbacks=config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
        )

    @abstractmethod
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self._get_relevant_documents, run_manager=run_manager), query
        )

    def get_relevant_documents(
        self,
        query: str,
        *,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Retrieve documents relevant to a query.
        Args:
            query: string to find relevant documents for
            callbacks: Callback manager or list of callbacks
            tags: Optional list of tags associated with the retriever. Defaults to None
                These tags will be associated with each call to this retriever,
                and passed as arguments to the handlers defined in `callbacks`.
            metadata: Optional metadata associated with the retriever. Defaults to None
                This metadata will be associated with each call to this retriever,
                and passed as arguments to the handlers defined in `callbacks`.
        Returns:
            List of relevant documents
        """
        from langchain.callbacks.manager import CallbackManager

        callback_manager = CallbackManager.configure(
            callbacks,
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=tags,
            local_tags=self.tags,
            inheritable_metadata=metadata,
            local_metadata=self.metadata,
        )
        run_manager = callback_manager.on_retriever_start(
            dumpd(self),
            query,
            name=run_name,
            **kwargs,
        )
        try:
            _kwargs = kwargs if self._expects_other_args else {}
            if self._new_arg_supported:
                result = self._get_relevant_documents(
                    query, run_manager=run_manager, **_kwargs
                )
            else:
                result = self._get_relevant_documents(query, **_kwargs)
        except Exception as e:
            run_manager.on_retriever_error(e)
            raise e
        else:
            run_manager.on_retriever_end(
                result,
                **kwargs,
            )
            return result

    async def aget_relevant_documents(
        self,
        query: str,
        *,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query.
        Args:
            query: string to find relevant documents for
            callbacks: Callback manager or list of callbacks
            tags: Optional list of tags associated with the retriever. Defaults to None
                These tags will be associated with each call to this retriever,
                and passed as arguments to the handlers defined in `callbacks`.
            metadata: Optional metadata associated with the retriever. Defaults to None
                This metadata will be associated with each call to this retriever,
                and passed as arguments to the handlers defined in `callbacks`.
        Returns:
            List of relevant documents
        """
        from langchain.callbacks.manager import AsyncCallbackManager

        callback_manager = AsyncCallbackManager.configure(
            callbacks,
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=tags,
            local_tags=self.tags,
            inheritable_metadata=metadata,
            local_metadata=self.metadata,
        )
        run_manager = await callback_manager.on_retriever_start(
            dumpd(self),
            query,
            name=run_name,
            **kwargs,
        )
        try:
            _kwargs = kwargs if self._expects_other_args else {}
            if self._new_arg_supported:
                result = await self._aget_relevant_documents(
                    query, run_manager=run_manager, **_kwargs
                )
            else:
                result = await self._aget_relevant_documents(query, **_kwargs)
        except Exception as e:
            await run_manager.on_retriever_error(e)
            raise e
        else:
            await run_manager.on_retriever_end(
                result,
                **kwargs,
            )
            return result
