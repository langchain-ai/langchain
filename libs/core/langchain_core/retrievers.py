"""**Retriever** class returns Documents given a text **query**.

It is more general than a vector store. A retriever does not need to be able to
store documents, only to return (or retrieve) it. Vector stores can be used as
the backbone of a retriever, but there are other types of retrievers as well.

**Class hierarchy:**

.. code-block::

    BaseRetriever --> <name>Retriever  # Examples: ArxivRetriever, MergerRetriever

**Main helpers:**

.. code-block::

    RetrieverInput, RetrieverOutput, RetrieverLike, RetrieverOutputLike,
    Document, Serializable, Callbacks,
    CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from inspect import signature
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import ConfigDict
from typing_extensions import TypedDict

from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableSerializable,
    ensure_config,
)
from langchain_core.runnables.config import run_in_executor

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
        Callbacks,
    )

RetrieverInput = str
RetrieverOutput = List[Document]
RetrieverLike = Runnable[RetrieverInput, RetrieverOutput]
RetrieverOutputLike = Runnable[Any, RetrieverOutput]


class LangSmithRetrieverParams(TypedDict, total=False):
    """LangSmith parameters for tracing."""

    ls_retriever_name: str
    """Retriever name."""
    ls_vector_store_provider: Optional[str]
    """Vector store provider."""
    ls_embedding_provider: Optional[str]
    """Embedding provider."""
    ls_embedding_model: Optional[str]
    """Embedding model."""


class BaseRetriever(RunnableSerializable[RetrieverInput, RetrieverOutput], ABC):
    """Abstract base class for a Document retrieval system.

    A retrieval system is defined as something that can take string queries and return
    the most 'relevant' Documents from some source.

    Usage:

    A retriever follows the standard Runnable interface, and should be used
    via the standard Runnable methods of `invoke`, `ainvoke`, `batch`, `abatch`.

    Implementation:

    When implementing a custom retriever, the class should implement
    the `_get_relevant_documents` method to define the logic for retrieving documents.

    Optionally, an async native implementations can be provided by overriding the
    `_aget_relevant_documents` method.

    Example: A retriever that returns the first 5 documents from a list of documents

        .. code-block:: python

            from langchain_core import Document, BaseRetriever
            from typing import List

            class SimpleRetriever(BaseRetriever):
                docs: List[Document]
                k: int = 5

                def _get_relevant_documents(self, query: str) -> List[Document]:
                    \"\"\"Return the first k documents from the list of documents\"\"\"
                    return self.docs[:self.k]

                async def _aget_relevant_documents(self, query: str) -> List[Document]:
                    \"\"\"(Optional) async native implementation.\"\"\"
                    return self.docs[:self.k]

    Example: A simple retriever based on a scikit-learn vectorizer

        .. code-block:: python

            from sklearn.metrics.pairwise import cosine_similarity

            class TFIDFRetriever(BaseRetriever, BaseModel):
                vectorizer: Any
                docs: List[Document]
                tfidf_array: Any
                k: int = 4

                class Config:
                    arbitrary_types_allowed = True

                def _get_relevant_documents(self, query: str) -> List[Document]:
                    # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
                    query_vec = self.vectorizer.transform([query])
                    # Op -- (n_docs,1) -- Cosine Sim with each doc
                    results = cosine_similarity(self.tfidf_array, query_vec).reshape((-1,))
                    return [self.docs[i] for i in results.argsort()[-self.k :][::-1]]
    """  # noqa: E501

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    _new_arg_supported: bool = False
    _expects_other_args: bool = False
    tags: Optional[List[str]] = None
    """Optional list of tags associated with the retriever. Defaults to None.
    These tags will be associated with each call to this retriever,
    and passed as arguments to the handlers defined in `callbacks`.
    You can use these to eg identify a specific instance of a retriever with its 
    use case.
    """
    metadata: Optional[Dict[str, Any]] = None
    """Optional metadata associated with the retriever. Defaults to None.
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
                stacklevel=4,
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
                stacklevel=4,
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

    def _get_ls_params(self, **kwargs: Any) -> LangSmithRetrieverParams:
        """Get standard params for tracing."""

        default_retriever_name = self.get_name()
        if default_retriever_name.startswith("Retriever"):
            default_retriever_name = default_retriever_name[9:]
        elif default_retriever_name.endswith("Retriever"):
            default_retriever_name = default_retriever_name[:-9]
        default_retriever_name = default_retriever_name.lower()

        ls_params = LangSmithRetrieverParams(ls_retriever_name=default_retriever_name)
        return ls_params

    def invoke(
        self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> List[Document]:
        """Invoke the retriever to get relevant documents.

        Main entry point for synchronous retriever invocations.

        Args:
            input: The query string.
            config: Configuration for the retriever. Defaults to None.
            kwargs: Additional arguments to pass to the retriever.

        Returns:
            List of relevant documents.

        Examples:

        .. code-block:: python

            retriever.invoke("query")
        """
        from langchain_core.callbacks.manager import CallbackManager

        config = ensure_config(config)
        inheritable_metadata = {
            **(config.get("metadata") or {}),
            **self._get_ls_params(**kwargs),
        }
        callback_manager = CallbackManager.configure(
            config.get("callbacks"),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=config.get("tags"),
            local_tags=self.tags,
            inheritable_metadata=inheritable_metadata,
            local_metadata=self.metadata,
        )
        run_manager = callback_manager.on_retriever_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=kwargs.pop("run_id", None),
        )
        try:
            _kwargs = kwargs if self._expects_other_args else {}
            if self._new_arg_supported:
                result = self._get_relevant_documents(
                    input, run_manager=run_manager, **_kwargs
                )
            else:
                result = self._get_relevant_documents(input, **_kwargs)
        except Exception as e:
            run_manager.on_retriever_error(e)
            raise e
        else:
            run_manager.on_retriever_end(
                result,
            )
            return result

    async def ainvoke(
        self,
        input: str,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Asynchronously invoke the retriever to get relevant documents.

        Main entry point for asynchronous retriever invocations.

        Args:
            input: The query string.
            config: Configuration for the retriever. Defaults to None.
            kwargs: Additional arguments to pass to the retriever.

        Returns:
            List of relevant documents.

        Examples:

        .. code-block:: python

            await retriever.ainvoke("query")
        """
        from langchain_core.callbacks.manager import AsyncCallbackManager

        config = ensure_config(config)
        inheritable_metadata = {
            **(config.get("metadata") or {}),
            **self._get_ls_params(**kwargs),
        }
        callback_manager = AsyncCallbackManager.configure(
            config.get("callbacks"),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=config.get("tags"),
            local_tags=self.tags,
            inheritable_metadata=inheritable_metadata,
            local_metadata=self.metadata,
        )
        run_manager = await callback_manager.on_retriever_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=kwargs.pop("run_id", None),
        )
        try:
            _kwargs = kwargs if self._expects_other_args else {}
            if self._new_arg_supported:
                result = await self._aget_relevant_documents(
                    input, run_manager=run_manager, **_kwargs
                )
            else:
                result = await self._aget_relevant_documents(input, **_kwargs)
        except Exception as e:
            await run_manager.on_retriever_error(e)
            raise e
        else:
            await run_manager.on_retriever_end(
                result,
            )
            return result

    @abstractmethod
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query.

        Args:
            query: String to find relevant documents for.
            run_manager: The callback handler to use.
        Returns:
            List of relevant documents.
        """

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query.

        Args:
            query: String to find relevant documents for
            run_manager: The callback handler to use
        Returns:
            List of relevant documents
        """
        return await run_in_executor(
            None,
            self._get_relevant_documents,
            query,
            run_manager=run_manager.get_sync(),
        )

    @deprecated(since="0.1.46", alternative="invoke", removal="1.0")
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

        Users should favor using `.invoke` or `.batch` rather than
        `get_relevant_documents directly`.

        Args:
            query: string to find relevant documents for.
            callbacks: Callback manager or list of callbacks. Defaults to None.
            tags: Optional list of tags associated with the retriever.
                These tags will be associated with each call to this retriever,
                and passed as arguments to the handlers defined in `callbacks`.
                Defaults to None.
            metadata: Optional metadata associated with the retriever.
                This metadata will be associated with each call to this retriever,
                and passed as arguments to the handlers defined in `callbacks`.
                Defaults to None.
            run_name: Optional name for the run. Defaults to None.
            kwargs: Additional arguments to pass to the retriever.

        Returns:
            List of relevant documents.
        """
        config: RunnableConfig = {}
        if callbacks:
            config["callbacks"] = callbacks
        if tags:
            config["tags"] = tags
        if metadata:
            config["metadata"] = metadata
        if run_name:
            config["run_name"] = run_name
        return self.invoke(query, config, **kwargs)

    @deprecated(since="0.1.46", alternative="ainvoke", removal="1.0")
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

        Users should favor using `.ainvoke` or `.abatch` rather than
        `aget_relevant_documents directly`.

        Args:
            query: string to find relevant documents for.
            callbacks: Callback manager or list of callbacks.
            tags: Optional list of tags associated with the retriever.
                These tags will be associated with each call to this retriever,
                and passed as arguments to the handlers defined in `callbacks`.
                Defaults to None.
            metadata: Optional metadata associated with the retriever.
                This metadata will be associated with each call to this retriever,
                and passed as arguments to the handlers defined in `callbacks`.
                Defaults to None.
            run_name: Optional name for the run. Defaults to None.
            kwargs: Additional arguments to pass to the retriever.

        Returns:
            List of relevant documents.
        """
        config: RunnableConfig = {}
        if callbacks:
            config["callbacks"] = callbacks
        if tags:
            config["tags"] = tags
        if metadata:
            config["metadata"] = metadata
        if run_name:
            config["run_name"] = run_name
        return await self.ainvoke(query, config, **kwargs)
