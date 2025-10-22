"""**Retriever** class returns Documents given a text **query**.

It is more general than a vector store. A retriever does not need to be able to
store documents, only to return (or retrieve) it. Vector stores can be used as
the backbone of a retriever, but there are other types of retrievers as well.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from inspect import signature
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict
from typing_extensions import Self, TypedDict, override

from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager
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
    )

RetrieverInput = str
RetrieverOutput = list[Document]
RetrieverLike = Runnable[RetrieverInput, RetrieverOutput]
RetrieverOutputLike = Runnable[Any, RetrieverOutput]


class LangSmithRetrieverParams(TypedDict, total=False):
    """LangSmith parameters for tracing."""

    ls_retriever_name: str
    """Retriever name."""
    ls_vector_store_provider: str | None
    """Vector store provider."""
    ls_embedding_provider: str | None
    """Embedding provider."""
    ls_embedding_model: str | None
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

    ```python
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    class SimpleRetriever(BaseRetriever):
        docs: list[Document]
        k: int = 5

        def _get_relevant_documents(self, query: str) -> list[Document]:
            \"\"\"Return the first k documents from the list of documents\"\"\"
            return self.docs[:self.k]

        async def _aget_relevant_documents(self, query: str) -> list[Document]:
            \"\"\"(Optional) async native implementation.\"\"\"
            return self.docs[:self.k]
    ```

    Example: A simple retriever based on a scikit-learn vectorizer

    ```python
    from sklearn.metrics.pairwise import cosine_similarity


    class TFIDFRetriever(BaseRetriever, BaseModel):
        vectorizer: Any
        docs: list[Document]
        tfidf_array: Any
        k: int = 4

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(self, query: str) -> list[Document]:
            # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
            query_vec = self.vectorizer.transform([query])
            # Op -- (n_docs,1) -- Cosine Sim with each doc
            results = cosine_similarity(self.tfidf_array, query_vec).reshape((-1,))
            return [self.docs[i] for i in results.argsort()[-self.k :][::-1]]
    ```
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    _new_arg_supported: bool = False
    _expects_other_args: bool = False
    tags: list[str] | None = None
    """Optional list of tags associated with the retriever.
    These tags will be associated with each call to this retriever,
    and passed as arguments to the handlers defined in `callbacks`.
    You can use these to eg identify a specific instance of a retriever with its
    use case.
    """
    metadata: dict[str, Any] | None = None
    """Optional metadata associated with the retriever.
    This metadata will be associated with each call to this retriever,
    and passed as arguments to the handlers defined in `callbacks`.
    You can use these to eg identify a specific instance of a retriever with its
    use case.
    """

    @override
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        parameters = signature(cls._get_relevant_documents).parameters
        cls._new_arg_supported = parameters.get("run_manager") is not None
        if (
            not cls._new_arg_supported
            and cls._aget_relevant_documents == BaseRetriever._aget_relevant_documents
        ):
            # we need to tolerate no run_manager in _aget_relevant_documents signature
            async def _aget_relevant_documents(
                self: Self, query: str
            ) -> list[Document]:
                return await run_in_executor(None, self._get_relevant_documents, query)  # type: ignore[call-arg]

            cls._aget_relevant_documents = _aget_relevant_documents  # type: ignore[assignment]

        # If a V1 retriever broke the interface and expects additional arguments
        cls._expects_other_args = (
            len(set(parameters.keys()) - {"self", "query", "run_manager"}) > 0
        )

    def _get_ls_params(self, **_kwargs: Any) -> LangSmithRetrieverParams:
        """Get standard params for tracing."""
        default_retriever_name = self.get_name()
        if default_retriever_name.startswith("Retriever"):
            default_retriever_name = default_retriever_name[9:]
        elif default_retriever_name.endswith("Retriever"):
            default_retriever_name = default_retriever_name[:-9]
        default_retriever_name = default_retriever_name.lower()

        return LangSmithRetrieverParams(ls_retriever_name=default_retriever_name)

    @override
    def invoke(
        self, input: str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> list[Document]:
        """Invoke the retriever to get relevant documents.

        Main entry point for synchronous retriever invocations.

        Args:
            input: The query string.
            config: Configuration for the retriever.
            **kwargs: Additional arguments to pass to the retriever.

        Returns:
            List of relevant documents.

        Examples:
        ```python
        retriever.invoke("query")
        ```
        """
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
            kwargs_ = kwargs if self._expects_other_args else {}
            if self._new_arg_supported:
                result = self._get_relevant_documents(
                    input, run_manager=run_manager, **kwargs_
                )
            else:
                result = self._get_relevant_documents(input, **kwargs_)
        except Exception as e:
            run_manager.on_retriever_error(e)
            raise
        else:
            run_manager.on_retriever_end(
                result,
            )
            return result

    @override
    async def ainvoke(
        self,
        input: str,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Asynchronously invoke the retriever to get relevant documents.

        Main entry point for asynchronous retriever invocations.

        Args:
            input: The query string.
            config: Configuration for the retriever.
            **kwargs: Additional arguments to pass to the retriever.

        Returns:
            List of relevant documents.

        Examples:
        ```python
        await retriever.ainvoke("query")
        ```
        """
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
            kwargs_ = kwargs if self._expects_other_args else {}
            if self._new_arg_supported:
                result = await self._aget_relevant_documents(
                    input, run_manager=run_manager, **kwargs_
                )
            else:
                result = await self._aget_relevant_documents(input, **kwargs_)
        except Exception as e:
            await run_manager.on_retriever_error(e)
            raise
        else:
            await run_manager.on_retriever_end(
                result,
            )
            return result

    @abstractmethod
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Get documents relevant to a query.

        Args:
            query: String to find relevant documents for.
            run_manager: The callback handler to use.

        Returns:
            List of relevant documents.
        """

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
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
