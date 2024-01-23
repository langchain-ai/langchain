import asyncio
import threading
import time
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
    no_type_check,
)

from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.base import coerce_to_runnable

if TYPE_CHECKING:
    from .document_transformers import DocumentTransformers
else:
    DocumentTransformers = Any

_LEGACY = True  # Use legacy langchain transformer interface

"""
    We propose an alternative way of making transformers compatible with LCEL.
    The first keeps the current protocol (RunnableDocumentTransformer).
    The second takes advantage of this to propose
    a lazy approach to transformations (RunnableGeneratorDocumentTransformer).
    It's better for the memory, pipeline, etc.
    
    Now, it's possible to create a pipeline of transformer like:
    Example:
    ..code - block:: python
    class UpperTransformer(RunnableGeneratorDocumentTransformer):
        def lazy_transform_documents(
                self,
                documents: Iterator[Document],
                **kwargs: Any
        ) -> Iterator[Document]:
            ...

        async def alazy_transform_documents(
                self,
                documents: Union[AsyncIterator[Document],Iterator[Document]],
                **kwargs: Any
        ) -> AsyncIterator[Document]:
            ...

    runnable = (UpperTransformer() | LowerTransformer())
    result = list(runnable.invoke(documents))
"""

T = TypeVar("T")


async def to_async_iterator(iterator: Iterator[T]) -> AsyncIterator[T]:
    """Convert an iterable to an async iterator."""
    for item in iterator:
        yield item


_DONE = ""
_TIMEOUT = 1


def to_sync_iterator(async_iterable: AsyncIterator[T], maxsize: int = 0) -> Iterator[T]:
    def _run_coroutine(
        loop: asyncio.AbstractEventLoop,
        async_iterable: AsyncIterator[T],
        queue: asyncio.Queue,
    ) -> None:
        async def _consume_async_iterable(
            async_iterable: AsyncIterator[T], queue: asyncio.Queue
        ) -> None:
            async for x in async_iterable:
                await queue.put(x)

            await queue.put(_DONE)

        loop.run_until_complete(_consume_async_iterable(async_iterable, queue))

    queue: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)
    loop = asyncio.new_event_loop()

    t = threading.Thread(target=_run_coroutine, args=(loop, async_iterable, queue))
    t.daemon = True
    t.start()

    while True:
        if not queue.empty():
            x = queue.get_nowait()

            if x is _DONE:
                break
            else:
                yield x
        else:
            time.sleep(_TIMEOUT)

    t.join()


Input = Union[AsyncIterator[Document], Iterator[Document], Sequence[Document]]
Output = Union[AsyncIterator[Document], Iterator[Document]]


class RunnableGeneratorDocumentTransformer(
    Runnable[
        Input,
        Output,
    ],
    BaseDocumentTransformer,
    BaseModel,  # Pydantic v2
    ABC,
):
    """
    Runnable Document Transformer with lazy transformation.

    You can compose a list of transformations with the *or* operator.

        .. code-block:: python

            runnable=TokenTextSplitter(...) | CharacterTextSplitter(...)
            docs = list(runnable.invoke(input_docs))

    To apply multiple transformations with the same input, use the *plus* operator.

        .. code-block:: python

            runnable=TokenTextSplitter(...) + CharacterTextSplitter(...)
            docs = list(runnable.invoke(input_docs))

    and, you can combine these two operator


        .. code-block:: python

            runnable=(
                (TokenTextSplitter(...) | CharacterTextSplitter(...) ) +
                CharacterTextSplitter(...))
            docs = list(runnable.invoke(input_docs))

    > This class is a transition class for proposing lazy transformers,
    > compatible with LCEL.
    > Later, it can be integrated into BaseDocumentTransformer
    >if you agree to add a lazy approach to transformations.
    >All subclass of BaseDocumentTransformer must be updated to be compatible with this.
    """

    def __add__(
        self,
        other: "RunnableGeneratorDocumentTransformer",
    ) -> "DocumentTransformers":
        """Compose this runnable with another object to create a RunnableSequence."""
        # return RunnableSequence(first=self, last=coerce_to_runnable(other))
        from .document_transformers import DocumentTransformers

        if isinstance(other, DocumentTransformers):
            return DocumentTransformers(
                transformers=list(other.transformers) + [self],
            )
        else:
            if _LEGACY:
                return DocumentTransformers(transformers=[self, other])
            else:
                return DocumentTransformers(
                    transformers=[
                        self,
                        cast(BaseDocumentTransformer, coerce_to_runnable(other)),
                    ]
                )

    if _LEGACY:

        def __radd__(
            self,
            other: DocumentTransformers,
        ) -> "RunnableGeneratorDocumentTransformer":
            return self.__add__(other)

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        # Convert lazy to classical transformation
        return list(self.lazy_transform_documents(iter(documents), **kwargs))

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        return [
            doc
            async for doc in self.alazy_transform_documents(iter(documents), **kwargs)
        ]

    @abstractmethod
    def lazy_transform_documents(
        self, documents: Iterator[Document], **kwargs: Any
    ) -> Iterator[Document]:
        """Transform an iterator of documents.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            An iterator oftransformed Documents.
        """
        raise NotImplementedError()

    @abstractmethod
    async def _alazy_transform_documents(  # type: ignore
        self, documents: AsyncIterator[Document], **kwargs: Any
    ) -> AsyncIterator[Document]:
        raise NotImplementedError()

    @no_type_check  # Bug in Mypy
    async def alazy_transform_documents(
        self,
        documents: Input,
        **kwargs: Any,
    ) -> AsyncIterator[Document]:
        """Asynchronously transform an iterator of documents.

        Args:
            documents: An iterator of Documents to be transformed.

        Returns:
            An iterator of transformed Documents.
        """
        if isinstance(documents, Iterable):
            async_documents = to_async_iterator(iter(documents))
        elif isinstance(documents, AsyncIterator):
            async_documents = documents
        else:
            async_documents = to_async_iterator(documents)

        async for doc in self._alazy_transform_documents(async_documents):
            yield doc

    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Output:
        if isinstance(input, AsyncIterator):
            raise ValueError("Use ainvoke() with async iterator")
        config = config or {}

        if hasattr(self, "lazy_transform_documents"):
            iterator = iter(input) if isinstance(input, Sequence) else input
            return self.lazy_transform_documents(iterator, **config)

        # Default implementation, without generator/iterator
        if isinstance(input, Sequence):
            return iter(self.transform_documents(input, **config))
        else:
            return iter(self.transform_documents(list(input), **config))

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        # # Default implementation, without generator
        config = config or {}

        if hasattr(self, "lazy_transform_documents"):
            iterator = iter(input) if isinstance(input, Sequence) else input
            return self.alazy_transform_documents(iterator, **config)

        # Default implementation, without generator/iterator
        if isinstance(input, Sequence):
            return iter(await self.atransform_documents(input, **config))
        elif isinstance(input, AsyncIterator):
            return iter(
                await self.atransform_documents([doc async for doc in input], **config)
            )
        else:
            return iter(await self.atransform_documents(list(input), **config))
