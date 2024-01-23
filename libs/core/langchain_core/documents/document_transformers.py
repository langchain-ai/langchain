import itertools
import sys
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Iterable,
    Iterator,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    no_type_check,
)

from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.runnables.base import RunnableParallel, coerce_to_runnable

from .runnable_document_transformer import (
    _LEGACY,
    RunnableGeneratorDocumentTransformer,
)

if sys.version_info.major > 3 or sys.version_info.minor >= 12:
    from itertools import batched  # type: ignore[attr-defined]
else:
    T = TypeVar("T")  # Only in python 3.12

    def batched(iterable: Iterable[T], n: int) -> Iterator[Tuple[T, ...]]:
        if n < 1:
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(itertools.islice(it, n)):
            yield batch


BATCH_SIZE = 16


def _transform_documents_generator(
    documents: Iterator[Document],
    *,
    transformers: Sequence[RunnableGeneratorDocumentTransformer],
) -> Iterator[Document]:
    Input = Union[AsyncIterator[Document], Iterator[Document]]
    steps = {
        f"transform_documents_{i}": transformer
        for i, transformer in enumerate(transformers)
    }
    # Implementation when all transformers are compatible with Runnable
    for batch in batched(documents, BATCH_SIZE):
        result = RunnableParallel[Input](steps=steps).invoke(iter(batch))
        for chunk in result["steps"].values():
            yield chunk


async def abatched(iterable: AsyncIterable[T], n: int) -> AsyncIterable[Tuple[T, ...]]:
    if n < 1:
        raise ValueError("n must be at least one")
    batch = []
    async for t in iterable:
        batch.append(t)
        if len(batch) > n:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


class DocumentTransformers(RunnableGeneratorDocumentTransformer):
    """Document transformer that uses a list of Transformers.
    Take each input document, and apply all transformations present in the
    `transformers` attribute.
    """

    class Config:
        arbitrary_types_allowed = True

    if _LEGACY:
        transformers: Sequence[BaseDocumentTransformer]
    else:
        transformers: Sequence[  # type: ignore[no-redef]
            RunnableGeneratorDocumentTransformer
        ]
    """List of document transformer that are applied in parallel."""

    def __add__(
        self,
        other: RunnableGeneratorDocumentTransformer,
    ) -> "DocumentTransformers":
        """Compose this runnable with another object to create a RunnableSequence."""
        if isinstance(other, DocumentTransformers):
            return DocumentTransformers(
                transformers=list(other.transformers) + list(self.transformers),
            )
        else:
            if _LEGACY:
                return DocumentTransformers(
                    transformers=list(self.transformers) + [other]
                )
            else:
                return DocumentTransformers(
                    transformers=list(self.transformers)
                    + [cast(BaseDocumentTransformer, coerce_to_runnable(other))]
                )

    @no_type_check  # Bug in Mypy
    def lazy_transform_documents(
        self, documents: Iterator[Document], **kwargs: Any
    ) -> Iterator[Document]:
        """Transform an iterator of documents with the list of transformations.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            An iterator oftransformed Documents.
        """
        # Can be refactored to use parallelism
        if _LEGACY:
            # Implementation when all transformers are NOT compatible with Runnable
            # It's not compatible with lazy strategy. Load all documents and apply
            # all transformations.
            for batch in batched(documents, BATCH_SIZE):
                for t in self.transformers:
                    for doc in t.transform_documents(documents=list(batch)):
                        yield doc

        else:
            # Implementation with only the LCEL compatible transformers
            for batch in batched(documents, BATCH_SIZE):
                for t in self.transformers:
                    for doc in t.lazy_transform_documents(iter(batch)):
                        yield doc

    @no_type_check  # Bug in Mypy
    async def _alazy_transform_documents(
        self, documents: AsyncIterator[Document], **kwargs: Any
    ) -> AsyncIterator[Document]:
        """Asynchronously transform an iterator of documents with a list
        of transformations.

        Args:
            documents: An iterator of Documents to be transformed.

        Returns:
            An iterator of transformed Documents.
        """
        if _LEGACY:
            # Implementation when all transformers are NOT compatible with Runnable
            # It's not compatible with lazy strategy. Load all documents and apply
            # all transformations.
            async for batch in abatched(documents, BATCH_SIZE):
                for transformer in self.transformers:
                    for doc in await transformer.atransform_documents(batch):
                        yield doc
        else:
            # # Get a batch of documents, then apply each transformation by batch
            async for batch in abatched(documents, BATCH_SIZE):
                for transformer in self.transformers:
                    async for doc in transformer.alazy_transform_documents(iter(batch)):
                        yield doc
