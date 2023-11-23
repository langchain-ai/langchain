"""
Some very simple transformer (lower, upper), lazy and compatible with LCEL.
"""
import copy
from typing import Any, AsyncIterator, Callable, Iterator

from langchain.document_transformers.runnable_document_transformer import (
    RunnableGeneratorDocumentTransformer,
)
from langchain.schema import Document


class _LazyTransformer(RunnableGeneratorDocumentTransformer):
    """Implementation of a runnable transformer, with lazy transformation"""

    fn: Callable[[Any], str]

    def lazy_transform_documents(
        self, documents: Iterator[Document], **kwargs: Any
    ) -> Iterator[Document]:
        return (
            Document(
                page_content=self.fn(doc.page_content),
                metadata=copy.deepcopy(doc.metadata),
            )
            for doc in documents
        )

    async def _alazy_transform_documents(  # type:ignore
        self, documents: AsyncIterator[Document], **kwargs: Any
    ) -> AsyncIterator[Document]:
        async for doc in documents:
            yield Document(
                page_content=self.fn(doc.page_content),
                metadata=copy.deepcopy(doc.metadata),
            )


class LowerLazyTransformer(_LazyTransformer):
    def __init__(self, **kwargs: Any):
        super().__init__(fn=str.lower, **kwargs)


class UpperLazyTransformer(_LazyTransformer):
    def __init__(self, **kwargs: Any):
        super().__init__(fn=str.upper, **kwargs)
