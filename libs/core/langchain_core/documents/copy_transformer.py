import copy
from typing import Any, AsyncIterator, Iterator

# Note: Import directly from langchain_core is not stable and generate some errors
from langchain_core.documents import Document

from langchain_core.documents.runnable_document_transformer import (
    RunnableGeneratorDocumentTransformer,
)


class CopyDocumentTransformer(RunnableGeneratorDocumentTransformer):
    """
    This class is a transparent transformation. The input is just sent back to
    the output.
    This is useful when you want to apply several transformations to a list of
    documents, and you also want to keep a copy of the original.
    With a DocumentTransformers, it's possible to do this in LCEL syntax.
    """
    def lazy_transform_documents(
        self, documents: Iterator[Document], **kwargs: Any
    ) -> Iterator[Document]:
        yield from (copy.deepcopy(doc) for doc in documents)

    async def _alazy_transform_documents(  # type: ignore
        self, documents: AsyncIterator[Document], **kwargs: Any
    ) -> AsyncIterator[Document]:
        async for doc in documents:
            yield copy.deepcopy(doc)
