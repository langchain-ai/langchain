from typing import Any, Iterable, Iterator, Sequence, Union

from langchain.schema import BaseDocumentTransformer, Document
from langchain.schema.document import _deduplicate_in_order, _HashedDocument


class DedupDocumentTransformer(BaseDocumentTransformer):
    def __init__(
        self, by_content: bool = True, by_metadata: Union[bool, Sequence[str]] = False
    ) -> None:
        self.by_content = by_content
        self.by_metadata = by_metadata

    def _hashed_documents(
        self, documents: Iterable[Document]
    ) -> Iterator[_HashedDocument]:
        for doc in documents:
            page_content = doc.page_content if self.by_content else ""
            if isinstance(self.by_metadata, Sequence):
                metadata = {k: doc.metadata[k] for k in self.by_metadata}
            elif self.by_metadata:
                metadata = doc.metadata
            else:
                metadata = {}
            _doc = Document(page_content=page_content, metadata=metadata)
            yield _HashedDocument.from_document(_doc)

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        return list(_deduplicate_in_order(self._hashed_documents(documents)))

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        raise NotImplementedError
