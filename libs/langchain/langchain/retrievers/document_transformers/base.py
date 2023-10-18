from typing import List, Sequence, Any
from xml.dom.minidom import Document

from langchain.schema import BaseDocumentTransformer

class DocumentTransformerPipeline(BaseDocumentTransformer):
    """Document transformer that uses a pipeline of Transformers."""

    def __init__(self,
                 transformers: Sequence[BaseDocumentTransformer]
                 ):
        self.transformers = transformers

    """List of document transformer that are applied in sequence."""

    def transform_documents(
            self, documents: Sequence[Document],
            **kwargs: Any
    ) -> Sequence[Document]:
        """Transform a list of documents."""
        all_documents = []
        for _transformer in self.transformers:
            all_documents.extend(_transformer.transform_documents(documents, **kwargs))
        return all_documents

    async def atransform_documents(
            self,
            documents: Sequence[Document],
            **kwargs: Any
    ) -> Sequence[Document]:
        """Compress retrieved documents given the query context."""
        all_documents = []
        for _transformer in self.transformers:
            all_documents.extend(await _transformer.atransform_documents(documents))
        return documents
