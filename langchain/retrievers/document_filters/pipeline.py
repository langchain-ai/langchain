"""Document compressor that uses a pipeline of other compressors."""
from typing import List

from langchain.retrievers.document_filters.base import (
    BaseDocumentCompressor,
    _RetrievedDocument,
)


class DocumentCompressorPipeline(BaseDocumentCompressor):
    """Document compressor that uses a pipeline of other compressors."""

    compressors: List[BaseDocumentCompressor]
    """List of document filters that are chained together and run in sequence."""

    def compress_documents(
        self, documents: List[_RetrievedDocument], query: str
    ) -> List[_RetrievedDocument]:
        """Compress retrieved documents given the query context."""
        for _compressor in self.compressors:
            documents = _compressor.compress_documents(documents, query)
        return documents

    async def acompress_documents(
        self, documents: List[_RetrievedDocument], query: str
    ) -> List[_RetrievedDocument]:
        """Compress retrieved documents given the query context."""
        raise NotImplementedError
