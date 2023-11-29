from typing import Any, Iterator, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob


class DocumentIntelligenceParser(BaseBlobParser):
    """Loads a PDF with Azure Document Intelligence
    (formerly Forms Recognizer)."""

    def __init__(
        self,
        api_endpoint: str,
        api_key: str,
        api_version: Optional[str] = None,
        api_model: str = "prebuilt-layout",
        mode: str = "markdown",
    ):
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential

        kwargs = {"headers": {"x-ms-useragent": "langchain-parser/1.0.0"}}
        if api_version is not None:
            kwargs["api_version"] = api_version
        self.client = DocumentIntelligenceClient(
            endpoint=api_endpoint,
            credential=AzureKeyCredential(api_key),
            **kwargs,
        )
        self.api_model = api_model
        self.mode = mode
        assert self.mode in ["single", "page", "object", "markdown"]

    def _generate_docs_page(self, blob: Blob, result: Any) -> Iterator[Document]:
        for p in result.pages:
            content = " ".join([line.content for line in p.lines])

            d = Document(
                page_content=content,
                metadata={
                    "source": blob.source,
                    "page": p.page_number,
                },
            )
            yield d

    def _generate_docs_single(self, blob: Blob, result: Any) -> Iterator[Document]:
        yield Document(page_content=result.content, metadata={})

    def _generate_docs_object(self, blob: Blob, result: Any) -> Iterator[Document]:
        # record relationship between page id and span offset
        page_offset = []
        for page in result.pages:
            # assume that spans only contain 1 element, to double check
            page_offset.append(page.spans[0]["offset"])

        # paragraph
        # warning: paragraph content is overlapping with table content
        for para in result.paragraphs:
            yield Document(
                page_content=para.content,
                metadata={
                    "role": para.role,
                    "page": para.bounding_regions[0].page_number,
                    "bounding_box": para.bounding_regions[0].polygon,
                    "type": "paragraph",
                },
            )

        # table
        for table in result.tables:
            yield Document(
                page_content=table.cells,  # json object
                metadata={
                    "footnote": table.footnotes,
                    "caption": table.caption,
                    "page": para.bounding_regions[0].page_number,
                    "bounding_box": para.bounding_regions[0].polygon,
                    "row_count": table.row_count,
                    "column_count": table.column_count,
                    "type": "table",
                },
            )

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""

        with blob.as_bytes_io() as file_obj:
            poller = self.client.begin_analyze_document(
                self.api_model,
                file_obj,
                content_type="application/octet-stream",
                output_content_format="markdown" if self.mode == "markdown" else "text",
            )
            result = poller.result()

            if self.mode in ["single", "markdown"]:
                yield from self._generate_docs_single(blob, result)
            elif self.mode == ["page"]:
                yield from self._generate_docs_page(blob, result)
            else:
                yield from self._generate_docs_object(blob, result)
