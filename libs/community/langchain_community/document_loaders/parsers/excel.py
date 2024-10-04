from typing import Iterator, Sequence

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob


class ExcelParser(BaseBlobParser):
    """Parse the Microsoft Excel documents from a blob."""

    def post_process_unstructured_elements(
        self, elements: Sequence[str], source: str
    ) -> Iterator[Document]:
        try:
            import htmltabletomd
        except ImportError as e:
            raise ImportError(
                "Could not import htmltabletomd, please install with `pip install "
                "htmltabletomd`."
            ) from e
        last_page = None
        for element in elements:
            if element.metadata.page_number != last_page:
                if last_page:
                    metadata = {"source": source, "title": element.metadata.page_name}  # type: ignore[attr-defined]
                    yield Document(page_content=page_text, metadata=metadata)
                page_text = f"# {element.metadata.page_name}\n"
                last_page = element.metadata.page_number

            if type(element).__name__ == "Title":
                page_text += f"## {element.text}\n"
            elif type(element).__name__ == "Table":
                page_text += (
                    htmltabletomd.convert_table(element.metadata.text_as_html) + "\n"
                )
            else:
                page_text += f"{element.text}\n"
        # yield last page
        metadata = {"source": source, "title": element.metadata.page_name}  # type: ignore[attr-defined]
        yield Document(page_content=page_text, metadata=metadata)

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Parse a Microsoft Excel document into the Document iterator.

        Args:
            blob: The blob to parse.

        Returns: An iterator of Documents.

        """
        try:
            from unstructured.partition.xlsx import partition_xlsx
        except ImportError as e:
            raise ImportError(
                "Could not import unstructured, please install with `pip install "
                "unstructured`."
            ) from e

        # TODO: add pre-2007 .xls support
        mime_type_parser = {
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": (
                partition_xlsx
            ),
        }
        if blob.mimetype not in (  # type: ignore[attr-defined]
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ):
            raise ValueError("This blob type is not supported for this parser.")
        with blob.as_bytes_io() as excel_document:  # type: ignore[attr-defined]
            elements = mime_type_parser[blob.mimetype](file=excel_document)  # type: ignore[attr-defined]
            yield from self.post_process_unstructured_elements(elements, blob.source)
