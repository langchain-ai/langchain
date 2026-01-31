from __future__ import annotations

from typing import Iterator, List

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class DoclingLoader(BaseLoader):
    """Load PDF documents using Docling for layout-aware parsing.

    This loader supports:
    - High-quality text extraction
    - Table extraction (as Markdown)
    - Page-level metadata
    """

    def __init__(self, file_path: str, extract_tables: bool = True) -> None:
        self.file_path = file_path
        self.extract_tables = extract_tables

    def lazy_load(self) -> Iterator[Document]:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as e:
            raise ImportError(
                "Docling is required to use DoclingLoader. "
                "Install it with `pip install docling`."
            ) from e

        converter = DocumentConverter()
        result = converter.convert(self.file_path)

        for page in result.pages:
            base_metadata = {
                "source": self.file_path,
                "page": page.page_no,
            }

            # Main text
            if page.text and page.text.strip():
                yield Document(
                    page_content=page.text,
                    metadata=base_metadata,
                )

            # Tables (optional)
            if self.extract_tables:
                for table in page.tables:
                    yield Document(
                        page_content=table.to_markdown(),
                        metadata={
                            **base_metadata,
                            "content_type": "table",
                        },
                    )

    def load(self) -> List[Document]:
        return list(self.lazy_load())
