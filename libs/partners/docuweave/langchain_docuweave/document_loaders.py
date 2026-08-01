"""DocuWeave document loader — layout-aware PDF chunking for RAG pipelines."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Union

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class DocuWeaveLoader(BaseLoader):
    """Load a PDF with DocuWeave and return layout-aware LangChain Documents.

    DocuWeave reads font-size and bold signals from the PDF to reconstruct the
    heading hierarchy, then cuts chunks at section boundaries. Each returned
    ``Document`` carries rich metadata so retrieval can filter and explain
    results by section.

    Setup:

        .. code-block:: bash

            pip install -U langchain-docuweave

    Instantiate:

        .. code-block:: python

            from langchain_docuweave import DocuWeaveLoader

            loader = DocuWeaveLoader(file_path="paper.pdf", max_tokens=512)

    Load:

        .. code-block:: python

            docs = loader.load()
            print(docs[0].metadata["section_path"])
            # "3 Methods > 3.2 Experimental Setup"

    Lazy load:

        .. code-block:: python

            for doc in loader.lazy_load():
                print(doc.page_content[:80])
                print(doc.metadata["section_path"])

    Metadata fields returned per chunk:

    - ``source`` — absolute path to the source PDF
    - ``section_title`` — heading text of the section this chunk belongs to
    - ``section_path`` — full breadcrumb path, e.g.
      ``"3 Methods > 3.2 Experimental Setup"``
    - ``section_level`` — nesting depth (0 = top-level section)
    - ``page_start`` / ``page_end`` — 1-based page numbers
    - ``previous_chunk_id`` / ``next_chunk_id`` — linked-list pointers for
      context-window expansion at query time
    - ``hierarchy_confidence`` — ``[0.0, 1.0]`` score indicating how much
      usable heading structure was found. Values below 0.3 suggest a scanned
      or image-heavy PDF where chunking quality will be lower.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        max_tokens: int = 512,
    ) -> None:
        """Initialize DocuWeaveLoader.

        Args:
            file_path: Path to a local PDF file.
            max_tokens: Maximum tokens per chunk. DocuWeave uses the
                ``cl100k_base`` tiktoken encoding for counting.
                Defaults to 512.
        """
        self.file_path = str(file_path)
        self.max_tokens = max_tokens

    def lazy_load(self) -> Iterator[Document]:
        """Yield one :class:`~langchain_core.documents.Document` per chunk.

        Raises:
            ImportError: If ``docuweave`` is not installed.
            ValueError: If the file is missing or not a valid PDF.
        """
        try:
            from docuweave import parse
        except ImportError as exc:
            raise ImportError(
                "docuweave is required to use DocuWeaveLoader. "
                "Install it with: pip install docuweave"
            ) from exc

        doc = parse(self.file_path)
        confidence = doc.hierarchy_confidence

        for chunk in doc.iter_chunks(max_tokens=self.max_tokens):
            yield Document(
                page_content=chunk.get("text", ""),
                metadata={
                    "source": self.file_path,
                    "section_title": chunk.get("section_title", ""),
                    "section_path": chunk.get("section_path", ""),
                    "section_level": chunk.get("section_level", 0),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "previous_chunk_id": chunk.get("previous_chunk_id"),
                    "next_chunk_id": chunk.get("next_chunk_id"),
                    "hierarchy_confidence": confidence,
                },
            )

    def load(self) -> List[Document]:
        return list(self.lazy_load())
