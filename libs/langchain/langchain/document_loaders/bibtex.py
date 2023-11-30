import logging
import re
from pathlib import Path
from typing import Any, Iterator, List, Mapping, Optional

from langchain_core.documents import Document

from langchain.document_loaders.base import BaseLoader
from langchain.utilities.bibtex import BibtexparserWrapper

logger = logging.getLogger(__name__)


class BibtexLoader(BaseLoader):
    """Load a `bibtex` file.

    Each document represents one entry from the bibtex file.

    If a PDF file is present in the `file` bibtex field, the original PDF
    is loaded into the document text. If no such file entry is present,
    the `abstract` field is used instead.
    """

    def __init__(
        self,
        file_path: str,
        *,
        parser: Optional[BibtexparserWrapper] = None,
        max_docs: Optional[int] = None,
        max_content_chars: Optional[int] = 4_000,
        load_extra_metadata: bool = False,
        file_pattern: str = r"[^:]+\.pdf",
    ):
        """Initialize the BibtexLoader.

        Args:
            file_path: Path to the bibtex file.
            parser: The parser to use. If None, a default parser is used.
            max_docs: Max number of associated documents to load. Use -1 means
                           no limit.
            max_content_chars: Maximum number of characters to load from the PDF.
            load_extra_metadata: Whether to load extra metadata from the PDF.
            file_pattern: Regex pattern to match the file name in the bibtex.
        """
        self.file_path = file_path
        self.parser = parser or BibtexparserWrapper()
        self.max_docs = max_docs
        self.max_content_chars = max_content_chars
        self.load_extra_metadata = load_extra_metadata
        self.file_regex = re.compile(file_pattern)

    def _load_entry(self, entry: Mapping[str, Any]) -> Optional[Document]:
        import fitz

        parent_dir = Path(self.file_path).parent
        # regex is useful for Zotero flavor bibtex files
        file_names = self.file_regex.findall(entry.get("file", ""))
        if not file_names:
            return None
        texts: List[str] = []
        for file_name in file_names:
            try:
                with fitz.open(parent_dir / file_name) as f:
                    texts.extend(page.get_text() for page in f)
            except FileNotFoundError as e:
                logger.debug(e)
        content = "\n".join(texts) or entry.get("abstract", "")
        if self.max_content_chars:
            content = content[: self.max_content_chars]
        metadata = self.parser.get_metadata(entry, load_extra=self.load_extra_metadata)
        return Document(
            page_content=content,
            metadata=metadata,
        )

    def lazy_load(self) -> Iterator[Document]:
        """Load bibtex file using bibtexparser and get the article texts plus the
        article metadata.
        See https://bibtexparser.readthedocs.io/en/master/

        Returns:
            a list of documents with the document.page_content in text format
        """
        try:
            import fitz  # noqa: F401
        except ImportError:
            raise ImportError(
                "PyMuPDF package not found, please install it with "
                "`pip install pymupdf`"
            )

        entries = self.parser.load_bibtex_entries(self.file_path)
        if self.max_docs:
            entries = entries[: self.max_docs]
        for entry in entries:
            doc = self._load_entry(entry)
            if doc:
                yield doc

    def load(self) -> List[Document]:
        """Load bibtex file documents from the given bibtex file path.

        See https://bibtexparser.readthedocs.io/en/master/

        Args:
            file_path: the path to the bibtex file

        Returns:
            a list of documents with the document.page_content in text format
        """
        return list(self.lazy_load())
