"""Util that calls bibtexparser."""
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.schema import Document

logger = logging.getLogger(__name__)


class BibtexparserWrapper(BaseModel):
    """Wrapper around bibtexparser.

    To use, you should have the ``bibtexparser`` python package installed.
    https://bibtexparser.readthedocs.io/en/master/

    This wrapper will use bibtexparser to load a collection of references from
    a bibtex file and fetch document summaries.
    It limits the Document content by doc_content_chars_max.
    Set doc_content_chars_max=None if you don't want to limit the content size.
    """

    bibtexparser_client: Any  #: :meta private:

    load_max_docs: int = -1
    """Max number of associated documents to load. Use -1 means no limit."""

    load_all_available_meta: bool = False
    """Load all available metadata or restrict to most informative fields."""

    max_content_chars: Optional[int] = 4000
    """The maximum number of characters in the content of a Document 4000 by default"""

    file_pattern = r"[^:]+\.pdf"
    r"""File pattern in the bibtex entry to decide which files to load r'[^:]+\.pdf'"""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        try:
            import bibtexparser  # noqa
        except ImportError:
            raise ValueError(
                "Could not import bibtexparser python package. "
                "Please install it with `pip install bibtexparser`."
            )

        return values

    def _meta_str(self, entry: Mapping[str, Any]) -> str:
        """Return the meta information of an article in string format."""
        metadata = self._get_metadata(entry)
        return "\n".join([f"{key}: {value}" for key, value in metadata.items()])

    def load_bibtex_entries(self, path: str) -> List[Dict[str, Any]]:
        """Load bibtex entries from the bibtex file at the given path."""
        import bibtexparser
        from bibtexparser.bibtexexpression import BibtexExpression

        try:
            with open(path) as file:
                entries = bibtexparser.load(file).entries[: self.load_max_docs]
        except BibtexExpression.ParseException as e:
            entries = []
            logger.debug("Error on bibtexparser: %s", e)
        return entries

    def _get_metadata(self, entry: Mapping[str, Any]) -> Dict[str, Any]:
        """Get metadata for the given entry."""
        meta = {
            "id": entry.get("ID", ""),
            "published": entry.get("year", "Published date unknown"),
            "title": entry.get("title", "Unknown"),
            "publication": entry.get("journal")
            or entry.get("booktitle")
            or "Unknown publication",
            "authors": entry.get("author", "Unknown authors"),
            "summary": entry.get("abstract", "No abstract available"),
            "url": entry.get("url", None) or f'https://doi.org/{entry.get("doi", "")}'
            if entry.get("doi", None)
            else "",
        }
        optional_fields = [
            "annote",
            "booktitle",
            "editor",
            "howpublished",
            "journal",
            "keywords",
            "note",
            "organization",
            "publisher",
            "school",
            "series",
            "type",
            "doi",
            "issn",
            "isbn",
        ]
        if self.load_all_available_meta:
            for field in optional_fields:
                if field in entry:
                    meta[field] = entry[field]
        return meta

    def run(self, file_path: str) -> str:
        """Load bibtex file using bibtexparser and get the article meta information.

        See https://bibtexparser.readthedocs.io/en/master/
        It uses only the most informative fields of article meta information.
        """
        docs = [self._meta_str(entry) for entry in self.load_bibtex_entries(file_path)]
        return (
            "\n\n".join(docs)[: self.max_content_chars]
            if docs
            else "No good bibtex information found. Check your bibtex file."
        )

    def lazy_load(self, file_path: str) -> Iterator[Document]:
        """Load bibtex file using bibtexparser and get the article texts plus the

        article metadata.

        See https://bibtexparser.readthedocs.io/en/master/

        Returns:
            a list of documents with the document.page_content in text format
        """
        try:
            import fitz
        except ImportError:
            raise ValueError(
                "PyMuPDF package not found, please install it with "
                "`pip install pymupdf`"
            )

        pdf_regex = re.compile(self.file_pattern)
        entries = self.load_bibtex_entries(file_path)

        for entry in entries:
            try:
                # regex is useful for Zotero flavor bibtex files
                filenames = pdf_regex.findall(entry.get("file", ""))
                file_paths = [Path(file_path).parent / name for name in filenames]
                text: str = ""
                for file_path in file_paths:
                    with fitz.open(file_path) as doc_file:
                        text += "\n".join(page.get_text() for page in doc_file)
                if not text:
                    # if nothing retrieved, just use the abstract as content
                    text = entry.get("abstract", "")

                yield Document(
                    page_content=text[: self.max_content_chars],
                    metadata=self._get_metadata(entry),
                )
            except FileNotFoundError as f_ex:
                logger.debug(f_ex)

    def load(self, file_path: str) -> List[Document]:
        """Load bibtex file documents from the given bibtex file path.

        See https://bibtexparser.readthedocs.io/en/master/

        Args:
            file_path: the path to the bibtex file

        Returns:
            a list of documents with the document.page_content in text format
        """
        return list(self.lazy_load(file_path))
