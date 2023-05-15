"""Util that calls bibtexparser."""
import logging
import re
from typing import Any, Dict, List, BinaryIO

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

    Parameters:
        load_max_docs: a limit to the number of loaded documents
          -1 by default, does not limit the number of loaded documents
        load_all_available_meta:
          if True: the `metadata` of the loaded Documents gets all available meta info
          if False: the `metadata` gets only the most informative fields.
        doc_content_chars_max: the maximum number of characters in the content of a Document
          4000 by default
        pdf_pattern: the regex pattern to find the pdf file name in the bibtex entry
          r'[^:]+\.pdf' by default
    """

    bibtexparser_client: Any  #: :meta private:
    bibtexparser_exceptions: Any  # :meta private:
    load_max_docs: int = -1
    load_all_available_meta: bool = False
    doc_content_chars_max: int = 4000
    pdf_pattern = r'[^:]+\.pdf'

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        try:
            import bibtexparser
            values["load_file"] = bibtexparser.load
            values["bibtexparser_exceptions"] = (
                bibtexparser.bibtexexpression.BibtexExpression().ParseException,
            )
        except ImportError:
            raise ValueError(
                "Could not import bibtexparser python package. "
                "Please install it with `pip install bibtexparser`."
            )
        return values
    
    def _meta_str(self, entry: Any) -> str:
        """
        Return the meta information of an article in string format.
        """
        if entry.get('ENTRYTYPE','') == "article":
            return self._meta_str_article(entry)
        else:
            return self._meta_str_other(entry)
    
    def _meta_str_article(self, entry: Any) -> str:
        metadata = self._meta_article(entry)
        return '\n'.join([f"{key}: {value}" for key, value in metadata.items()])
        
    def _meta_article(self, entry: Any) -> str:
        return {
            "Published": entry.get('year', 'Published date unknown'),
            "Title": entry.get('title', ''),
            "Authors": entry.get('author', 'Unknown authors'),
            "Journal": entry.get('journal', 'Unknown journal'),
            "Summary": entry.get('abstract', 'No abstract available'),
            "Keywords": entry.get('keywords', ''),
            "URL": entry.get('url', None) if entry.get('url', None) else f'https://doi.org/{entry.get("doi", "")}' if entry.get('doi', None) else 'No URL available'
        }

    def _meta_str_other(self, entry: Any) -> str:
        # not implemented
        return ""

    def run(self, file: BinaryIO) -> str:
        """
        Load bibtex file using bibtexparser and get the article meta information.
        See https://bibtexparser.readthedocs.io/en/master/
        It uses only the most informative fields of article meta information.
        """
        print("Running bibtexparser wrapper")
        try:
            docs = [
                self._meta_str(entry)
                for entry in self.load_file(file).entries[:self.load_max_docs]
            ]
            return (
                "\n\n".join(docs)[: self.doc_content_chars_max]
                if docs
                else "No good bibtex information found. Check your bibtex file."
            )
        except self.bibtexparser_exceptions as ex:
            return f"Bibtexparser exception: {ex}"

    def load(self, file: str) -> List[Document]:
        """
        Load bibtex file using bibtexparser and get the article texts plus the article meta information.
        See https://bibtexparser.readthedocs.io/en/master/

        Returns: a list of documents with the document.page_content in text format

        """
        print("Running bibtexparser wrapper LOADER")
        try:
            import fitz
        except ImportError:
            raise ValueError(
                "PyMuPDF package not found, please install it with "
                "`pip install pymupdf`"
            )

        try:
            docs: List[Document] = []
            pdf_regex = re.compile(self.pdf_pattern)
            for entry in self.load_file(file).entries[:self.load_max_docs]:
                try:
                    # regex is usefull for Zotero flavor bibtex files 
                    doc_file_names = pdf_regex.findall(entry.get('file', ''))
                    text: str = ''
                    for doc_file_name in doc_file_names:
                        with fitz.open(doc_file_name) as doc_file:
                            text += "\n".join(page.get_text() for page in doc_file)
                    if not text:
                            # if nothing retrieved, just use the abstract as content
                            text: str = entry.get('abstract', '')

                    add_meta = (
                                {
                                    "entry_id": entry.get('ID', ''),
                                    "note": entry.get('note', ''),
                                    "doi": entry.get('doi',''),
                                }
                                if self.load_all_available_meta
                                else {}
                            )
                    

                    metadata = {**self._meta_article(entry), **add_meta}
                    doc = Document(
                        page_content=text[: self.doc_content_chars_max],
                        metadata = metadata
                    )
                    docs.append(doc)
                except FileNotFoundError as f_ex:
                    logger.debug(f_ex)
            return docs
        except self.bibtexparser_exceptions as ex:
            logger.debug("Error on bibtexparser: %s", ex)
            return []