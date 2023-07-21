"""Util that calls Arxiv."""
import logging
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, root_validator

from langchain.schema import Document

logger = logging.getLogger(__name__)


class ArxivAPIWrapper(BaseModel):
    """Wrapper around ArxivAPI.

    To use, you should have the ``arxiv`` python package installed.
    https://lukasschwab.me/arxiv.py/index.html
    This wrapper will use the Arxiv API to conduct searches and
    fetch document summaries. By default, it will return the document summaries
    of the top-k results.
    It limits the Document content by doc_content_chars_max.
    Set doc_content_chars_max=None if you don't want to limit the content size.

    Parameters:
        top_k_results: number of the top-scored document used for the arxiv tool
        ARXIV_MAX_QUERY_LENGTH: the cut limit on the query used for the arxiv tool.
        load_max_docs: a limit to the number of loaded documents
        load_all_available_meta:
          if True: the `metadata` of the loaded Documents gets all available meta info
            (see https://lukasschwab.me/arxiv.py/index.html#Result),
          if False: the `metadata` gets only the most informative fields.

    """

    arxiv_search: Any  #: :meta private:
    arxiv_exceptions: Any  # :meta private:
    top_k_results: int = 3
    ARXIV_MAX_QUERY_LENGTH = 300
    load_max_docs: int = 100
    load_all_available_meta: bool = False
    doc_content_chars_max: Optional[int] = 4000

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        try:
            import arxiv

            values["arxiv_search"] = arxiv.Search
            values["arxiv_exceptions"] = (
                arxiv.ArxivError,
                arxiv.UnexpectedEmptyPageError,
                arxiv.HTTPError,
            )
            values["arxiv_result"] = arxiv.Result
        except ImportError:
            raise ImportError(
                "Could not import arxiv python package. "
                "Please install it with `pip install arxiv`."
            )
        return values

    def run(self, query: str) -> str:
        """
        Run Arxiv search and get the article meta information.
        See https://lukasschwab.me/arxiv.py/index.html#Search
        See https://lukasschwab.me/arxiv.py/index.html#Result
        It uses only the most informative fields of article meta information.
        """
        try:
            results = self.arxiv_search(  # type: ignore
                query[: self.ARXIV_MAX_QUERY_LENGTH], max_results=self.top_k_results
            ).results()
        except self.arxiv_exceptions as ex:
            return f"Arxiv exception: {ex}"
        docs = [
            f"Published: {result.updated.date()}\nTitle: {result.title}\n"
            f"Authors: {', '.join(a.name for a in result.authors)}\n"
            f"Summary: {result.summary}"
            for result in results
        ]
        if docs:
            return "\n\n".join(docs)[: self.doc_content_chars_max]
        else:
            return "No good Arxiv Result was found"

    def load(self, query: str) -> List[Document]:
        """
        Run Arxiv search and get the article texts plus the article meta information.
        See https://lukasschwab.me/arxiv.py/index.html#Search

        Returns: a list of documents with the document.page_content in text format

        """
        try:
            import fitz
        except ImportError:
            raise ImportError(
                "PyMuPDF package not found, please install it with "
                "`pip install pymupdf`"
            )

        try:
            results = self.arxiv_search(  # type: ignore
                query[: self.ARXIV_MAX_QUERY_LENGTH], max_results=self.load_max_docs
            ).results()
        except self.arxiv_exceptions as ex:
            logger.debug("Error on arxiv: %s", ex)
            return []

        docs: List[Document] = []
        for result in results:
            try:
                doc_file_name: str = result.download_pdf()
                with fitz.open(doc_file_name) as doc_file:
                    text: str = "".join(page.get_text() for page in doc_file)
            except FileNotFoundError as f_ex:
                logger.debug(f_ex)
                continue
            if self.load_all_available_meta:
                extra_metadata = {
                    "entry_id": result.entry_id,
                    "published_first_time": str(result.published.date()),
                    "comment": result.comment,
                    "journal_ref": result.journal_ref,
                    "doi": result.doi,
                    "primary_category": result.primary_category,
                    "categories": result.categories,
                    "links": [link.href for link in result.links],
                }
            else:
                extra_metadata = {}
            metadata = {
                "Published": str(result.updated.date()),
                "Title": result.title,
                "Authors": ", ".join(a.name for a in result.authors),
                "Summary": result.summary,
                **extra_metadata,
            }
            doc = Document(
                page_content=text[: self.doc_content_chars_max], metadata=metadata
            )
            docs.append(doc)
            os.remove(doc_file_name)
        return docs
