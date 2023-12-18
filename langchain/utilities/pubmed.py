"""Util that calls Arxiv."""
import logging
import os
from typing import Any, Dict, List

from pydantic import BaseModel, Extra, root_validator

from langchain.schema import Document

logger = logging.getLogger(__name__)


class PubmedAPIWrapper(BaseModel):
    """Wrapper around Pubmed.

    To use, you should have the ``Bio`` python package installed.
    https://biopython.org/
    This wrapper will use the pubmed_lib API to conduct searches and
    fetch document summaries. By default, it will return the document summaries
    of the top-k results.
    It limits the Document content by doc_content_chars_max.
    Set doc_content_chars_max=None if you don't want to limit the content size.

    Parameters:
        top_k_results: number of the top-scored document used for the arxiv tool
        PUBMED_MAX_QUERY_LENGTH: the cut limit on the query used for the arxiv tool.
        load_max_docs: a limit to the number of loaded documents
        load_all_available_meta:
          if True: the `metadata` of the loaded Documents gets all available meta info
            (see https://dmaturana81.github.io/pubmed_lib/result.html),
          if False: the `metadata` gets only the most informative fields.
    """

    top_k_results: int = 3
    doc_content_chars_max: int = 4000

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        try:
            import pubmed_lib

            values["pubmed_search"] = pubmed_lib.Search(retmax=values['top_k_results'])

        except ImportError:
            raise ImportError(
                "Could not import pubmed_lib python package. "
                "Please install it with `pip install pubmed-lib==0.0.1`."
            )
        return values

    def run(self, query: str) -> str:
        """
        Run Arxiv search and get the article meta information.
        See https://dmaturana81.github.io/pubmed_lib/search.html
        See https://dmaturana81.github.io/pubmed_lib/result.html
        It uses only the most informative fields of article meta information.
        """
        try:
            results = self.pubmed_search.results( 
                query
            )
        except :
            return f"Pubmed exception !!"
        docs = [
        f"Published: {result.published}\nTitle: {result.title}\n"
        f"Abstract: {result.abstract}"
        for result in results
        ]
        if docs:
            return "\n\n".join(docs)[: self.doc_content_chars_max]
        else:
            return "No good Arxiv Result was found"

