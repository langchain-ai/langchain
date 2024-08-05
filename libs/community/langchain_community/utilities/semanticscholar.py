"""Utils for interacting with the Semantic Scholar API."""

import logging
from typing import Any, Dict, Optional

from langchain_core.pydantic_v1 import BaseModel, root_validator

logger = logging.getLogger(__name__)


class SemanticScholarAPIWrapper(BaseModel):
    """Wrapper around semanticscholar.org API.
    https://github.com/danielnsilva/semanticscholar

    You should have this library installed.

    `pip install semanticscholar`

    Semantic Scholar API can conduct searches and fetch document metadata
    like title, abstract, authors, etc.

    Attributes:
    top_k_results: number of the top-scored document used for the Semantic Scholar tool
    load_max_docs: a limit to the number of loaded documents

    Example:
    .. code-block:: python

    from langchain_community.utilities.semanticscholar import SemanticScholarAPIWrapper
    ss = SemanticScholarAPIWrapper(
        top_k_results = 3,
        load_max_docs = 3
    )
    ss.run("biases in large language models")
    """

    semanticscholar_search: Any  #: :meta private:
    top_k_results: int = 5
    S2_MAX_QUERY_LENGTH: int = 300
    load_max_docs: int = 100
    doc_content_chars_max: Optional[int] = 4000
    returned_fields = [
        "title",
        "abstract",
        "venue",
        "year",
        "paperId",
        "citationCount",
        "openAccessPdf",
        "authors",
        "externalIds",
    ]

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        try:
            from semanticscholar import SemanticScholar

            sch = SemanticScholar()
            values["semanticscholar_search"] = sch.search_paper
        except ImportError:
            raise ImportError(
                "Could not import Semanticscholar python package. "
                "Please install it with `pip install semanticscholar`."
            )
        return values

    def run(self, query: str) -> str:
        """Run the Semantic Scholar API."""
        results = self.semanticscholar_search(
            query, limit=self.load_max_docs, fields=self.returned_fields
        )
        documents = []
        for item in results[: self.top_k_results]:
            authors = ", ".join(
                author["name"] for author in getattr(item, "authors", [])
            )
            documents.append(
                f"Published year: {getattr(item, 'year', None)}\n"
                f"Title: {getattr(item, 'title', None)}\n"
                f"Authors: {authors}\n"
                f"Abstract: {getattr(item, 'abstract', None)}\n"
            )

        if documents:
            return "\n\n".join(documents)[: self.doc_content_chars_max]
        else:
            return "No results found."
