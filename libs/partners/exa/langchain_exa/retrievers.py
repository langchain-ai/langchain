from typing import Any, Dict, List, Literal, Optional, Union

from exa_py import Exa  # type: ignore
from exa_py.api import HighlightsContentsOptions, TextContentsOptions  # type: ignore
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.retrievers import BaseRetriever

from langchain_exa._utilities import initialize_client


def _get_metadata(result: Any) -> Dict[str, Any]:
    """Get the metadata from a result object."""
    metadata = {
        "title": result.title,
        "url": result.url,
        "id": result.id,
        "score": result.score,
        "published_date": result.published_date,
        "author": result.author,
    }
    if getattr(result, "highlights"):
        metadata["highlights"] = result.highlights
    if getattr(result, "highlight_scores"):
        metadata["highlight_scores"] = result.highlight_scores
    return metadata


class ExaSearchRetriever(BaseRetriever):
    """Exa Search retriever."""

    k: int = 10  # num_results
    """The number of search results to return."""
    include_domains: Optional[List[str]] = None
    """A list of domains to include in the search."""
    exclude_domains: Optional[List[str]] = None
    """A list of domains to exclude from the search."""
    start_crawl_date: Optional[str] = None
    """The start date for the crawl (in YYYY-MM-DD format)."""
    end_crawl_date: Optional[str] = None
    """The end date for the crawl (in YYYY-MM-DD format)."""
    start_published_date: Optional[str] = None
    """The start date for when the document was published (in YYYY-MM-DD format)."""
    end_published_date: Optional[str] = None
    """The end date for when the document was published (in YYYY-MM-DD format)."""
    use_autoprompt: Optional[bool] = None
    """Whether to use autoprompt for the search."""
    type: str = "neural"
    """The type of search, 'keyword' or 'neural'. Default: neural"""
    highlights: Optional[Union[HighlightsContentsOptions, bool]] = None
    """Whether to set the page content to the highlights of the results."""
    text_contents_options: Union[TextContentsOptions, Literal[True]] = True
    """How to set the page content of the results"""

    client: Exa = Field(default=None)
    exa_api_key: SecretStr = Field(default=None)
    exa_base_url: Optional[str] = None

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate the environment."""
        values = initialize_client(values)
        return values

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        response = self.client.search_and_contents(  # type: ignore[misc]
            query,
            num_results=self.k,
            text=self.text_contents_options,
            highlights=self.highlights,  # type: ignore
            include_domains=self.include_domains,
            exclude_domains=self.exclude_domains,
            start_crawl_date=self.start_crawl_date,
            end_crawl_date=self.end_crawl_date,
            start_published_date=self.start_published_date,
            end_published_date=self.end_published_date,
            use_autoprompt=self.use_autoprompt,
        )

        results = response.results

        return [
            Document(
                page_content=(result.text),
                metadata=_get_metadata(result),
            )
            for result in results
        ]
