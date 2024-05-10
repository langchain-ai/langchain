"""Util that calls Wikipedia."""
import logging
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator

logger = logging.getLogger(__name__)

WIKIPEDIA_MAX_QUERY_LENGTH = 300


class WikipediaAPIWrapper(BaseModel):
    """Wrapper around WikipediaAPI.

    To use, you should have the ``wikipedia`` python package installed.
    This wrapper will use the Wikipedia API to conduct searches and
    fetch page summaries. By default, it will return the page summaries
    of the top-k results.
    It limits the Document content by doc_content_chars_max.
    """

    wiki_client: Any  #: :meta private:
    top_k_results: int = 3
    lang: str = "en"
    load_all_available_meta: bool = False
    doc_content_chars_max: int = 4000

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        try:
            import wikipedia

            wikipedia.set_lang(values["lang"])
            values["wiki_client"] = wikipedia
        except ImportError:
            raise ImportError(
                "Could not import wikipedia python package. "
                "Please install it with `pip install wikipedia`."
            )
        return values

    def run(self, query: str) -> str:
        """Run Wikipedia search and get page summaries."""
        page_titles = self.wiki_client.search(
            query[:WIKIPEDIA_MAX_QUERY_LENGTH], results=self.top_k_results
        )
        summaries = []
        for page_title in page_titles[: self.top_k_results]:
            if wiki_page := self._fetch_page(page_title):
                if summary := self._formatted_page_summary(page_title, wiki_page):
                    summaries.append(summary)
        if not summaries:
            return "No good Wikipedia Search Result was found"
        return "\n\n".join(summaries)[: self.doc_content_chars_max]

    @staticmethod
    def _formatted_page_summary(page_title: str, wiki_page: Any) -> Optional[str]:
        return f"Page: {page_title}\nSummary: {wiki_page.summary}"

    def _page_to_document(self, page_title: str, wiki_page: Any) -> Document:
        main_meta = {
            "title": page_title,
            "summary": wiki_page.summary,
            "source": wiki_page.url,
        }
        add_meta = (
            {
                "categories": wiki_page.categories,
                "page_url": wiki_page.url,
                "image_urls": wiki_page.images,
                "related_titles": wiki_page.links,
                "parent_id": wiki_page.parent_id,
                "references": wiki_page.references,
                "revision_id": wiki_page.revision_id,
                "sections": wiki_page.sections,
            }
            if self.load_all_available_meta
            else {}
        )
        doc = Document(
            page_content=wiki_page.content[: self.doc_content_chars_max],
            metadata={
                **main_meta,
                **add_meta,
            },
        )
        return doc

    def _fetch_page(self, page: str) -> Optional[str]:
        try:
            return self.wiki_client.page(title=page, auto_suggest=False)
        except (
            self.wiki_client.exceptions.PageError,
            self.wiki_client.exceptions.DisambiguationError,
        ):
            return None

    def load(self, query: str) -> List[Document]:
        """
        Run Wikipedia search and get the article text plus the meta information.
        See

        Returns: a list of documents.

        """
        return list(self.lazy_load(query))

    def lazy_load(self, query: str) -> Iterator[Document]:
        """
        Run Wikipedia search and get the article text plus the meta information.
        See

        Returns: a list of documents.

        """
        page_titles = self.wiki_client.search(
            query[:WIKIPEDIA_MAX_QUERY_LENGTH], results=self.top_k_results
        )
        for page_title in page_titles[: self.top_k_results]:
            if wiki_page := self._fetch_page(page_title):
                if doc := self._page_to_document(page_title, wiki_page):
                    yield doc
