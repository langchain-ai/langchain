"""Util that calls Wikipedia."""
import logging
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Extra, root_validator

from langchain.schema import Document

logger = logging.getLogger(__name__)

WIKIPEDIA_MAX_QUERY_LENGTH = 300


class WikipediaAPIWrapper(BaseModel):
    """Wrapper around WikipediaAPI.

    To use, you should have the ``wikipedia`` python package installed.
    This wrapper will use the Wikipedia API to conduct searches and
    fetch page summaries. By default, it will return the page summaries
    of the top-k results of an input search.
    """

    wiki_client: Any  #: :meta private:
    top_k_results: int = 3
    lang: str = "en"
    load_all_available_meta: bool = False

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        try:
            import wikipedia

            wikipedia.set_lang(values["lang"])
            values["wiki_client"] = wikipedia
        except ImportError:
            raise ValueError(
                "Could not import wikipedia python package. "
                "Please install it with `pip install wikipedia`."
            )
        return values

    def run(self, query: str) -> str:
        """Run Wikipedia search and get page summaries."""
        page_titles = self.wiki_client.search(query[:WIKIPEDIA_MAX_QUERY_LENGTH])
        summaries = []
        for page_title in page_titles[:self.top_k_results]:
            if wiki_page := self._fetch_page(page_title):
                if summary := self._formatted_page_summary(page_title, wiki_page):
                    summaries.append(summary)
        if not summaries:
            return "No good Wikipedia Search Result was found"
        return "\n\n".join(summaries)

    @staticmethod
    def _formatted_page_summary(page_title: str, wiki_page: str) -> Optional[str]:
        return f"Page: {page_title}\nSummary: {wiki_page.summary}"

    def _page_to_document(self, page_title: str, wiki_page: str) -> Document:
        main_meta = {
            "title": page_title,
            "summary": wiki_page.summary,
        }
        add_meta = (
            {
                "categories": wiki_page.categories,
                # "coordinates": wiki_page.coordinates,
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
            page_content=wiki_page.content,
            metadata={                    **main_meta,                    **add_meta,                }
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

        Returns: a list of documents with the document.page_content in PDF format

        """
        page_titles = self.wiki_client.search(query[:WIKIPEDIA_MAX_QUERY_LENGTH])
        docs = []
        for page_title in page_titles[:self.top_k_results]:
            if wiki_page := self._fetch_page(page_title):
                if doc := self._page_to_document(page_title, wiki_page):
                    docs.append(doc)
        return docs

        # try:
        #     docs: List[Document] = []
        #     for result in self.arxiv_search(  # type: ignore
        #         query[: self.ARXIV_MAX_QUERY_LENGTH], max_results=self.load_max_docs
        #     ).results():
        #         try:
        #             doc_file_name: str = result.download_pdf()
        #             with fitz.open(doc_file_name) as doc_file:
        #                 text: str = "".join(page.get_text() for page in doc_file)
        #                 add_meta = (
        #                     {
        #                         "entry_id": result.entry_id,
        #                         "published_first_time": str(result.published.date()),
        #                         "comment": result.comment,
        #                         "journal_ref": result.journal_ref,
        #                         "doi": result.doi,
        #                         "primary_category": result.primary_category,
        #                         "categories": result.categories,
        #                         "links": [link.href for link in result.links],
        #                     }
        #                     if self.load_all_available_meta
        #                     else {}
        #                 )
        #                 doc = Document(
        #                     page_content=text,
        #                     metadata=(
        #                         {
        #                             "Published": str(result.updated.date()),
        #                             "Title": result.title,
        #                             "Authors": ", ".join(
        #                                 a.name for a in result.authors
        #                             ),
        #                             "Summary": result.summary,
        #                             **add_meta,
        #                         }
        #                     ),
        #                 )
        #                 docs.append(doc)
        #         except FileNotFoundError as f_ex:
        #             logger.debug(f_ex)
        #     return docs
        # except self.arxiv_exceptions as ex:
        #     logger.debug("Error on arxiv: %s", ex)
        #     return []
