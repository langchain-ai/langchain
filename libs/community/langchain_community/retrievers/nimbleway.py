from enum import Enum
from typing import List, Any
import requests
import os

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents.base import Document
from langchain_core.retrievers import BaseRetriever


class SearchEngine(str, Enum):
    """
    Enum representing the search engines supported by Nimble
    """
    GOOGLE = "google_search"
    GOOGLE_SGE = "google_sge"
    BING = "bing_search"
    YANDEX = "yandex_search"


class ParsingType(str, Enum):
    """
    Enum representing the parsing types supported by Nimble
    """
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    SIMPLIFIED_HTML = "simplified_html"


class NimblewayRetriever(BaseRetriever):
    """Nimbleway Search API retriever.
    Allows you to retrieve search results from Google, Bing, and Yandex.
    Visit https://www.nimbleway.com/ and sign up to receive an API key and to see more info.

    Args:
        api_key: The API key for Nimbleway.
        search_engine: The search engine to use. Default is Google.
    """

    api_key: str
    num_results: int = 3
    search_engine: SearchEngine = SearchEngine.GOOGLE
    parse: bool = False
    render: bool = True
    locale: str = "en"
    country: str = "US"
    parsing_type: ParsingType = ParsingType.PLAIN_TEXT

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        request_body = {
            'query': query,
            'num_results': self.num_results,
            'search_engine': self.search_engine.value,
            'parse': self.parse,
            'render': self.render,
            'locale': self.locale,
            'country': self.country,
            'parsing_type': self.parsing_type
        }

        response = requests.post("https://searchit-server.crawlit.live/search",
                                 json=request_body,
                                 headers={
                                     'Authorization': f'Basic {self.api_key or os.getenv("NIMBLE_API_KEY")}',
                                     'Content-Type': 'application/json'
                                 })
        response.raise_for_status()
        raw_json_content = response.json()
        docs = [Document(page_content=doc.get("page_content", ""),
                         metadata={
                             "title": doc.get("metadata", {}).get("title", ""),
                             "snippet": doc.get("metadata", {}).get("snippet", ""),
                             "url": doc.get("metadata", {}).get("url", ""),
                             "position": doc.get("metadata", {}).get("position", -1),
                             "entity_type": doc.get("metadata", {}).get("entity_type", "")
                         }
                         )
                for doc in raw_json_content.get('body', [])]
        return docs
