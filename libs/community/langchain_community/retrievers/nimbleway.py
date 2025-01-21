import os
from enum import Enum
from typing import List, Any

import requests
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


class NimbleRetriever(BaseRetriever):
    """Nimbleway Search API retriever.
    Allows you to retrieve search results from Google, Bing, and Yandex.
    Visit https://www.nimbleway.com/ and sign up to receive an
    API key and to see more info.

    Args:
        api_key: The API key for Nimbleway.
        search_engine: The search engine to use. Default is Google.
        render: Whether to render the results web sites. Default is True.
        locale: The locale to use. Default is "en".
        country: The country to use. Default is "US".
        parsing_type: The parsing type to use. Default is "plain_text".
        links: The list of links to search for. Default is None. (if enabled will
        ignore the query)
    """

    api_key: str = None
    k: int = 3
    search_engine: SearchEngine = SearchEngine.GOOGLE
    parse: bool = False
    render: bool = True
    locale: str = "en"
    country: str = "US"
    parsing_type: ParsingType = ParsingType.PLAIN_TEXT
    links: List[str] = None

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun,
            **kwargs: Any
    ) -> List[Document]:
        request_body = {
            "query": query,
            "num_results": kwargs.get("k", self.k),
            "search_engine": kwargs.get("search_engine", self.search_engine),
            "parse": kwargs.get("parse", self.parse),
            "render": kwargs.get("render", self.render),
            "locale": kwargs.get("locale", self.locale),
            "country": kwargs.get("country", self.country),
            "parsing_type": kwargs.get("parsing_type", self.parsing_type),
            "links": kwargs.get("links", self.links)
        }
        route = "extract" if self.links else "search"
        response = requests.post(
            f"https://searchit-server.crawlit.live/{route}",
            json=request_body,
            headers={
                "Authorization": f"Basic {self.api_key or os.getenv('NIMBLE_API_KEY')}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        raw_json_content = response.json()
        docs = [
            Document(
                page_content=doc.get("page_content", ""),
                metadata={
                    "title": doc.get("metadata", {}).get("title", ""),
                    "snippet": doc.get("metadata", {}).get("snippet", ""),
                    "url": doc.get("metadata", {}).get("url", ""),
                    "position": doc.get("metadata", {}).get("position", -1),
                    "entity_type": doc.get("metadata", {}).get("entity_type", ""),
                },
            )
            for doc in raw_json_content.get("body", [])
        ]
        return docs
