import os
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class SearchDepth(Enum):
    """Search depth as enumerator."""

    BASIC = "basic"
    ADVANCED = "advanced"


class TavilySearchAPIRetriever(BaseRetriever):
    """Tavily Search API retriever."""

    k: int = 10
    include_generated_answer: bool = False
    include_raw_content: bool = False
    include_images: bool = False
    search_depth: SearchDepth = SearchDepth.BASIC
    include_domains: Optional[List[str]] = None
    exclude_domains: Optional[List[str]] = None
    kwargs: Optional[Dict[str, Any]] = {}
    api_key: Optional[str] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        try:
            from tavily import Client
        except ImportError:
            raise ImportError(
                "Tavily python package not found. "
                "Please install it with `pip install tavily-python`."
            )

        tavily = Client(api_key=self.api_key or os.environ["TAVILY_API_KEY"])
        max_results = self.k if not self.include_generated_answer else self.k - 1
        response = tavily.search(
            query=query,
            max_results=max_results,
            search_depth=self.search_depth.value,
            include_answer=self.include_generated_answer,
            include_domains=self.include_domains,
            exclude_domains=self.exclude_domains,
            include_raw_content=self.include_raw_content,
            include_images=self.include_images,
            **self.kwargs,
        )
        docs = [
            Document(
                page_content=result.get("content", "")
                if not self.include_raw_content
                else result.get("raw_content", ""),
                metadata={
                    "title": result.get("title", ""),
                    "source": result.get("url", ""),
                    **{
                        k: v
                        for k, v in result.items()
                        if k not in ("content", "title", "url", "raw_content")
                    },
                    "images": response.get("images"),
                },
            )
            for result in response.get("results")
        ]
        if self.include_generated_answer:
            docs = [
                Document(
                    page_content=response.get("answer", ""),
                    metadata={
                        "title": "Suggested Answer",
                        "source": "https://tavily.com/",
                    },
                ),
                *docs,
            ]

        return docs
