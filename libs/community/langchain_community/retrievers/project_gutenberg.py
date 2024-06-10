import re
from typing import List, Any
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.runnables.config import run_in_executor
from langchain_core.retrievers import BaseRetriever

try:
    import requests
except ImportError:
    raise ImportError(
        "Could not import requests package. "
        "Please install it with `pip install requests`."
    )


try:
    import diskcache as dc
    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False

"""Search by title on Project Gutenberg."""
SEARCH_URL = "http://gutendex.com/books/?search="
"""Fetch content of books from Project Gutenberg."""
FILES_URL = "https://www.gutenberg.org/files/"

class ProjectGutenbergRetriever(BaseRetriever):
    """Project Gutenberg retriever."""

    def __init__(self, /, use_cache: bool = False, cache_dir: str = "cache", **data: Any):
        super().__init__(**data)
        self.use_cache = use_cache
        if self.use_cache and DISKCACHE_AVAILABLE:
            self.cache = dc.Cache(cache_dir)
        elif self.use_cache:
            raise ImportError(
                "Could not import diskcache python package. "
                "Please install it with `pip install diskcache`."
            )
            raise ImportError("diskcache is not installed. Install it to use caching.")
        else:
            self.cache = None

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """Get documents relevant to a query from Project Gutenberg."""
        return self.maybe_get_gutenberg_books(query)

    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> List[Document]:
        """Asynchronously get documents relevant to a query from Project Gutenberg."""
        return await run_in_executor(
            None,
            self._get_relevant_documents,
            query,
            run_manager=run_manager.get_sync(),
        )

    def maybe_get_gutenberg_books(self, title: str) -> List[Document]:
        if self.use_cache and self.cache:
            cache_key = f"gutenberg_books_{title}"
            if cache_key in self.cache:
                return self.cache[cache_key]

        url = f"{SEARCH_URL}{title}"
        response = requests.get(url)

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            raise Exception(f"Failed to get books from Project Gutenberg with title: {title}")

        books = response.json().get("results", [])
        book_ids = []
        for book in books:
            if title.lower() in book["title"].lower():
                book_id = book["id"]
                book_ids.append(book_id)

        documents = []
        for book_id in book_ids:
            cache_key = f"gutenberg_book_{book_id}"
            if self.use_cache and self.cache and cache_key in self.cache:
                documents.extend(self.cache[cache_key])
                continue
            else:
                book_url = f"{FILES_URL}{book_id}/{book_id}-0.txt"
                response = requests.get(book_url)
                text = response.text
                # Get rid of binary characters
                text = re.sub(r"[^\x00-\x7F]+", "", text)
                documents.append(Document(content=text, metadata={"title": title, "url": book_url, "book_id": book_id}))
                if self.use_cache and self.cache:
                    self.cache[cache_key] = documents

        return documents
