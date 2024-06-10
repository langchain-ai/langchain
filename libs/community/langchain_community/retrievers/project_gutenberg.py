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


"""Search by title on Project Gutenberg."""
SEARCH_URL = "http://gutendex.com/books/?search="
"""Fetch content of books from Project Gutenberg."""
FILES_URL = "https://www.gutenberg.org/files/"

class ProjectGutenbergRetriever(BaseRetriever):

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
                if book["title"] == title:
                    book_id = book["id"]
                    book_ids.append(book_id)
                    break

        documents = []
        for book_id in book_ids:
            book_url = f"{FILES_URL}{book_id}/{book_id}-0.txt"
            response = requests.get(book_url)
            text = response.text
            documents.append(Document(page_content=text, metadata={"title": title, "url": book_url, "book_id": book_id}))

        return documents
