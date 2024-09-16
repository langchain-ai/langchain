from typing import Iterator, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper


class WikipediaLoader(BaseLoader):
    """Load from `Wikipedia`.

    The hard limit on the length of the query is 300 for now.

    Each wiki page represents one Document.
    """

    def __init__(
        self,
        query: str,
        lang: str = "en",
        load_max_docs: Optional[int] = 25,
        load_all_available_meta: Optional[bool] = False,
        doc_content_chars_max: Optional[int] = 4000,
    ):
        """
        Initializes a new instance of the WikipediaLoader class.

        Args:
            query (str): The query string to search on Wikipedia.
            lang (str, optional): The language code for the Wikipedia language edition.
                Defaults to "en".
            load_max_docs (int, optional): The maximum number of documents to load.
                Defaults to 100.
            load_all_available_meta (bool, optional): Indicates whether to load all
                available metadata for each document. Defaults to False.
            doc_content_chars_max (int, optional): The maximum number of characters
                for the document content. Defaults to 4000.
        """
        self.query = query
        self.lang = lang
        self.load_max_docs = load_max_docs
        self.load_all_available_meta = load_all_available_meta
        self.doc_content_chars_max = doc_content_chars_max

    def lazy_load(self) -> Iterator[Document]:
        """
        Loads the query result from Wikipedia into a list of Documents.

        Returns:
            A list of Document objects representing the loaded
                Wikipedia pages.
        """
        client = WikipediaAPIWrapper(  # type: ignore[call-arg]
            lang=self.lang,
            top_k_results=self.load_max_docs,  # type: ignore[arg-type]
            load_all_available_meta=self.load_all_available_meta,  # type: ignore[arg-type]
            doc_content_chars_max=self.doc_content_chars_max,  # type: ignore[arg-type]
        )
        yield from client.load(self.query)
