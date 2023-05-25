from typing import List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.utilities.wikipedia import WikipediaAPIWrapper


class WikipediaLoader(BaseLoader):
    """Loads a query result from www.wikipedia.org into a list of Documents.
    The hard limit on the number of downloaded Documents is 300 for now.

    Each wiki page represents one Document.
    """

    def __init__(
        self,
        query: str,
        lang: str = "en",
        load_max_docs: Optional[int] = 100,
        load_all_available_meta: Optional[bool] = False,
    ):
        self.query = query
        self.lang = lang
        self.load_max_docs = load_max_docs
        self.load_all_available_meta = load_all_available_meta

    def load(self) -> List[Document]:
        client = WikipediaAPIWrapper(
            lang=self.lang,
            top_k_results=self.load_max_docs,
            load_all_available_meta=self.load_all_available_meta,
        )
        docs = client.load(self.query)
        return docs
