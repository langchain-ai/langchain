from typing import List

import requests
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents.base import Document
from langchain_core.retrievers import BaseRetriever


class BreebsRetriever(BaseRetriever):
    """A retriever class for `Breebs`.

    See https://www.breebs.com/ for more info.
    Args:
        breeb_key: The key to trigger the breeb
        (specialized knowledge pill on a specific topic).

    To retrieve the list of all available Breebs : you can call https://breebs.promptbreeders.com/web/listbreebs
    """

    breeb_key: str
    url = "https://breebs.promptbreeders.com/knowledge"

    def __init__(self, breeb_key: str):
        super().__init__(breeb_key=breeb_key)
        self.breeb_key = breeb_key

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve context for given query.
        Note that for time being there is no score."""
        r = requests.post(
            self.url,
            json={
                "breeb_key": self.breeb_key,
                "query": query,
            },
        )
        if r.status_code != 200:
            return []
        else:
            chunks = r.json()
            return [
                Document(
                    page_content=chunk["content"],
                    metadata={"source": chunk["source_url"], "score": 1},
                )
                for chunk in chunks
            ]
