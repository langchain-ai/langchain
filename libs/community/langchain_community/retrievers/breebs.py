import requests

from typing import Any, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents.base import Document

from langchain_core.retrievers import BaseRetriever


class BreebsKnowledgeRetriever(BaseRetriever):
    """A retriever class for `Breebs`.

    See https://www.breebs.com/ for more info.
    Args:
        breeb_key: The key to trigger the breeb (specialized knowledge pill on a specific topic).
    """

    breeb_key: str
    endpoint_url = "https://3evn0x5te0.execute-api.eu-west-3.amazonaws.com/prod/breeb"

    def __init__(self, breeb_key: str):
        super().__init__(breeb_key=breeb_key)
        self.breeb_key = breeb_key

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        r = requests.post(
            self.endpoint_url,
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


# breeb_key = "Node21Docs"
# retriever = BreebsKnowledgeRetriever(breeb_key)
# documents = retriever._get_relevant_documents(
#     "How promises work in the last documentation of Node js?",
#     run_manager=CallbackManagerForRetrieverRun,
# )
