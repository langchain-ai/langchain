from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.utilities.wikipedia import WikipediaAPIWrapper


class WikipediaRetriever(BaseRetriever, WikipediaAPIWrapper):
    """`Wikipedia API` retriever.

    It wraps load() to get_relevant_documents().
    It uses all WikipediaAPIWrapper arguments without any change.
    """

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.load(query=query)
