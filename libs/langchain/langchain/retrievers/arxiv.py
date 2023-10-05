from typing import List

from langchain.schema import BaseRetriever, Document
from langchain.schema.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.utilities.arxiv import ArxivAPIWrapper


class ArxivRetriever(BaseRetriever, ArxivAPIWrapper):
    """`Arxiv` retriever.

    It wraps load() to get_relevant_documents().
    It uses all ArxivAPIWrapper arguments without any change.
    """

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.load(query=query)
