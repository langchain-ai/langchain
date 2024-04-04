from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_community.utilities.rememberizer import RememberizerAPIWrapper


class RememberizerRetriever(BaseRetriever, RememberizerAPIWrapper):
    """`Rememberizer` retriever.

    It wraps load() to get_relevant_documents().
    It uses all RememberizerAPIWrapper arguments without any change.
    """

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.load(query=query)
