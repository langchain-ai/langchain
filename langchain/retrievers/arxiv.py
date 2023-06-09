from typing import Any, List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.schema import BaseRetriever, Document
from langchain.utilities.arxiv import ArxivAPIWrapper


class ArxivRetriever(BaseRetriever, ArxivAPIWrapper):
    """
    It is effectively a wrapper for ArxivAPIWrapper.
    It wraps load() to get_relevant_documents().
    It uses all ArxivAPIWrapper arguments without any change.
    """

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any,
    ) -> List[Document]:
        return self.load(query=query)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None,
        **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError
