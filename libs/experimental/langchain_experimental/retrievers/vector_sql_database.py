"""Vector SQL Database Chain Retriever"""

from typing import Any, Dict, List

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_experimental.sql.vector_sql import VectorSQLDatabaseChain


class VectorSQLDatabaseChainRetriever(BaseRetriever):
    """Retriever that uses Vector SQL Database."""

    sql_db_chain: VectorSQLDatabaseChain
    """SQL Database Chain"""
    page_content_key: str = "content"
    """column name for page content of documents"""

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        ret: List[Dict[str, Any]] = self.sql_db_chain(
            query, callbacks=run_manager.get_child(), **kwargs
        )["result"]
        return [
            Document(page_content=r[self.page_content_key], metadata=r) for r in ret
        ]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        raise NotImplementedError
