"""SQL Database Chain Retriever"""
from typing import Any, Dict, List

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.chains.sql_database.base import SQLDatabaseChain
from langchain.schema import BaseRetriever, Document


class SQLDatabaseChainRetriever(BaseRetriever):
    """Retriever that uses SQLDatabase as Retriever"""

    sql_db_chain: SQLDatabaseChain
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
        if not self.sql_db_chain.native_format:
            raise TypeError(
                "SQL Database Chain must return in native format. \
                 Try to turn `native_format` in this chain to `True`."
            )
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
