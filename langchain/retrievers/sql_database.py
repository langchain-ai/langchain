"""SQL Database Chain Retriever"""
from typing import Any, List, Coroutine

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)

from langchain.chains.sql_database.base import SQLDatabaseChain
from langchain.schema import BaseRetriever, Document

from langchain.schema.document import Document

class SQLDatabaseChainRetriever(BaseRetriever):
    """Retriever that uses SQLDatabase as Retriever"""
    
    def __init__(self, sql_database_chain: SQLDatabaseChain):
        self.chain = sql_database_chain
        
        
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs) -> List[Document]:
        return self.chain.run(query=query, callbacks=run_manager, return_direct=True, **kwargs)
    
    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> Coroutine[Any, Any, List[Document]]:
        raise NotImplementedError