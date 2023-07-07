"""SQL Database Chain Retriever"""
from typing import Any, List, Coroutine
from pydantic import Field

from langchain.callbacks.manager import (
    CallbackManagerForRetrieverRun,
)

from pydantic import BaseModel

from langchain.chains.sql_database.base import SQLDatabaseChain
from langchain.schema import BaseRetriever, Document

from langchain.schema.document import Document
from langchain.sql_database import SQLDatabase

class SQLDatabaseChainRetriever(BaseRetriever):
    """Retriever that uses SQLDatabase as Retriever"""
    
    sql_db_chain: SQLDatabaseChain
    """SQL Database Chain"""
    page_content_key: str = "content"
    """column name for page content of documents"""
        
    def _get_relevant_documents(self, query: str, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        if not self.sql_db_chain.native_format:
            raise TypeError("SQL Database Chain must return in native format. Try to turn `native_format` in this chain to `True`.")
        ret = self.sql_db_chain.run(query=query, callbacks=run_manager)
        return [Document(page_content=r[self.page_content_key], metadata=r) for r in ret]
    
    async def _aget_relevant_documents(self, query: str, *args: Any, **kwargs: Any) -> Coroutine[Any, Any, List[Document]]:
        raise NotImplementedError