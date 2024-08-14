from typing import Any, Dict, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pymongo.collection import Collection

from langchain_mongodb.pipelines import text_search_stage
from langchain_mongodb.utils import make_serializable


class MongoDBAtlasFullTextSearchRetriever(BaseRetriever):
    """Hybrid Search Retriever performs full-text searches
    using Lucene's standard (BM25) analyzer.
    """

    collection: Collection
    """MongoDB Collection on an Atlas cluster"""
    search_index_name: str
    """Atlas Search Index name"""
    search_field: str
    """Collection field that contains the text to be searched. It must be indexed"""
    top_k: Optional[int] = None
    """Number of documents to return. Default is no limit"""
    filter: Optional[Dict[str, Any]] = None
    """(Optional) List of MQL match expression comparing an indexed field"""
    show_embeddings: float = False
    """If true, returned Document metadata will include vectors"""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve documents that are highest scoring / most similar  to query.

        Args:
            query: String to find relevant documents for
            run_manager: The callback handler to use
        Returns:
            List of relevant documents
        """

        pipeline = text_search_stage(  # type: ignore
            query=query,
            search_field=self.search_field,
            index_name=self.search_index_name,
            limit=self.top_k,
            filter=self.filter,
        )

        # Execution
        cursor = self.collection.aggregate(pipeline)  # type: ignore[arg-type]

        # Formatting
        docs = []
        for res in cursor:
            text = res.pop(self.search_field)
            make_serializable(res)
            docs.append(Document(page_content=text, metadata=res))
        return docs
