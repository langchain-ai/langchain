from typing import Any, Dict, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pymongo.collection import Collection

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.pipelines import (
    combine_pipelines,
    final_hybrid_stage,
    reciprocal_rank_stage,
    text_search_stage,
    vector_search_stage,
)
from langchain_mongodb.utils import make_serializable


class MongoDBAtlasHybridSearchRetriever(BaseRetriever):
    """Hybrid Search Retriever combines vector and full-text searches
    weighting them the via Reciprocal Rank Fusion (RRF) algorithm.

    Increasing the vector_penalty will reduce the importance on the vector search.
    Increasing the fulltext_penalty will correspondingly reduce the fulltext score.
    For more on the algorithm,see
    https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking
    """

    vectorstore: MongoDBAtlasVectorSearch
    """MongoDBAtlas VectorStore"""
    search_index_name: str
    """Atlas Search Index (full-text) name"""
    top_k: int = 4
    """Number of documents to return."""
    oversampling_factor: int = 10
    """This times top_k is the number of candidates chosen at each step"""
    pre_filter: Optional[Dict[str, Any]] = None
    """(Optional) Any MQL match expression comparing an indexed field"""
    post_filter: Optional[List[Dict[str, Any]]] = None
    """(Optional) Pipeline of MongoDB aggregation stages for postprocessing."""
    vector_penalty: float = 60.0
    """Penalty applied to vector search results in RRF: scores=1/(rank + penalty)"""
    fulltext_penalty: float = 60.0
    """Penalty applied to full-text search results in RRF: scores=1/(rank + penalty)"""
    show_embeddings: float = False
    """If true, returned Document metadata will include vectors."""

    @property
    def collection(self) -> Collection:
        return self.vectorstore._collection

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve documents that are highest scoring / most similar  to query.

        Note that the same query is used in both searches,
        embedded for vector search, and as-is for full-text search.

        Args:
            query: String to find relevant documents for
            run_manager: The callback handler to use
        Returns:
            List of relevant documents
        """

        query_vector = self.vectorstore._embedding.embed_query(query)

        scores_fields = ["vector_score", "fulltext_score"]
        pipeline: List[Any] = []

        # First we build up the aggregation pipeline,
        # then it is passed to the server to execute
        # Vector Search stage
        vector_pipeline = [
            vector_search_stage(
                query_vector=query_vector,
                search_field=self.vectorstore._embedding_key,
                index_name=self.vectorstore._index_name,
                top_k=self.top_k,
                filter=self.pre_filter,
                oversampling_factor=self.oversampling_factor,
            )
        ]
        vector_pipeline += reciprocal_rank_stage("vector_score", self.vector_penalty)

        combine_pipelines(pipeline, vector_pipeline, self.collection.name)

        # Full-Text Search stage
        text_pipeline = text_search_stage(
            query=query,
            search_field=self.vectorstore._text_key,
            index_name=self.search_index_name,
            limit=self.top_k,
            filter=self.pre_filter,
        )

        text_pipeline.extend(
            reciprocal_rank_stage("fulltext_score", self.fulltext_penalty)
        )

        combine_pipelines(pipeline, text_pipeline, self.collection.name)

        # Sum and sort stage
        pipeline.extend(
            final_hybrid_stage(scores_fields=scores_fields, limit=self.top_k)
        )

        # Removal of embeddings unless requested.
        if not self.show_embeddings:
            pipeline.append({"$project": {self.vectorstore._embedding_key: 0}})
        # Post filtering
        if self.post_filter is not None:
            pipeline.extend(self.post_filter)

        # Execution
        cursor = self.collection.aggregate(pipeline)  # type: ignore[arg-type]

        # Formatting
        docs = []
        for res in cursor:
            text = res.pop(self.vectorstore._text_key)
            # score = res.pop("score")  # The score remains buried!
            make_serializable(res)
            docs.append(Document(page_content=text, metadata=res))
        return docs
