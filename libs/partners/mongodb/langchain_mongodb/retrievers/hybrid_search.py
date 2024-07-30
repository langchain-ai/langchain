from typing import (
    List,
    Optional,
)

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pymongo.collection import Collection

from langchain_mongodb.pipelines import (
    MongoDBDocument,
    combine_pipelines,
    final_hybrid_stage,
    reciprocal_rank_stage,
    text_search_stage,
    vector_search_stage,
)
from langchain_mongodb.utils import make_serializable


class MongoDBAtlasHybridSearchRetriever(BaseRetriever):
    """Hybrid Search Retriever combines vector and full-text searches
    weighting them the via Reciprocal Rank Fusion algorithm.
    """

    collection: Collection
    """Collection on Atlas cluster."""
    embedding_model: Embeddings
    """Text-embedding model, e.g. langchain_openai.OpenAIEmbeddings"""
    vector_search_index_name: str
    """Atlas Vector Search Index name"""
    search_index_name: str
    """Atlas Search Index name"""
    page_content_field: str
    """Field containing text content to embed"""
    embedding_field: str = "embedding"
    """Field in Collection containing embedding vectors"""
    top_k: int = 4
    """Number of documents to return."""
    oversampling_factor: int = 10
    """This times top_k is the number of candidates chosen at each step"""
    pre_filter: Optional[MongoDBDocument] = None
    """(Optional) Any MQL match expression comparing an indexed field"""
    post_filter: Optional[MongoDBDocument] = None
    """(Optional) Pipeline of MongoDB aggregation stages for postprocessing."""
    vector_penalty: float = 0.0
    """Penalty applied to vector search results in RRF: scores=1/(rank + penalty)"""
    fulltext_penalty: float = 0.0
    """Penalty applied to full-text search results in RRF: scores=1/(rank + penalty)"""
    show_embeddings: float = False
    """If true, returned Document metadata will include vectors."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve documents that are highest scoring / most similar  to query.

        Note that the query is embedded for vector search,
        but the text search component must be included in constructor.

        Args:
            query: String to find relevant documents for
            run_manager: The callback handler to use
        Returns:
            List of relevant documents
        """

        query_vector = self.embedding_model.embed_query(query)

        scores_fields = ["vector_score", "fulltext_score"]
        pipeline = []

        # Vector Search stage
        vector_pipeline = [
            vector_search_stage(
                query_vector,
                self.embedding_field,
                self.vector_search_index_name,
                self.top_k,
                self.pre_filter,
                self.oversampling_factor,
            )
        ]
        vector_pipeline += reciprocal_rank_stage("vector_score", self.vector_penalty)

        combine_pipelines(pipeline, vector_pipeline, self.collection.name)

        # Full-Text Search stage  # TODO Compare with index.text_search_stage and move
        text_pipeline = text_search_stage(
            query=query,
            search_field=self.page_content_field,
            index_name=self.search_index_name,
            limit=self.top_k,
            pre_filter=self.pre_filter,
        )

        text_pipeline += reciprocal_rank_stage("fulltext_score", self.fulltext_penalty)

        combine_pipelines(pipeline, text_pipeline, self.collection.name)

        # Sum and sort stage
        pipeline += final_hybrid_stage(scores_fields=scores_fields, limit=self.top_k)

        # Removal of embeddings unless requested.
        if not self.show_embeddings:
            pipeline.append({"$project": {self.embedding_field: 0}})
        # Post filtering
        if self.post_filter is not None:
            pipeline.extend(self.post_filter)

        # Execution
        cursor = self.collection.aggregate(pipeline)  # type: ignore[arg-type]

        # Formatting
        docs = []
        for res in cursor:
            text = res.pop(self.page_content_field)
            # score = res.pop("score")  # The score remains buried!
            make_serializable(res)
            docs.append(Document(page_content=text, metadata=res))
        return docs
