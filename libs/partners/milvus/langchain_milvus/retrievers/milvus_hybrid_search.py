from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pymilvus import AnnSearchRequest, Collection
from pymilvus.client.abstract import BaseRanker, SearchResult  # type: ignore

from langchain_milvus.utils.sparse import BaseSparseEmbedding


class MilvusCollectionHybridSearchRetriever(BaseRetriever):
    """Hybrid search retriever
    that uses Milvus Collection to retrieve documents based on multiple fields.

    For more information, please refer to:
    https://milvus.io/docs/release_notes.md#Multi-Embedding---Hybrid-Search
    """

    collection: Collection
    """Milvus Collection object."""
    rerank: BaseRanker
    """Milvus ranker object. Such as WeightedRanker or RRFRanker."""
    anns_fields: List[str]
    """The names of vector fields that are used for ANNS search."""
    field_embeddings: List[Union[Embeddings, BaseSparseEmbedding]]
    """The embedding functions of each vector fields, 
    which can be either Embeddings or BaseSparseEmbedding."""
    field_search_params: Optional[List[Dict]] = None
    """The search parameters of each vector fields. 
    If not specified, the default search parameters will be used."""
    field_limits: Optional[List[int]] = None
    """Limit number of results for each ANNS field. 
    If not specified, the default top_k will be used."""
    field_exprs: Optional[List[Optional[str]]] = None
    """The boolean expression for filtering the search results."""
    top_k: int = 4
    """Final top-K number of documents to retrieve."""
    text_field: str = "text"
    """The text field name, 
    which will be used as the `page_content` of a `Document` object."""
    output_fields: Optional[List[str]] = None
    """Final output fields of the documents. 
    If not specified, all fields except the vector fields will be used as output fields,
    which will be the `metadata` of a `Document` object."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        # If some parameters are not specified, set default values
        if self.field_search_params is None:
            default_search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }
            self.field_search_params = [default_search_params] * len(self.anns_fields)
        if self.field_limits is None:
            self.field_limits = [self.top_k] * len(self.anns_fields)
        if self.field_exprs is None:
            self.field_exprs = [None] * len(self.anns_fields)

        # Check the fields
        self._validate_fields_num()
        self.output_fields = self._get_output_fields()
        self._validate_fields_name()

        # Load collection
        self.collection.load()

    def _validate_fields_num(self) -> None:
        assert (
            len(self.anns_fields) >= 2
        ), "At least two fields are required for hybrid search."
        lengths = [len(self.anns_fields)]
        if self.field_limits is not None:
            lengths.append(len(self.field_limits))
        if self.field_exprs is not None:
            lengths.append(len(self.field_exprs))

        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All field-related lists must have the same length.")

        if len(self.field_search_params) != len(self.anns_fields):  # type: ignore[arg-type]
            raise ValueError(
                "field_search_params must have the same length as anns_fields."
            )

    def _validate_fields_name(self) -> None:
        collection_fields = [x.name for x in self.collection.schema.fields]
        for field in self.anns_fields:
            assert (
                field in collection_fields
            ), f"{field} is not a valid field in the collection."
        assert (
            self.text_field in collection_fields
        ), f"{self.text_field} is not a valid field in the collection."
        for field in self.output_fields:  # type: ignore[union-attr]
            assert (
                field in collection_fields
            ), f"{field} is not a valid field in the collection."

    def _get_output_fields(self) -> List[str]:
        if self.output_fields:
            return self.output_fields
        output_fields = [x.name for x in self.collection.schema.fields]
        for field in self.anns_fields:
            if field in output_fields:
                output_fields.remove(field)
        if self.text_field not in output_fields:
            output_fields.append(self.text_field)
        return output_fields

    def _build_ann_search_requests(self, query: str) -> List[AnnSearchRequest]:
        search_requests = []
        for ann_field, embedding, param, limit, expr in zip(
            self.anns_fields,
            self.field_embeddings,
            self.field_search_params,  # type: ignore[arg-type]
            self.field_limits,  # type: ignore[arg-type]
            self.field_exprs,  # type: ignore[arg-type]
        ):
            request = AnnSearchRequest(
                data=[embedding.embed_query(query)],
                anns_field=ann_field,
                param=param,
                limit=limit,
                expr=expr,
            )
            search_requests.append(request)
        return search_requests

    def _parse_document(self, data: dict) -> Document:
        return Document(
            page_content=data.pop(self.text_field),
            metadata=data,
        )

    def _process_search_result(
        self, search_results: List[SearchResult]
    ) -> List[Document]:
        documents = []
        for result in search_results[0]:
            data = {x: result.entity.get(x) for x in self.output_fields}  # type: ignore[union-attr]
            doc = self._parse_document(data)
            documents.append(doc)
        return documents

    def hybrid_search(
        self,
        query: str,
    ) -> List[SearchResult]:
        requests = self._build_ann_search_requests(query)
        search_result = self.collection.hybrid_search(
            requests, self.rerank, limit=self.top_k, output_fields=self.output_fields
        )
        return search_result

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        search_result = self.hybrid_search(query)
        documents = self._process_search_result(search_result)
        return documents
