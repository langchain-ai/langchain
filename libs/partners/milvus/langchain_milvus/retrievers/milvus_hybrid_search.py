"""Milvus Retriever"""
import time
from uuid import uuid4
from pymilvus import AnnSearchRequest, Collection, WeightedRanker, MilvusClient, DataType

from typing import Any, Dict, List, Optional, Union

from pymilvus.client.abstract import BaseRanker, SearchResult

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pymilvus import (
    Collection,
    utility,
    AnnSearchRequest, connections, WeightedRanker
)


# class MilvusHybridSearchRetriever(BaseRetriever):
#     # TODO: Update to MilvusClient + Hybrid Search when available
#     client: MilvusClient
#     collection_name: str
#     ...


class MilvusCollectionHybridSearchRetriever(BaseRetriever):
    collection: Collection
    rerank: BaseRanker
    top_k: int = 4
    anns_fields: List[str]
    field_embeddings: List[Embeddings]
    field_search_params: List[Dict] = None
    field_limits: Optional[List[int]] = None
    field_exprs: Optional[List[str]] = None
    output_fields: [str] = None
    # TODO decide the name, whether to unify with other name in other files
    text_field_name: str = 'text'

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if self.field_search_params is None:
            default_search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }
            self.field_search_params = [default_search_params] * len(self.anns_fields)
            self.validate_field_lengths()

    def validate_field_lengths(self):
        # validate the anns_fields, field_params, field_limits, field_exprs has the correct length
        lengths = [len(self.anns_fields)]
        if self.field_limits is not None:
            lengths.append(len(self.field_limits))
        if self.field_exprs is not None:
            lengths.append(len(self.field_exprs))

        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All field-related lists must have the same length.")

        # validate that field_search_params has the correct length
        if len(self.field_search_params) != len(self.anns_fields):
            raise ValueError("field_search_params must have the same length as anns_fields.")

    @classmethod
    def from_client(cls, client: MilvusClient, collection_name: str, **kwargs):
        # TODO return 一个MilvusCollectionHybridSearchRetriever实例
        try:
            # list all aliases
            res = client.list_aliases(collection_name=collection_name)
            if not res['aliases']:
                # if no alias exists, create a new one
                alias = 'alias_' + uuid4().hex
                client.create_alias(collection_name=collection_name, alias=alias)
            else:
                # use the first available alias
                alias = res['aliases'][0]

            # connect to the Milvus server using the alias
            connections.connect(alias, **kwargs)

            # create a Collection instance using the alias
            col = Collection(collection_name, using=alias)

            # get all data fields and initialized in output fields
            fields: List[str] = []
            if isinstance(col, Collection):
                schema = col.schema
                for x in schema.fields:
                    fields.append(x.name)

            # TODO adjust the code to make it more readable
            # initialize and return the retriever instance with the collection and any additional parameters
            return cls(collection=col, output_fields=fields[:], **kwargs)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None


    def _build_ann_search_requests(self, query: str) -> List[AnnSearchRequest]:
        search_requests = []
        for ann_field, embedding, param, limit, expr in zip(self.anns_fields, self.field_embeddings,
                                                            self.field_search_params, self.field_limits,
                                                            self.field_exprs):

            # TODO need to choose the proper limit
            limit = limit if limit is not None else self.top_k # choose top_k

            request = AnnSearchRequest(
                data=embedding.embed_query(query),
                anns_field=ann_field,
                param=param,
                limit=limit,
                expr=expr
            )
            search_requests.append(request)
        return search_requests

    def _parse_document(self, data: dict) -> Document:
        return Document(
            page_content=data.pop(self.text_field_name),
            metadata=data,
        )

    def _process_search_result(self, search_results: List[SearchResult]) -> List[Document]:
        # process the best result with text data and else in metadata
        documents = []
        for result in search_results[0]:
            data = {x: result.entity.get(x) for x in self.output_fields}
            doc = self._parse_document(data)
            documents.append(doc)
        return documents

    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
            **kwargs: Any,
    ) -> List[Document]:
        requests = self._build_ann_search_requests(query)

        # TODO deal with vector field that not in self.anns_fields
        # remove vector
        for field in self.anns_fields:
            if field in self.output_fields:
                self.output_fields.remove(field)

        # pass in output fields without anns field
        search_result = self.collection.hybrid_search(
            requests,  # List of AnnSearchRequests created in step 1
            self.rerank,  # Reranking strategy specified in step 2
            limit=self.top_k,  # Number of final search results to return
            output_fields=self.output_fields
        )
        documents = self._process_search_result(search_result)
        return documents


# ========================================================
# 以下都是测试代码
def insert_data_using_orm():
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
    import random

    connections.connect(
        host="10.100.30.11",  # Replace with your Milvus server IP
        port="19530"
    )

    fields = [
        FieldSchema(name="film_id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="filmVector", dtype=DataType.FLOAT_VECTOR, dim=5),  # Vector field for film vectors
        FieldSchema(name="posterVector", dtype=DataType.FLOAT_VECTOR, dim=5)]  # Vector field for poster vectors

    schema = CollectionSchema(fields=fields, enable_dynamic_field=False)

    collection = Collection(name="test_collection", schema=schema)
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }

    collection.create_index("filmVector", index_params)
    collection.create_index("posterVector", index_params)

    entities = []

    for _ in range(1000):
        # generate random values for each field in the schema
        film_id = random.randint(1, 1000)
        film_vector = [random.random() for _ in range(5)]
        poster_vector = [random.random() for _ in range(5)]

        # creat a dictionary for each entity
        entity = {
            "film_id": film_id,
            "filmVector": film_vector,
            "posterVector": poster_vector
        }

        # add the entity to the list
        entities.append(entity)

    collection.insert(entities)


def insert_data_using_milvus_client():
    client = MilvusClient(
        uri="http://10.100.30.11:19530"
    )

    # client.create_collection(
    #     collection_name="quick_setup",
    #     dimension=5
    # )
    #
    # res = client.get_load_state(
    #     collection_name="quick_setup"
    # )
    #
    # print(res)

    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )

    schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=5)
    schema.add_field(field_name="my_vector2", datatype=DataType.FLOAT_VECTOR, dim=5)

    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="my_id",
        index_type="STL_SORT"
    )

    index_params.add_index(
        field_name="my_vector",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": 128}
    )
    index_params.add_index(
        field_name="my_vector2",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": 128}
    )

    client.create_collection(
        collection_name="customized_setup_1",
        schema=schema,
        index_params=index_params
    )

    time.sleep(5)

    res = client.get_load_state(
        collection_name="customized_setup_1"
    )

    print(res)


def run_retriever():
    retriever = MilvusCollectionHybridSearchRetriever(
        ...
    )
    res_documents = retriever.invoke('foo')
    print(res_documents)


if __name__ == '__main__':
    # insert_data_using_orm()
    # insert_data_using_milvus_client()
    run_retriever()