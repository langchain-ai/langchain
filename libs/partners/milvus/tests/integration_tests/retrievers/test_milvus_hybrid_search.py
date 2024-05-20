import random
import unittest

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    WeightedRanker,
    connections,
)

from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from tests.integration_tests.utils import FakeEmbeddings

#
# To run this test properly, please start a Milvus server with the following command:
#
# ```shell
# wget https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh
# bash standalone_embed.sh start
# ```
#
# Here is the reference:
# https://milvus.io/docs/install_standalone-docker.md
#


class TestMilvusHybridSearch(unittest.TestCase):
    def setUp(self) -> None:
        self.connection_uri = (
            "http://localhost:19530"  # Replace with your Milvus server IP
        )
        self.insert_data_using_orm()

    def tearDown(self) -> None:
        self.collection.drop()

    def insert_data_using_orm(self) -> None:
        connections.connect(uri=self.connection_uri)
        dim = len(FakeEmbeddings().embed_query("foo"))
        fields = [
            FieldSchema(name="film_id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(
                name="filmVector", dtype=DataType.FLOAT_VECTOR, dim=dim
            ),  # Vector field for film vectors
            FieldSchema(
                name="posterVector", dtype=DataType.FLOAT_VECTOR, dim=dim
            ),  # Vector field for poster vectors
            FieldSchema(
                name="film_description", dtype=DataType.VARCHAR, max_length=65_535
            ),
        ]

        schema = CollectionSchema(fields=fields, enable_dynamic_field=False)

        self.collection = Collection(name="film_information", schema=schema)
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }

        self.collection.create_index("filmVector", index_params)
        self.collection.create_index("posterVector", index_params)

        entities = []

        for _ in range(1000):
            # generate random values for each field in the schema
            film_id = random.randint(1, 1000)
            film_vector = [random.random() for _ in range(dim)]
            poster_vector = [random.random() for _ in range(dim)]

            # creat a dictionary for each entity
            entity = {
                "film_id": film_id,
                "filmVector": film_vector,
                "posterVector": poster_vector,
                "film_description": "test_description",
            }

            # add the entity to the list
            entities.append(entity)

        self.collection.insert(entities)
        self.collection.load()

    def test_retriever(self) -> None:
        retriever = MilvusCollectionHybridSearchRetriever(
            collection=self.collection,
            rerank=WeightedRanker(0.5, 0.5),
            anns_fields=["filmVector", "posterVector"],
            field_embeddings=[FakeEmbeddings(), FakeEmbeddings()],
            top_k=5,
            text_field="film_description",
        )
        res_documents = retriever.invoke("foo")
        assert len(res_documents) == 5
