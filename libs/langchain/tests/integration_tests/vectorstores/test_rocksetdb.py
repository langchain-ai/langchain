import logging
import os

import rockset
import rockset.models

from langchain.docstore.document import Document
from langchain.vectorstores.rocksetdb import Rockset
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
    fake_texts,
)

logger = logging.getLogger(__name__)


# To run these tests, make sure you have a collection with the name `langchain_demo`
# and the following ingest transformation:
#
#   SELECT
#       _input.* EXCEPT(_meta),
#       VECTOR_ENFORCE(_input.description_embedding, 10, 'float') as
#           description_embedding
#   FROM
#       _input
#
# We're using FakeEmbeddings utility to create text embeddings.
# It generates vector embeddings of length 10.
#
# Set env ROCKSET_DELETE_DOCS_ON_START=1 if you want to delete all docs from
# the collection before running any test. Be careful, this will delete any
# existing documents in your Rockset collection.
#
# See https://rockset.com/blog/introducing-vector-search-on-rockset/ for more details.

collection_name = "langchain_demo"
text_key = "description"
embedding_key = "description_embedding"


class TestRockset:
    rockset_vectorstore: Rockset

    @classmethod
    def setup_class(cls) -> None:
        assert os.environ.get("ROCKSET_API_KEY") is not None
        assert os.environ.get("ROCKSET_REGION") is not None

        api_key = os.environ.get("ROCKSET_API_KEY")
        region = os.environ.get("ROCKSET_REGION")
        if region == "use1a1":
            host = rockset.Regions.use1a1
        elif region == "usw2a1":
            host = rockset.Regions.usw2a1
        elif region == "euc1a1":
            host = rockset.Regions.euc1a1
        elif region == "dev":
            host = rockset.DevRegions.usw2a1
        else:
            logger.warn(
                "Using ROCKSET_REGION:%s as it is.. \
                You should know what you're doing...",
                region,
            )

            host = region

        client = rockset.RocksetClient(host, api_key)
        if os.environ.get("ROCKSET_DELETE_DOCS_ON_START") == "1":
            logger.info(
                "Deleting all existing documents from the Rockset collection %s",
                collection_name,
            )

            query_response = client.Queries.query(
                sql={"query": "select _id from {}".format(collection_name)}
            )
            ids = [
                str(r["_id"])
                for r in getattr(
                    query_response, query_response.attribute_map["results"]
                )
            ]
            logger.info("Existing ids in collection: %s", ids)
            client.Documents.delete_documents(
                collection=collection_name,
                data=[rockset.models.DeleteDocumentsRequestData(id=i) for i in ids],
            )

        embeddings = ConsistentFakeEmbeddings()
        embeddings.embed_documents(fake_texts)
        cls.rockset_vectorstore = Rockset(
            client, embeddings, collection_name, text_key, embedding_key
        )

    def test_rockset_insert_and_search(self) -> None:
        """Test end to end vector search in Rockset"""

        texts = ["foo", "bar", "baz"]
        metadatas = [{"metadata_index": i} for i in range(len(texts))]
        ids = self.rockset_vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
        )
        assert len(ids) == len(texts)
        # Test that `foo` is closest to `foo`
        output = self.rockset_vectorstore.similarity_search(
            query="foo", distance_func=Rockset.DistanceFunction.COSINE_SIM, k=1
        )
        assert output == [Document(page_content="foo", metadata={"metadata_index": 0})]

        # Find closest vector to `foo` which is not `foo`
        output = self.rockset_vectorstore.similarity_search(
            query="foo",
            distance_func=Rockset.DistanceFunction.COSINE_SIM,
            k=1,
            where_str="metadata_index != 0",
        )
        assert output == [Document(page_content="bar", metadata={"metadata_index": 1})]

    def test_build_query_sql(self) -> None:
        vector = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        q_str = self.rockset_vectorstore._build_query_sql(
            vector,
            Rockset.DistanceFunction.COSINE_SIM,
            4,
        )
        vector_str = ",".join(map(str, vector))
        expected = f"""\
SELECT * EXCEPT(description_embedding), \
COSINE_SIM(description_embedding, [{vector_str}]) as dist
FROM langchain_demo
ORDER BY dist DESC
LIMIT 4
"""
        assert q_str == expected

    def test_build_query_sql_with_where(self) -> None:
        vector = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        q_str = self.rockset_vectorstore._build_query_sql(
            vector,
            Rockset.DistanceFunction.COSINE_SIM,
            4,
            "age >= 10",
        )
        vector_str = ",".join(map(str, vector))
        expected = f"""\
SELECT * EXCEPT(description_embedding), \
COSINE_SIM(description_embedding, [{vector_str}]) as dist
FROM langchain_demo
WHERE age >= 10
ORDER BY dist DESC
LIMIT 4
"""
        assert q_str == expected
