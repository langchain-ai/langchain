import pytest

from langchain_community.vectorstores.aerospike import Aerospike
from langchain_community.vectorstores.utils import DistanceStrategy
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


@pytest.mark.requires("aerospike_vector")
def test_aerospike() -> None:
    """Ensure an error is raised when search with score in hybrid mode
    because in this case Elasticsearch does not return any score.
    """
    from aerospike_vector.types import HostPort
    from aerospike_vector.vectordb_client import VectorDbClient
    import grpc

    query_string = "foo"
    embedding = FakeEmbeddings()

    store = Aerospike(
        client=VectorDbClient(
            seeds=[
                HostPort(host="dummy-host", port=3000),
            ]
        ),
        embedding=embedding,
        text_key="text",
        vector_key="vector",
        index_name="dummy_index",
        namespace="test",
        set_name="testset",
        distance_strategy=DistanceStrategy.COSINE,
    )

    embedded_query = embedding.embed_query(query_string)
    with pytest.raises(grpc.aio.AioRpcError): # TODO: Replace this when we have sync client
        store.similarity_search_by_vector(embedded_query)
