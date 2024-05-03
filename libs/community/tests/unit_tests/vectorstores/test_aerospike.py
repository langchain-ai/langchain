import pytest

from langchain_community.vectorstores.aerospike import Aerospike
from langchain_community.vectorstores.utils import DistanceStrategy
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


@pytest.mark.requires("aerospike_vector_search")
def test_aerospike() -> None:
    """Ensure an error is raised when search with score in hybrid mode
    because in this case Elasticsearch does not return any score.
    """
    import grpc
    from aerospike_vector_search import Client
    from aerospike_vector_search.types import HostPort

    query_string = "foo"
    embedding = FakeEmbeddings()

    store = Aerospike(
        client=Client(
            seeds=[
                HostPort(host="dummy-host", port=3000),
            ],
            is_loadbalancer=True, # TODO: remove after 0.5.1 client release
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
    with pytest.raises(
        grpc.RpcError
    ):
        store.similarity_search_by_vector(embedded_query)
