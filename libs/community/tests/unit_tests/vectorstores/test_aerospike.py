from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from langchain_community.vectorstores.aerospike import Aerospike
from langchain_community.vectorstores.utils import DistanceStrategy
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

pytestmark = pytest.mark.requires("aerospike_vector_search")


@pytest.fixture(scope="module")
def client():
    try:
        from aerospike_vector_search import Client
        from aerospike_vector_search.types import HostPort
    except ImportError:
        pytest.skip("aerospike_vector_search not installed")

    client = Client(
        seeds=[
            HostPort(host="dummy-host", port=3000),
        ],
    )

    yield client

    client.close()


@pytest.fixture
def mock_client(mocker):
    try:
        from aerospike_vector_search import Client
    except ImportError:
        pytest.skip("aerospike_vector_search not installed")

    return mocker.MagicMock(Client)


def test_aerospike(client) -> None:
    """Ensure an error is raised when search with score in hybrid mode
    because in this case Elasticsearch does not return any score.
    """
    import grpc

    query_string = "foo"
    embedding = FakeEmbeddings()

    store = Aerospike(
        client=client,
        embedding=embedding,
        text_key="text",
        vector_key="vector",
        index_name="dummy_index",
        namespace="test",
        set_name="testset",
        distance_strategy=DistanceStrategy.COSINE,
    )

    # TODO: Remove grpc import when aerospike_vector_search wraps grpc errors
    with pytest.raises(grpc.RpcError):
        store.similarity_search_by_vector(embedding.embed_query(query_string))


def test_init_bad_embedding(client):
    def bad_embedding():
        return None

    with pytest.warns(
        UserWarning,
        match="Passing in `embedding` as a Callable is deprecated. Please pass in an Embeddings object instead.",
    ):
        Aerospike(
            client=client,
            embedding=bad_embedding,
            text_key="text",
            vector_key="vector",
            index_name="dummy_index",
            namespace="test",
            set_name="testset",
            distance_strategy=DistanceStrategy.COSINE,
        )


def test_init_bad_client(client):
    class BadClient:
        pass

    with pytest.raises(
        ValueError,
        match="client should be an instance of aerospike_vector_search.Client, got <class 'tests.unit_tests.vectorstores.test_aerospike.test_init_bad_client.<locals>.BadClient'>",
    ):
        Aerospike(
            client=BadClient(),
            embedding=FakeEmbeddings(),
            text_key="text",
            vector_key="vector",
            index_name="dummy_index",
            namespace="test",
            set_name="testset",
            distance_strategy=DistanceStrategy.COSINE,
        )


def test_add_texts_wait_for_index_error(client):
    aerospike = Aerospike(
        client=client,
        embedding=FakeEmbeddings(),
        text_key="text",
        vector_key="vector",
        # index_name="dummy_index",
        namespace="test",
        set_name="testset",
        distance_strategy=DistanceStrategy.COSINE,
    )

    with pytest.raises(
        ValueError, match="if wait_for_index is True, index_name must be provided"
    ):
        aerospike.add_texts(["foo", "bar"], wait_for_index=True)


def test_similarity_search_by_vector_with_score_missing_index_name(client):
    aerospike = Aerospike(
        client=client,
        embedding=FakeEmbeddings(),
        text_key="text",
        vector_key="vector",
        # index_name="dummy_index",
        namespace="test",
        set_name="testset",
        distance_strategy=DistanceStrategy.COSINE,
    )

    with pytest.raises(ValueError, match="index_name must be provided"):
        aerospike.similarity_search_by_vector_with_score([1.0, 2.0, 3.0])


def test_similarity_search_by_vector_with_score_filters_missing_text_key(
    mock_client: MagicMock,
):
    from aerospike_vector_search.types import Neighbor

    text_key = "text"
    mock_client.vector_search.return_value = [
        Neighbor(key="key1", bins={text_key: 1}, distance=1.0),
        Neighbor(key="key2", bins={}, distance=0.0),
        Neighbor(key="key3", bins={text_key: 3}, distance=3.0),
    ]
    aerospike = Aerospike(
        client=mock_client,
        embedding=FakeEmbeddings(),
        text_key=text_key,
        vector_key="vector",
        index_name="dummy_index",
        namespace="test",
        set_name="testset",
        distance_strategy=DistanceStrategy.COSINE,
    )

    actual = aerospike.similarity_search_by_vector_with_score(
        [1.0, 2.0, 3.0], k=10, metadata_keys=["foo"]
    )

    expected = [
        (Document(page_content=1), 1.0),
        (Document(page_content=3), 3.0),
    ]
    mock_client.vector_search.assert_called_once_with(
        index_name="dummy_index",
        namespace="test",
        query=[1.0, 2.0, 3.0],
        limit=10,
        bin_names=[text_key, "foo"],
    )

    assert expected == actual


def test_similarity_search_by_vector_with_score_overwrite_index_name(
    mock_client: MagicMock,
):
    mock_client.vector_search.return_value = []
    aerospike = Aerospike(
        client=mock_client,
        embedding=FakeEmbeddings(),
        text_key="text",
        vector_key="vector",
        index_name="dummy_index",
        namespace="test",
        set_name="testset",
        distance_strategy=DistanceStrategy.COSINE,
    )

    aerospike.similarity_search_by_vector_with_score(
        [1.0, 2.0, 3.0], index_name="other_index"
    )

    mock_client.vector_search.assert_called_once_with(
        index_name="other_index",
        namespace="test",
        query=[1.0, 2.0, 3.0],
        limit=4,
        bin_names=None,
    )


@pytest.mark.parametrize(
    "distance_strategy,expected_fn",
    [
        (DistanceStrategy.COSINE, Aerospike._cosine_relevance_score_fn),
        (DistanceStrategy.EUCLIDEAN_DISTANCE, Aerospike._euclidean_relevance_score_fn),
        (DistanceStrategy.DOT_PRODUCT, Aerospike._max_inner_product_relevance_score_fn),
        (DistanceStrategy.JACCARD, ValueError),
    ],
)
def test_select_relevance_score_fn(client, distance_strategy, expected_fn):
    aerospike = Aerospike(
        client=client,
        embedding=FakeEmbeddings(),
        text_key="text",
        vector_key="vector",
        index_name="dummy_index",
        namespace="test",
        set_name="testset",
        distance_strategy=distance_strategy,
    )

    if expected_fn == ValueError:
        with pytest.raises(ValueError):
            aerospike._select_relevance_score_fn()

    else:
        fn = aerospike._select_relevance_score_fn()

        assert fn == expected_fn
