import sys
from typing import Any, Callable, Generator
from unittest.mock import MagicMock, Mock, call

import pytest
from langchain_core.documents import Document

from langchain_community.vectorstores.aerospike import Aerospike
from langchain_community.vectorstores.utils import DistanceStrategy
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

pytestmark = pytest.mark.requires("aerospike_vector_search") and pytest.mark.skipif(
    sys.version_info < (3, 9), reason="requires python3.9 or higher"
)


@pytest.fixture(scope="module")
def client() -> Generator[Any, None, None]:
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
def mock_client(mocker: Any) -> None:
    try:
        from aerospike_vector_search import Client
    except ImportError:
        pytest.skip("aerospike_vector_search not installed")

    return mocker.MagicMock(Client)


def test_aerospike(client: Any) -> None:
    """Ensure an error is raised when search with score in hybrid mode
    because in this case Elasticsearch does not return any score.
    """
    from aerospike_vector_search import AVSError

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
    with pytest.raises(AVSError):
        store.similarity_search_by_vector(embedding.embed_query(query_string))


def test_init_aerospike_distance(client: Any) -> None:
    from aerospike_vector_search.types import VectorDistanceMetric

    embedding = FakeEmbeddings()
    aerospike = Aerospike(
        client=client,
        embedding=embedding,
        text_key="text",
        vector_key="vector",
        index_name="dummy_index",
        namespace="test",
        set_name="testset",
        distance_strategy=VectorDistanceMetric.COSINE,
    )

    assert aerospike._distance_strategy == DistanceStrategy.COSINE


def test_init_bad_embedding(client: Any) -> None:
    def bad_embedding() -> None:
        return None

    with pytest.warns(
        UserWarning,
        match=(
            "Passing in `embedding` as a Callable is deprecated. Please pass"
            + " in an Embeddings object instead."
        ),
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


def test_init_bad_client(client: Any) -> None:
    class BadClient:
        pass

    with pytest.raises(
        ValueError,
        match=(
            "client should be an instance of aerospike_vector_search.Client,"
            + " got <class 'tests.unit_tests.vectorstores.test_aerospike."
            + "test_init_bad_client.<locals>.BadClient'>"
        ),
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


def test_convert_distance_strategy(client: Any) -> None:
    from aerospike_vector_search.types import VectorDistanceMetric

    aerospike = Aerospike(
        client=client,
        embedding=FakeEmbeddings(),
        text_key="text",
        vector_key="vector",
        index_name="dummy_index",
        namespace="test",
        set_name="testset",
        distance_strategy=DistanceStrategy.COSINE,
    )

    converted_strategy = aerospike.convert_distance_strategy(
        VectorDistanceMetric.COSINE
    )
    assert converted_strategy == DistanceStrategy.COSINE

    converted_strategy = aerospike.convert_distance_strategy(
        VectorDistanceMetric.DOT_PRODUCT
    )
    assert converted_strategy == DistanceStrategy.DOT_PRODUCT

    converted_strategy = aerospike.convert_distance_strategy(
        VectorDistanceMetric.SQUARED_EUCLIDEAN
    )
    assert converted_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE

    with pytest.raises(ValueError):
        aerospike.convert_distance_strategy(VectorDistanceMetric.HAMMING)


def test_add_texts_wait_for_index_error(client: Any) -> None:
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


def test_add_texts_returns_ids(mock_client: MagicMock) -> None:
    aerospike = Aerospike(
        client=mock_client,
        embedding=FakeEmbeddings(),
        text_key="text",
        vector_key="vector",
        namespace="test",
        set_name="testset",
        distance_strategy=DistanceStrategy.COSINE,
    )

    excepted = ["0", "1"]
    actual = aerospike.add_texts(
        ["foo", "bar"],
        metadatas=[{"foo": 0}, {"bar": 1}],
        ids=["0", "1"],
        set_name="otherset",
        index_name="dummy_index",
        wait_for_index=True,
    )

    assert excepted == actual
    mock_client.upsert.assert_has_calls(
        calls=[
            call(
                namespace="test",
                key="0",
                set_name="otherset",
                record_data={
                    "_id": "0",
                    "text": "foo",
                    "vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    "foo": 0,
                },
            ),
            call(
                namespace="test",
                key="1",
                set_name="otherset",
                record_data={
                    "_id": "1",
                    "text": "bar",
                    "vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    "bar": 1,
                },
            ),
        ]
    )
    mock_client.wait_for_index_completion.assert_called_once_with(
        namespace="test",
        name="dummy_index",
    )


def test_delete_returns_false(mock_client: MagicMock) -> None:
    from aerospike_vector_search import AVSServerError

    mock_client.delete.side_effect = Mock(side_effect=AVSServerError(rpc_error=""))
    aerospike = Aerospike(
        client=mock_client,
        embedding=FakeEmbeddings(),
        text_key="text",
        vector_key="vector",
        namespace="test",
        set_name="testset",
        distance_strategy=DistanceStrategy.COSINE,
    )

    assert not aerospike.delete(["foo", "bar"], set_name="testset")
    mock_client.delete.assert_called_once_with(
        namespace="test", key="foo", set_name="testset"
    )


def test_similarity_search_by_vector_with_score_missing_index_name(
    client: Any,
) -> None:
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
) -> None:
    from aerospike_vector_search.types import Neighbor

    text_key = "text"
    mock_client.vector_search.return_value = [
        Neighbor(key="key1", fields={text_key: 1}, distance=1.0),
        Neighbor(key="key2", fields={}, distance=0.0),
        Neighbor(key="key3", fields={text_key: 3}, distance=3.0),
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
        (Document(page_content="1"), 1.0),
        (Document(page_content="3"), 3.0),
    ]
    mock_client.vector_search.assert_called_once_with(
        index_name="dummy_index",
        namespace="test",
        query=[1.0, 2.0, 3.0],
        limit=10,
        field_names=[text_key, "foo"],
    )

    assert expected == actual


def test_similarity_search_by_vector_with_score_overwrite_index_name(
    mock_client: MagicMock,
) -> None:
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
        field_names=None,
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
def test_select_relevance_score_fn(
    client: Any, distance_strategy: DistanceStrategy, expected_fn: Callable
) -> None:
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

    if expected_fn is ValueError:
        with pytest.raises(ValueError):
            aerospike._select_relevance_score_fn()

    else:
        fn = aerospike._select_relevance_score_fn()

        assert fn == expected_fn
