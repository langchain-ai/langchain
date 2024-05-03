"""Test Redis functionality."""

import inspect
import os
import subprocess
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pytest
from langchain_core.documents import Document

from langchain_community.vectorstores.aerospike import (
    Aerospike,
)
from langchain_community.vectorstores.utils import DistanceStrategy
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
    FakeEmbeddings,
)

if TYPE_CHECKING:
    from aerospike_vector_search import Client
    from aerospike_vector_search.admin import Client as AdminClient
    from aerospike_vector_search.types import HostPort

TEST_INDEX_NAME = "test-index"
TEST_NAMESPACE = "test"
TEST_AEROSPIKE_HOST_PORT = ("localhost", 5002)
TEXT_KEY = "_text"
VECTOR_KEY = "_vector"
ID_KEY = "_id"
EUCLIDEAN_SCORE = 1.0


@pytest.fixture
def texts() -> List[str]:
    return ["foo", "bar", "baz"]


def compose_up() -> None:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path += "/docker-compose/aerospike"
    subprocess.run(["docker", "compose", "up", "-d"], cwd=dir_path)

    time.sleep(10)


def compose_down() -> None:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path += "/docker-compose/aerospike"
    subprocess.run(["docker", "compose", "down"], cwd=dir_path)


@pytest.fixture(scope="class", autouse=True)
def docker_compose():
    compose_up()
    yield
    compose_down()


@pytest.fixture(scope="class")
def seeds():
    from aerospike_vector_search.types import HostPort

    yield HostPort(
        host=TEST_AEROSPIKE_HOST_PORT[0],
        port=TEST_AEROSPIKE_HOST_PORT[1],
    )


@pytest.fixture(scope="class")
def admin_client(seeds) -> Any:
    from aerospike_vector_search.admin import Client as AdminClient

    with AdminClient(seeds=seeds, is_loadbalancer=True) as admin_client:
        yield admin_client


@pytest.fixture(scope="class")
def client(seeds) -> Any:
    from aerospike_vector_search import Client

    with Client(seeds=seeds, is_loadbalancer=True) as client:
        yield client


@pytest.fixture
def embedder() -> Any:
    return ConsistentFakeEmbeddings()


@pytest.fixture
def aerospike(client, embedder) -> Aerospike:
    client: Client = client

    yield Aerospike(
        client,
        embedder,
        TEST_NAMESPACE,
        vector_key=VECTOR_KEY,
        text_key=TEXT_KEY,
        id_key=ID_KEY,
    )


def get_func_name() -> str:
    """
    Used to get the name of the calling function. The name is used for the index
    and set name in Aerospike tests for debugging purposes.
    """
    return inspect.stack()[1].function


"""
TODO: Add tests for delete()
"""


@pytest.mark.requires("aerospike_vector_search")
class TestAerospike:
    def test_from_text(self, client, admin_client, embedder: ConsistentFakeEmbeddings):
        admin_client: AdminClient = admin_client
        index_name = set_name = get_func_name()
        admin_client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
        )
        aerospike = Aerospike.from_texts(
            client,
            ["foo", "bar", "baz", "bay", "bax", "baw", "bav"],
            embedder,
            TEST_NAMESPACE,
            index_name=index_name,
            ids=["1", "2", "3", "4", "5", "6", "7"],
            set_name=set_name,
        )

        expected = [
            Document(
                page_content="foo",
                metadata={
                    ID_KEY: "1",
                    "_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                },
            ),
            Document(
                page_content="bar",
                metadata={
                    ID_KEY: "2",
                    "_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                },
            ),
            Document(
                page_content="baz",
                metadata={
                    ID_KEY: "3",
                    "_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
                },
            ),
        ]
        actual = aerospike.search(
            "foo", k=3, index_name=index_name, search_type="similarity"
        )

        actual = sorted(
            actual, key=lambda x: x.metadata[ID_KEY]
        )  # TODO: Remove this line after next release

        assert actual == expected

    def test_search_blocking(self, aerospike: Aerospike, admin_client):
        """Test end to end construction and search."""
        admin_client: AdminClient = admin_client
        index_name = set_name = get_func_name()
        admin_client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
        )

        aerospike.add_texts(
            ["foo", "bar", "baz"],
            ids=["1", "2", "3"],
            index_name=index_name,
            set_name=set_name,
        )  # Blocks until all vectors are indexed
        expected = [Document(page_content="foo", metadata={ID_KEY: "1"})]
        actual = aerospike.search(
            "foo",
            k=1,
            index_name=index_name,
            search_type="similarity",
            metadata_keys=[ID_KEY],
        )

        assert actual == expected

    def test_search_nonblocking(self, aerospike: Aerospike, admin_client):
        """Test end to end construction and search."""
        admin_client: AdminClient = admin_client
        index_name = set_name = get_func_name()
        admin_client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
        )

        aerospike.add_texts(
            ["foo", "bar", "baz"],
            ids=["1", "2", "3"],
            index_name=index_name,
            set_name=set_name,
            wait_for_index=True,
        )  # blocking
        aerospike.add_texts(
            ["bay"], index_name=index_name, set_name=set_name, wait_for_index=False
        )
        expected = [
            Document(page_content="foo", metadata={ID_KEY: "1"}),
            Document(page_content="bar", metadata={ID_KEY: "2"}),
            Document(page_content="baz", metadata={ID_KEY: "3"}),
        ]
        actual = aerospike.search(
            "foo",
            k=4,
            index_name=index_name,
            search_type="similarity",
            metadata_keys=[ID_KEY],
        )

        actual = sorted(
            actual, key=lambda x: x.metadata[ID_KEY]
        )  # TODO: Remove this line after next release

        # "bay"
        assert actual == expected

    def test_similarity_search_with_score(self, aerospike: Aerospike, admin_client):
        """Test end to end construction and search."""
        admin_client: AdminClient = admin_client
        expected = [(Document(page_content="foo", metadata={ID_KEY: "1"}), 0.0)]
        index_name = set_name = get_func_name()
        admin_client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
        )
        aerospike.add_texts(
            ["foo", "bar", "baz"],
            ids=["1", "2", "3"],
            index_name=index_name,
            set_name=set_name,
        )
        actual = aerospike.similarity_search_with_score(
            "foo", k=1, index_name=index_name, metadata_keys=[ID_KEY]
        )

        assert actual == expected

    def test_similarity_search_by_vector_with_score(
        self, aerospike: Aerospike, admin_client, embedder: ConsistentFakeEmbeddings
    ):
        """Test end to end construction and search."""
        admin_client: AdminClient = admin_client
        expected = [
            (Document(page_content="foo", metadata={"a": "b", ID_KEY: "1"}), 0.0)
        ]
        index_name = set_name = get_func_name()
        admin_client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
        )
        aerospike.add_texts(
            ["foo", "bar", "baz"],
            ids=["1", "2", "3"],
            index_name=index_name,
            set_name=set_name,
            metadatas=[{"a": "b", "1": "2"}, {"a": "c"}, {"a": "d"}],
        )
        actual = aerospike.similarity_search_by_vector_with_score(
            embedder.embed_query("foo"),
            k=1,
            index_name=index_name,
            metadata_keys=["a", ID_KEY],
        )

        assert actual == expected

    def test_similarity_search_by_vector(
        self, aerospike: Aerospike, admin_client, embedder: ConsistentFakeEmbeddings
    ):
        """Test end to end construction and search."""
        admin_client: AdminClient = admin_client
        expected = [
            Document(page_content="foo", metadata={"a": "b", ID_KEY: "1"}),
            Document(page_content="bar", metadata={"a": "c", ID_KEY: "2"}),
        ]
        index_name = set_name = get_func_name()
        admin_client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
        )
        aerospike.add_texts(
            ["foo", "bar", "baz"],
            ids=["1", "2", "3"],
            index_name=index_name,
            set_name=set_name,
            metadatas=[{"a": "b", "1": "2"}, {"a": "c"}, {"a": "d"}],
        )
        actual = aerospike.similarity_search_by_vector(
            embedder.embed_query("foo"),
            k=2,
            index_name=index_name,
            metadata_keys=["a", ID_KEY],
        )

        actual = sorted(
            actual, key=lambda x: x.metadata[ID_KEY]
        )  # TODO: Remove this line after next release

        assert actual == expected

    def test_similarity_search(self, aerospike: Aerospike, admin_client):
        """Test end to end construction and search."""
        admin_client: AdminClient = admin_client
        expected = [
            Document(page_content="foo", metadata={ID_KEY: "1"}),
            Document(page_content="bar", metadata={ID_KEY: "2"}),
            Document(page_content="baz", metadata={ID_KEY: "3"}),
        ]
        index_name = set_name = get_func_name()
        admin_client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
        )
        aerospike.add_texts(
            ["foo", "bar", "baz"],
            ids=["1", "2", "3"],
            index_name=index_name,
            set_name=set_name,
        )  # blocking
        actual = aerospike.similarity_search(
            "foo", k=3, index_name=index_name, metadata_keys=[ID_KEY]
        )

        actual = sorted(
            actual, key=lambda x: x.metadata[ID_KEY]
        )  # TODO: Remove this line after next release

        assert actual == expected

    def test_max_marginal_relevance_search_by_vector(
        self, client, admin_client, embedder: ConsistentFakeEmbeddings
    ):
        """Test max marginal relevance search."""
        admin_client: AdminClient = admin_client
        index_name = set_name = get_func_name()
        admin_client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
        )
        aerospike = Aerospike.from_texts(
            client,
            ["foo", "bar", "baz", "bay", "bax", "baw", "bav"],
            embedder,
            TEST_NAMESPACE,
            index_name=index_name,
            ids=["1", "2", "3", "4", "5", "6", "7"],
            set_name=set_name,
        )

        mmr_output = aerospike.max_marginal_relevance_search_by_vector(
            embedder.embed_query("foo"), index_name=index_name, k=3, fetch_k=3
        )
        sim_output = aerospike.similarity_search("foo", index_name=index_name, k=3)
        mmr_output = sorted(
            mmr_output, key=lambda x: x.metadata[ID_KEY]
        )  # TODO: Remove this line after next release
        sim_output = sorted(
            sim_output, key=lambda x: x.metadata[ID_KEY]
        )  # TODO: Remove this line after next release
        assert len(mmr_output) == 3
        assert mmr_output == sim_output

        mmr_output = aerospike.max_marginal_relevance_search_by_vector(
            embedder.embed_query("foo"), index_name=index_name, k=2, fetch_k=3
        )
        mmr_output = sorted(
            mmr_output, key=lambda x: x.metadata[ID_KEY]
        )  # TODO: Remove this line after next release
        assert len(mmr_output) == 2
        assert mmr_output[0].page_content == "foo"
        assert mmr_output[1].page_content == "bar"

        mmr_output = aerospike.max_marginal_relevance_search_by_vector(
            embedder.embed_query("foo"),
            index_name=index_name,
            k=2,
            fetch_k=3,
            lambda_mult=0.1,  # more diversity
        )
        mmr_output = sorted(
            mmr_output, key=lambda x: x.metadata[ID_KEY]
        )  # TODO: Remove this line after next release
        assert len(mmr_output) == 2
        assert mmr_output[0].page_content == "foo"
        assert mmr_output[1].page_content == "baz"

        # if fetch_k < k, then the output will be less than k
        mmr_output = aerospike.max_marginal_relevance_search_by_vector(
            embedder.embed_query("foo"), index_name=index_name, k=3, fetch_k=2
        )
        assert len(mmr_output) == 2

    def test_max_marginal_relevance_search(
        self, aerospike: Aerospike, admin_client
    ) -> None:
        """Test max marginal relevance search."""
        admin_client: AdminClient = admin_client
        index_name = set_name = get_func_name()
        admin_client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
        )
        aerospike.add_texts(
            ["foo", "bar", "baz", "bay", "bax", "baw", "bav"],
            ids=["1", "2", "3", "4", "5", "6", "7"],
            index_name=index_name,
            set_name=set_name,
        )

        mmr_output = aerospike.max_marginal_relevance_search(
            "foo", index_name=index_name, k=3, fetch_k=3
        )
        sim_output = aerospike.similarity_search("foo", index_name=index_name, k=3)
        mmr_output = sorted(
            mmr_output, key=lambda x: x.metadata[ID_KEY]
        )  # TODO: Remove this line after next release
        sim_output = sorted(
            sim_output, key=lambda x: x.metadata[ID_KEY]
        )  # TODO: Remove this line after next release
        assert len(mmr_output) == 3
        assert mmr_output == sim_output

        mmr_output = aerospike.max_marginal_relevance_search(
            "foo", index_name=index_name, k=2, fetch_k=3
        )
        mmr_output = sorted(
            mmr_output, key=lambda x: x.metadata[ID_KEY]
        )  # TODO: Remove this line after next release
        assert len(mmr_output) == 2
        assert mmr_output[0].page_content == "foo"
        assert mmr_output[1].page_content == "bar"

        mmr_output = aerospike.max_marginal_relevance_search(
            "foo",
            index_name=index_name,
            k=2,
            fetch_k=3,
            lambda_mult=0.1,  # more diversity
        )
        mmr_output = sorted(
            mmr_output, key=lambda x: x.metadata[ID_KEY]
        )  # TODO: Remove this line after next release
        assert len(mmr_output) == 2
        assert mmr_output[0].page_content == "foo"
        assert mmr_output[1].page_content == "baz"

        # if fetch_k < k, then the output will be less than k
        mmr_output = aerospike.max_marginal_relevance_search(
            "foo", index_name=index_name, k=3, fetch_k=2
        )
        assert len(mmr_output) == 2

    def test_cosine_distance(self, aerospike: Aerospike, admin_client) -> None:
        """Test cosine distance."""
        from aerospike_vector_search import types

        admin_client: AdminClient = admin_client
        index_name = set_name = get_func_name()
        admin_client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
            vector_distance_metric=types.VectorDistanceMetric.COSINE,
        )
        aerospike.add_texts(
            ["foo", "bar", "baz"],
            ids=["1", "2", "3"],
            index_name=index_name,
            set_name=set_name,
        )  # blocking

        """
        foo vector = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        far vector = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0]
        cosine similarity ~= 0.71
        cosine distance ~= 1 - cosine similarity = 0.29
        """
        expected = pytest.approx(0.292, abs=0.002)
        output = aerospike.similarity_search_with_score(
            "far", index_name=index_name, k=3
        )
        output.sort(
            key=lambda x: x[0].metadata[ID_KEY]
        )  # TODO: Remove this line after next release
        _, actual_score = output[0]

        assert actual_score == expected

    def test_dot_product_distance(self, aerospike: Aerospike, admin_client) -> None:
        """Test dot product distance."""
        from aerospike_vector_search import types

        admin_client: AdminClient = admin_client
        index_name = set_name = get_func_name()
        admin_client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
            vector_distance_metric=types.VectorDistanceMetric.DOT_PRODUCT,
        )
        aerospike.add_texts(
            ["foo", "bar", "baz"],
            ids=["1", "2", "3"],
            index_name=index_name,
            set_name=set_name,
        )  # blocking

        """
        foo vector = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        far vector = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0]
        dot product = 9.0
        dot product distance = dot product * -1 = -9.0
        """
        expected = -9.0
        output = aerospike.similarity_search_with_score(
            "far", index_name=index_name, k=3
        )
        output.sort(
            key=lambda x: x[0].metadata[ID_KEY]
        )  # TODO: Remove this line after next release
        _, actual_score = output[0]

        assert actual_score == expected

    def test_euclidean_distance(self, aerospike: Aerospike, admin_client) -> None:
        """Test dot product distance."""
        from aerospike_vector_search import types

        admin_client: AdminClient = admin_client
        index_name = set_name = get_func_name()
        admin_client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
            vector_distance_metric=types.VectorDistanceMetric.SQUARED_EUCLIDEAN,
        )
        aerospike.add_texts(
            ["foo", "bar", "baz"],
            ids=["1", "2", "3"],
            index_name=index_name,
            set_name=set_name,
        )  # blocking

        """
        foo vector = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        far vector = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0]
        euclidean distance = 9.0
        """
        expected = 9.0
        output = aerospike.similarity_search_with_score(
            "far", index_name=index_name, k=3
        )
        output.sort(
            key=lambda x: x[0].metadata[ID_KEY]
        )  # TODO: Remove this line after next release
        _, actual_score = output[0]

        assert actual_score == expected

    def test_as_retriever(self, aerospike: Aerospike, admin_client) -> None:
        admin_client: AdminClient = admin_client
        index_name = set_name = get_func_name()
        admin_client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
        )
        aerospike.add_texts(
            ["foo", "foo", "foo", "foo", "bar"],
            ids=["1", "2", "3"],
            index_name=index_name,
            set_name=set_name,
        )  # blocking

        aerospike._index_name = index_name
        retriever = aerospike.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )
        results = retriever.invoke("foo")
        assert len(results) == 3
        assert all([d.page_content == "foo" for d in results])

    def test_as_retriever_distance_threshold(
        self, aerospike: Aerospike, admin_client
    ) -> None:
        admin_client: AdminClient = admin_client
        index_name = set_name = get_func_name()
        admin_client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
        )
        aerospike.add_texts(
            ["foo", "foo", "foo", "bar", "bar", "bar", "bar", "bar"],
            ids=["1", "2", "3"],
            index_name=index_name,
            set_name=set_name,
        )  # blocking

        aerospike._index_name = index_name
        retriever = aerospike.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 9, "score_threshold": 0.1},
        )
        results = retriever.invoke("foo")

        for r in results:
            assert r.page_content == "foo"

        assert len(results) == 3
