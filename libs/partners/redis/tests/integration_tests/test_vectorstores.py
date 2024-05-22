import math
import os
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_openai import OpenAIEmbeddings
from redis import Redis
from redisvl.index import SearchIndex  # type: ignore
from redisvl.query import CountQuery, VectorQuery  # type: ignore
from redisvl.query.filter import (  # type: ignore
    FilterExpression,
    Geo,
    GeoRadius,
    Num,
    Tag,
    Text,
)
from redisvl.schema import IndexSchema  # type: ignore
from ulid import ULID

from langchain_redis import RedisConfig, RedisVectorStore

TEST_INDEX_NAME = "test"
TEST_SINGLE_RESULT = [Document(page_content="foo")]
TEST_RESULT = [Document(page_content="foo"), Document(page_content="foo")]


class CustomTestEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        Convert input text to a 'vector' (list of floats).
        If the text is a number, use it as the angle in radians.
        Any other input text becomes the zero vector!
        """
        try:
            angle = float(text)
            return [math.cos(angle), math.sin(angle)]
        except ValueError:
            return [0.0, 0.0]


@pytest.fixture
def redis_url() -> str:
    return "redis://localhost:6379"


@pytest.fixture
def texts() -> List[str]:
    return ["foo", "bar", "baz"]


def test_with_redis_url(texts: List[str], redis_url: str) -> None:
    """Test end to end construction and search."""
    # Create a unique index name for testing
    index_name = f"test_index_{str(ULID())}"
    result = RedisVectorStore.from_texts(
        texts,
        OpenAIEmbeddings(),
        index_name=index_name,
        key_prefix="tst1",
        redis_url=redis_url,
    )
    vector_store = cast(RedisVectorStore, result)
    output = vector_store.similarity_search("foo", k=1, return_metadata=False)
    assert output == TEST_SINGLE_RESULT
    # Clean up
    vector_store.index.delete(drop=True)


def test_with_existing_redis_client(texts: List[str], redis_url: str) -> None:
    """Test end to end construction and search."""
    # Create a unique index name for testing
    index_name = f"test_index_{str(ULID())}"
    result = RedisVectorStore.from_texts(
        texts,
        OpenAIEmbeddings(),
        index_name=index_name,
        key_prefix="tst1",
        redis_client=Redis.from_url("redis://localhost:6379"),
    )
    vector_store = cast(RedisVectorStore, result)
    output = vector_store.similarity_search("foo", k=1, return_metadata=False)
    assert output == TEST_SINGLE_RESULT
    # Clean up
    vector_store.index.delete(drop=True)


def test_redis_new_vector(texts: List[str], redis_url: str) -> None:
    """Test adding a new document"""
    # Create a unique index name for testing
    index_name = f"test_index_{str(ULID())}"
    vector_store = RedisVectorStore.from_texts(
        texts,
        OpenAIEmbeddings(),
        index_name=index_name,
        key_prefix="tst2",
        redis_url=redis_url,
    )
    vector_store.add_texts(["foo"])
    output = vector_store.similarity_search("foo", k=2, return_metadata=False)
    assert output == TEST_RESULT
    # Clean up
    vector_store.index.delete(drop=True)


def test_redis_from_existing(texts: List[str], redis_url: str) -> None:
    """Test adding a new document"""
    # Create a unique index name for testing
    index_name = f"test_index_{str(ULID())}"
    vector_store = RedisVectorStore.from_texts(
        texts,
        OpenAIEmbeddings(),
        index_name=index_name,
        key_prefix="tst3",
        redis_url=redis_url,
    )

    # write schema for the next test
    vector_store.index.schema.to_yaml("test_schema.yml")

    # Test creating from an existing
    vector_store2 = RedisVectorStore(
        OpenAIEmbeddings(),
        index_name=index_name,
        redis_url=redis_url,
        from_existing=True,
    )
    output = vector_store2.similarity_search("foo", k=1, return_metadata=False)
    assert output == TEST_SINGLE_RESULT


def test_redis_add_texts_to_existing(redis_url: str) -> None:
    """Test adding a new document"""
    # Test creating from an existing with yaml from file
    vector_store = RedisVectorStore(
        OpenAIEmbeddings(),
        index_name=TEST_INDEX_NAME,
        redis_url=redis_url,
        schema_path="test_schema.yml",
    )
    vector_store.add_texts(["foo"])
    output = vector_store.similarity_search("foo", k=2, return_metadata=False)
    assert output == TEST_RESULT
    # remove the test_schema.yml file
    os.remove("test_schema.yml")

    # Clean up
    vector_store.index.delete(drop=True)


def test_redis_from_texts_return_keys(redis_url: str, texts: List[str]) -> None:
    """Test from_texts_return_keys constructor."""
    # Create a unique index name for testing
    index_name = f"test_index_{str(ULID())}"
    result = RedisVectorStore.from_texts(
        texts,
        OpenAIEmbeddings(),
        index_name=index_name,
        key_prefix="tst4",
        return_keys=True,
        redis_url=redis_url,
    )

    vector_store, keys = cast(Tuple[RedisVectorStore, List[str]], result)

    output = vector_store.similarity_search("foo", k=1, return_metadata=False)
    assert output == TEST_SINGLE_RESULT
    assert len(keys) == len(texts)

    # Clean up
    vector_store.index.delete(drop=True)


def test_redis_from_documents(redis_url: str, texts: List[str]) -> None:
    """Test from_documents constructor."""
    docs = [Document(page_content=t, metadata={"a": "b"}) for t in texts]
    metadata_schema = [
        {"name": "a", "type": "tag"},
    ]
    vector_store = RedisVectorStore.from_documents(
        docs,
        OpenAIEmbeddings(),
        key_prefix="tst5",
        metadata_schema=metadata_schema,
        redis_url=redis_url,
    )
    output = vector_store.similarity_search("foo", k=1, return_metadata=True)

    assert "a" in output[0].metadata.keys()
    assert "b" in output[0].metadata.values()
    # Clean up
    vector_store.index.delete(drop=True)


def test_from_texts(redis_url: str) -> None:
    """Test end to end construction and search."""
    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create a unique index name for testing
    index_name = f"test_index_{str(ULID())}"

    texts = ["foo", "bar", "baz"]

    # Create the RedisVectorStore
    vector_store = RedisVectorStore.from_texts(
        texts,
        embeddings,
        index_name=index_name,
        key_prefix="tst6",
        redis_url=redis_url,
    )

    count_query = CountQuery(FilterExpression("*"))

    count = vector_store.index.query(count_query)

    assert 3 == count

    # Clean up
    vector_store.index.delete(drop=True)


def test_custom_keys(texts: List[str], redis_url: str) -> None:
    keys_in = ["test_key_1", "test_key_2", "test_key_3"]

    index_name = f"test_index_{str(ULID())}"
    result = RedisVectorStore.from_texts(
        texts,
        OpenAIEmbeddings(),
        index_name=index_name,
        key_prefix="tst7",
        keys=keys_in,
        return_keys=True,
        redis_url=redis_url,
    )
    vector_store, keys_out = cast(Tuple[RedisVectorStore, List[str]], result)

    # it will append the index key prefix to all keys
    assert keys_out == [f"{vector_store.key_prefix}:{key}" for key in keys_in]

    # Clean up
    vector_store.index.delete(drop=True)


def test_custom_keys_from_docs(texts: List[str], redis_url: str) -> None:
    keys_in = ["test_key_1", "test_key_2", "test_key_3"]
    docs = [Document(page_content=t, metadata={"a": "b"}) for t in texts]

    index_name = f"test_index_{str(ULID())}"
    result = RedisVectorStore.from_documents(
        docs,
        OpenAIEmbeddings(),
        index_name=index_name,
        key_prefix="tst8",
        keys=keys_in,
        return_keys=True,
        redis_url=redis_url,
    )
    vector_store, keys_out = cast(Tuple[RedisVectorStore, List[str]], result)

    client = vector_store.index.client
    # test keys are correct
    assert client.hget(f"{vector_store.key_prefix}:test_key_1", "text")
    # test metadata is stored
    assert client.hget(f"{vector_store.key_prefix}:test_key_1", "a") == bytes(
        "b", "utf-8"
    )
    # test all keys are stored
    assert client.hget(f"{vector_store.key_prefix}:test_key_2", "text")
    # Clean up
    vector_store.index.delete(drop=True)


# -- test filters -- #


@pytest.mark.parametrize(
    "filter_expr, expected_length, expected_nums",
    [
        (Text("val") == "foo", 1, None),
        (Text("val") == "foo", 1, None),
        (Text("val") % "ba*", 2, ["bar", "baz"]),
        (Num("num") > 2, 1, [3]),
        (Num("num") < 2, 1, [1]),
        (Num("num") >= 2, 2, [2, 3]),
        (Num("num") <= 2, 2, [1, 2]),
        (Num("num") != 2, 2, [1, 3]),
        (Num("num") != 2, 2, [1, 3]),
        (Tag("category") == "a", 3, None),
        (Tag("category") == "b", 2, None),
        (Tag("category") == "c", 2, None),
        (Tag("category") == ["b", "c"], 3, None),
    ],
    ids=[
        "text-filter-equals-foo",
        "alternative-text-equals-foo",
        "text-filter-fuzzy-match-ba",
        "number-filter-greater-than-2",
        "number-filter-less-than-2",
        "number-filter-greater-equals-2",
        "number-filter-less-equals-2",
        "number-filter-not-equals-2",
        "alternative-number-not-equals-2",
        "tag-filter-equals-a",
        "tag-filter-equals-b",
        "tag-filter-equals-c",
        "tag-filter-equals-b-or-c",
    ],
)
def test_redis_similarity_search_with_filters(
    filter_expr: FilterExpression,
    expected_length: int,
    expected_nums: Optional[List[Union[str, int]]],
    redis_url: str,
) -> None:
    metadata = [
        {"name": "joe", "num": 1, "val": "foo", "category": ["a", "b"]},
        {"name": "john", "num": 2, "val": "bar", "category": ["a", "c"]},
        {"name": "jane", "num": 3, "val": "baz", "category": ["b", "c", "a"]},
    ]
    documents = [Document(page_content="foo", metadata=m) for m in metadata]
    metadata_schema = [
        {"name": "name", "type": "tag"},
        {"name": "num", "type": "numeric"},
        {"name": "val", "type": "text"},
        {"name": "category", "type": "tag"},
    ]
    index_name = f"test_index_{str(ULID())}"
    vector_store = RedisVectorStore.from_documents(
        documents,
        OpenAIEmbeddings(),
        index_name=index_name,
        key_prefix="tst9",
        metadata_schema=metadata_schema,
        redis_url=redis_url,
    )

    sim_output = vector_store.similarity_search("foo", k=3, filter=filter_expr)

    assert len(sim_output) == expected_length

    if expected_nums is not None:
        for out in sim_output:
            assert (
                out.metadata["val"] in expected_nums
                or int(out.metadata["num"]) in expected_nums
            )

    # Clean up
    vector_store.index.delete(drop=True)


@pytest.mark.parametrize(
    "filter_expr, expected_length, expected_nums",
    [
        (Text("val") == "foo", 1, None),
        (Text("val") == "foo", 1, None),
        (Text("val") % "ba*", 2, ["bar", "baz"]),
        (Num("num") > 2, 1, [3]),
        (Num("num") < 2, 1, [1]),
        (Num("num") >= 2, 2, [2, 3]),
        (Num("num") <= 2, 2, [1, 2]),
        (Num("num") != 2, 2, [1, 3]),
        (Num("num") != 2, 2, [1, 3]),
        (Tag("category") == "a", 3, None),
        (Tag("category") == "b", 2, None),
        (Tag("category") == "c", 2, None),
        (Tag("category") == ["b", "c"], 3, None),
    ],
    ids=[
        "text-filter-equals-foo",
        "alternative-text-equals-foo",
        "text-filter-fuzzy-match-ba",
        "number-filter-greater-than-2",
        "number-filter-less-than-2",
        "number-filter-greater-equals-2",
        "number-filter-less-equals-2",
        "number-filter-not-equals-2",
        "alternative-number-not-equals-2",
        "tag-filter-equals-a",
        "tag-filter-equals-b",
        "tag-filter-equals-c",
        "tag-filter-equals-b-or-c",
    ],
)
def test_redis_mmr_with_filters(
    filter_expr: FilterExpression,
    expected_length: int,
    expected_nums: Optional[List[Union[str, int]]],
    redis_url: str,
) -> None:
    metadata = [
        {"name": "joe", "num": 1, "val": "foo", "category": ["a", "b"]},
        {"name": "john", "num": 2, "val": "bar", "category": ["a", "c"]},
        {"name": "jane", "num": 3, "val": "baz", "category": ["b", "c", "a"]},
    ]
    documents = [Document(page_content="foo", metadata=m) for m in metadata]
    metadata_schema = [
        {"name": "name", "type": "tag"},
        {"name": "num", "type": "numeric"},
        {"name": "val", "type": "text"},
        {"name": "category", "type": "tag"},
    ]
    index_name = f"test_index_{str(ULID())}"
    vector_store = RedisVectorStore.from_documents(
        documents,
        OpenAIEmbeddings(),
        index_name=index_name,
        key_prefix="tst10",
        metadata_schema=metadata_schema,
        redis_url=redis_url,
    )

    mmr_output = vector_store.max_marginal_relevance_search(
        "foo", k=3, fetch_k=5, filter=filter_expr
    )

    assert len(mmr_output) == expected_length

    if expected_nums is not None:
        for out in mmr_output:
            assert (
                out.metadata["val"] in expected_nums
                or int(out.metadata["num"]) in expected_nums
            )

    # Clean up
    vector_store.index.delete(drop=True)


def test_similarity_search(redis_url: str) -> None:
    """Test end to end construction and search."""
    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create a unique index name for testing
    index_name = f"test_index_{str(ULID())}"

    texts = ["foo", "bar", "baz"]

    # Create the RedisVectorStore
    vector_store = RedisVectorStore.from_texts(
        texts,
        embeddings,
        index_name=index_name,
        key_prefix="tst11",
        redis_url=redis_url,
    )

    # Perform similarity search
    output = vector_store.similarity_search("foo", k=1)
    assert output == [
        Document(
            page_content="foo",
            metadata={},
        )
    ]

    # Clean up
    vector_store.index.delete(drop=True)


def test_similarity_search_with_scores(redis_url: str) -> None:
    """Test end to end construction and search."""
    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create a unique index name for testing
    index_name = f"test_index_{str(ULID())}"

    texts = ["apple", "orange", "hammer"]

    # Create the RedisVectorStore
    vector_store = RedisVectorStore.from_texts(
        texts,
        embeddings,
        index_name=index_name,
        key_prefix="tst12",
        redis_url=redis_url,
    )

    # Perform similarity search
    output = vector_store.similarity_search_with_score("apple", k=3)
    docs = [o[0] for o in output]
    scores = [o[1] for o in output]

    assert docs == [
        Document(page_content="apple"),
        Document(page_content="orange"),
        Document(page_content="hammer"),
    ]

    # Check that "apple" has the highest similarity score
    assert scores[0] == min(scores)

    # Check that "orange" has a higher similarity score than "hammer"
    assert scores[1] < scores[2]

    # Clean up
    vector_store.index.delete(drop=True)


def test_add_texts(redis_url: str) -> None:
    """Test adding texts to an existing index."""
    embeddings = OpenAIEmbeddings()
    index_name = f"test_index_{str(ULID())}"

    init_texts = ["foo", "bar", "baz"]
    texts = ["bay", "bax", "baw", "bav"]

    vector_store = RedisVectorStore.from_texts(
        init_texts,
        embeddings,
        index_name=index_name,
        key_prefix="tst13",
        redis_url=redis_url,
    )

    vector_store.add_texts(texts)

    count_query = CountQuery(FilterExpression("*"))
    count = vector_store.index.query(count_query)

    assert 7 == count

    # Clean up
    vector_store.index.delete(drop=True)


def test_similarity_search_with_metadata_filtering(redis_url: str) -> None:
    """Test metadata storage and retrieval."""
    texts = [
        "The Toyota Camry is a reliable and comfortable family sedan.",
        "The Honda Civic is a compact car known for its fuel efficiency.",
        "The Ford Mustang is an iconic American muscle car with powerful engines.",
    ]
    metadatas = [
        {
            "color": "red",
            "brand": "Toyota",
            "model": "Camry",
            "msrp": "25000",
            "location": "-122.4194,37.7749",
            "review": "The Camry offers a smooth ride and great value for money.",
        },
        {
            "color": "blue",
            "brand": "Honda",
            "model": "Civic",
            "msrp": "22000",
            "location": "-122.3301,47.6062",
            "review": "The Civic is a reliable choice for daily commutes.",
        },
        {
            "color": "green",
            "brand": "Ford",
            "model": "Mustang",
            "msrp": "35000",
            "location": "-84.3880,33.7490",
            "review": "The Mustang is a thrilling car to drive with its \
                powerful engine.",
        },
    ]

    embeddings = OpenAIEmbeddings()
    index_name = f"test_index_{str(ULID())}"

    # Define the index schema with metadata fields
    metadata_schema = [
        {"name": "color", "type": "tag"},
        {"name": "brand", "type": "tag"},
        {"name": "model", "type": "text"},
        {"name": "msrp", "type": "numeric"},
        {"name": "location", "type": "geo"},
        {"name": "review", "type": "text"},
    ]

    vector_store = RedisVectorStore.from_texts(
        texts,
        embeddings,
        index_name=index_name,
        key_prefix="car",
        metadatas=metadatas,
        redis_url=redis_url,
        metadata_schema=metadata_schema,
    )

    # Perform similarity search with metadata filtering
    filtered_metadata1 = (Tag("color") == "blue") & (Tag("brand") == "Honda")
    output1 = vector_store.similarity_search(
        "fuel efficient car", k=1, filter=filtered_metadata1
    )

    assert len(output1) == 1
    assert {
        key: value for key, value in output1[0].metadata.items() if key != "text"
    } == metadatas[1]

    # Perform similarity search with numeric and geo filtering
    filtered_metadata2 = (Num("msrp") < 30000) & (
        Geo("location") == GeoRadius(-122.4194, 37.7749, 100, "km")
    )
    output2 = vector_store.similarity_search(
        "affordable car", k=1, filter=filtered_metadata2
    )

    assert len(output2) == 1
    assert {
        key: value for key, value in output2[0].metadata.items() if key != "text"
    } == metadatas[0]

    # Perform similarity search with text filtering
    filtered_metadata3 = Text("review") % "*powerful engine*"
    output3 = vector_store.similarity_search(
        "sports car", k=1, filter=filtered_metadata3
    )

    assert len(output3) == 1
    assert {
        key: value for key, value in output3[0].metadata.items() if key != "text"
    } == metadatas[2]

    # Clean up
    vector_store.index.delete(drop=True)


def test_max_marginal_relevance_search(redis_url: str) -> None:
    """Test max marginal relevance search."""

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create a unique index name for testing
    index_name = f"test_index_{str(ULID())}"

    texts = ["foo", "bar", "baz", "bay", "bax", "baw", "bav"]

    # Create the RedisVectorStore
    vector_store = RedisVectorStore.from_texts(
        texts,
        embeddings,
        index_name=index_name,
        key_prefix="tst14",
        redis_url=redis_url,
    )

    mmr_output = vector_store.max_marginal_relevance_search("foo", k=3, fetch_k=3)

    assert len(mmr_output) == 3
    assert mmr_output[0].page_content == "foo"

    mmr_output = vector_store.max_marginal_relevance_search("foo", k=2, fetch_k=3)

    assert len(mmr_output) == 2
    assert mmr_output[0].page_content == "foo"
    assert mmr_output[1].page_content in ["bar", "baz", "bay", "bax", "baw", "bav"]

    mmr_output = vector_store.max_marginal_relevance_search(
        "foo",
        k=2,
        fetch_k=3,
        lambda_mult=0.1,  # more diversity
    )

    assert len(mmr_output) == 2
    assert mmr_output[0].page_content == "foo"
    assert mmr_output[1].page_content in ["baz", "bay", "bax", "baw", "bav"]

    # if fetch_k < k, then the output will be less than k
    mmr_output = vector_store.max_marginal_relevance_search("foo", k=3, fetch_k=2)
    assert len(mmr_output) == 2

    # Clean up
    vector_store.index.delete(drop=True)


# -- test distance metrics -- #


def test_cosine(redis_url: str) -> None:
    """Test cosine distance."""
    # Create embeddings
    embeddings = CustomTestEmbeddings()
    # Create a unique index name for testing
    index_name = f"test_index_{str(ULID())}"
    # Create the RedisVectorStore
    vector_store = RedisVectorStore.from_texts(
        ["0", "0.5", "1"],
        embeddings,
        index_name=index_name,
        key_prefix="tst15",
        distance_metric="cosine",
        redis_url=redis_url,
    )

    output = vector_store.similarity_search_with_score("0.2", k=3)
    _, score1 = output[0]  # type: ignore[misc]
    _, score2 = output[1]  # type: ignore[misc]
    _, score3 = output[2]  # type: ignore[misc]

    # Calculate the expected cosine similarity scores
    expected_score1 = 1 - math.cos(0.2)
    expected_score2 = 1 - math.cos(0.5 - 0.2)
    expected_score3 = 1 - math.cos(1 - 0.2)

    assert score1 == pytest.approx(expected_score1, abs=0.001)
    assert score2 == pytest.approx(expected_score2, abs=0.001)
    assert score3 == pytest.approx(expected_score3, abs=0.001)

    assert score1 < score2 < score3

    # Clean up
    vector_store.index.delete(drop=True)


def test_l2(redis_url: str) -> None:
    """Test L2 distance."""
    # Create embeddings
    embeddings = CustomTestEmbeddings()
    # Create a unique index name for testing
    index_name = f"test_index_{str(ULID())}"
    # Create the RedisVectorStore
    vector_store = RedisVectorStore.from_texts(
        ["0", "0.5", "1"],
        embeddings,
        index_name=index_name,
        key_prefix="tst16",
        distance_metric="L2",
        redis_url=redis_url,
    )

    output = vector_store.similarity_search_with_score("0.2", k=3)

    _, score1 = output[0]  # type: ignore[misc]
    _, score2 = output[1]  # type: ignore[misc]
    _, score3 = output[2]  # type: ignore[misc]

    assert score1 < score2 < score3

    # Clean up
    vector_store.index.delete(drop=True)


def test_ip(redis_url: str) -> None:
    """Test inner product distance."""
    # Create embeddings
    embeddings = CustomTestEmbeddings()
    # Create a unique index name for testing
    index_name = f"test_index_{str(ULID())}"
    # Create the RedisVectorStore
    vector_store = RedisVectorStore.from_texts(
        ["0", "0.5", "1"],
        embeddings,
        index_name=index_name,
        key_prefix="tst17",
        distance_metric="IP",
        redis_url=redis_url,
    )

    output = vector_store.similarity_search_with_score("0.2", k=3)

    _, score1 = output[0]  # type: ignore[misc]
    _, score2 = output[1]  # type: ignore[misc]
    _, score3 = output[2]  # type: ignore[misc]

    assert score1 < score2 < score3

    # Clean up
    vector_store.index.delete(drop=True)


def test_similarity_search_limit_distance(redis_url: str) -> None:
    """Test similarity search limit score."""
    # Sample texts with discernible distances
    texts = [
        "The cat is on the mat.",
        "The dog is in the yard.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create a unique index name for testing
    index_name = f"test_index_{str(ULID())}"

    # Create the RedisVectorStore
    vector_store = RedisVectorStore.from_texts(
        texts,
        embeddings,
        index_name=index_name,
        key_prefix="tst18",
        redis_url=redis_url,
    )

    # Set a distance threshold that will return only 2 results
    output = vector_store.similarity_search(texts[0], k=3, distance_threshold=0.16)

    # Expect only 2 results due to distance threshold
    assert len(output) == 2

    # Clean up
    vector_store.index.delete(drop=True)


def test_similarity_search_with_score_with_limit_distance(redis_url: str) -> None:
    """Test similarity search with score with limit score."""
    # Sample texts with discernible distances
    texts = [
        "The cat is on the mat.",
        "The dog is in the yard.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create a unique index name for testing
    index_name = f"test_index_{str(ULID())}"

    # Create the RedisVectorStore
    vector_store = RedisVectorStore.from_texts(
        texts,
        embeddings,
        index_name=index_name,
        key_prefix="tst18",
        redis_url=redis_url,
    )

    # Perform similarity search with score and distance threshold
    output = vector_store.similarity_search_with_score(
        texts[0], k=3, distance_threshold=0.16, return_metadata=True
    )

    # Expect only 2 results due to distance threshold
    assert len(output) == 2

    # Print and verify the scores
    for doc, score in output:  # type: ignore[misc]
        assert score >= 0  # Ensure score is non-negative

    # Clean up
    vector_store.index.delete(drop=True)


def test_large_batch(redis_url: str) -> None:
    # Create a unique index name for testing
    index_name = f"test_index_{str(ULID())}"
    embeddings = FakeEmbeddings(size=255)
    texts = ["This is a test document"] * (10000)
    vector_store = RedisVectorStore.from_texts(
        texts,
        embeddings,
        index_name=index_name,
        key_prefix="tst19",
        redis_url=redis_url,
        ids=[str(ULID) for _ in range(len(texts))],
    )

    count_query = CountQuery(FilterExpression("*"))

    count = vector_store.index.query(count_query)

    assert 10000 == count

    # Clean up
    vector_store.index.delete(drop=True)


def test_similarity_search_by_vector_with_extra_fields(redis_url: str) -> None:
    index_name = f"test_index_{str(ULID())}"
    embeddings = OpenAIEmbeddings()

    # Create a schema with only some fields indexed
    index_schema = IndexSchema.from_dict(
        {
            "index": {"name": index_name, "storage_type": "hash"},
            "fields": [
                {"name": "indexed_metadata", "type": "text"},
                {"name": "text", "type": "text"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 1536,
                        "distance_metric": "cosine",
                        "algorithm": "FLAT",
                    },
                },
            ],
        }
    )

    config = RedisConfig(
        index_name=index_name, redis_url=redis_url, schema=index_schema
    )

    vector_store = RedisVectorStore(embeddings, config=config)

    # Add documents with extra non-indexed fields
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "The lazy dog is jumped over by the quick brown fox",
        "The fox is quick and brown, and jumps over dogs",
    ]
    metadatas = [
        {
            "indexed_metadata": "doc1",
            "extra_field1": "value1",
            "extra_field2": "value2",
        },
        {
            "indexed_metadata": "doc2",
            "extra_field1": "value3",
            "extra_field2": "value4",
        },
        {
            "indexed_metadata": "doc3",
            "extra_field1": "value5",
            "extra_field2": "value6",
        },
    ]
    vector_store.add_texts(texts, metadatas)

    # Perform similarity search without return_all
    query_embedding = embeddings.embed_query("quick fox")
    results_without_return_all = vector_store.similarity_search_by_vector(
        query_embedding, k=2, return_metadata=True
    )

    assert len(results_without_return_all) == 2
    for doc in results_without_return_all:
        assert doc.page_content in texts
        assert "indexed_metadata" in doc.metadata
        assert doc.metadata["indexed_metadata"].startswith("doc")
        assert "extra_field1" not in doc.metadata
        assert "extra_field2" not in doc.metadata

    # Perform similarity search with return_all=True
    results_with_return_all = vector_store.similarity_search_by_vector(
        query_embedding, k=2, return_all=True
    )

    assert len(results_with_return_all) == 2
    for doc in results_with_return_all:
        assert doc.page_content in texts
        assert "indexed_metadata" in doc.metadata
        assert doc.metadata["indexed_metadata"].startswith("doc")
        assert "extra_field1" in doc.metadata
        assert "extra_field2" in doc.metadata
        assert doc.metadata["extra_field1"].startswith("value")
        assert doc.metadata["extra_field2"].startswith("value")

    # Clean up
    vector_store.index.delete(drop=True)


def test_similarity_search_with_score_by_vector_with_extra_fields(
    redis_url: str,
) -> None:
    index_name = f"test_index_{str(ULID())}"
    embeddings = OpenAIEmbeddings()

    # Create a schema with only some fields indexed
    index_schema = IndexSchema.from_dict(
        {
            "index": {"name": index_name, "storage_type": "hash"},
            "fields": [
                {"name": "indexed_metadata", "type": "text"},
                {"name": "text", "type": "text"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 1536,
                        "distance_metric": "cosine",
                        "algorithm": "FLAT",
                    },
                },
            ],
        }
    )

    config = RedisConfig(
        index_name=index_name, redis_url=redis_url, schema=index_schema
    )

    vector_store = RedisVectorStore(embeddings, config=config)

    # Add documents with extra non-indexed fields
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "The lazy dog is jumped over by the quick brown fox",
        "The fox is quick and brown, and jumps over dogs",
    ]
    metadatas = [
        {
            "indexed_metadata": "doc1",
            "extra_field1": "value1",
            "extra_field2": "value2",
        },
        {
            "indexed_metadata": "doc2",
            "extra_field1": "value3",
            "extra_field2": "value4",
        },
        {
            "indexed_metadata": "doc3",
            "extra_field1": "value5",
            "extra_field2": "value6",
        },
    ]
    vector_store.add_texts(texts, metadatas)

    # Perform similarity search without return_all
    query_embedding = embeddings.embed_query("quick fox")
    results_without_return_all = vector_store.similarity_search_with_score_by_vector(
        query_embedding, k=2
    )

    assert len(results_without_return_all) == 2
    for doc, score in results_without_return_all:  # type: ignore[misc]
        assert doc.page_content in texts
        assert "indexed_metadata" in doc.metadata
        assert doc.metadata["indexed_metadata"].startswith("doc")
        assert "extra_field1" not in doc.metadata
        assert "extra_field2" not in doc.metadata
        assert isinstance(score, float)
        assert 0 <= score <= 2  # Cosine distance is between 0 and 2

    # Perform similarity search with return_all=True
    results_with_return_all = vector_store.similarity_search_with_score_by_vector(
        query_embedding, k=2, return_all=True
    )

    assert len(results_with_return_all) == 2
    for doc, score in results_with_return_all:  # type: ignore[misc]
        assert doc.page_content in texts
        assert "indexed_metadata" in doc.metadata
        assert doc.metadata["indexed_metadata"].startswith("doc")
        assert "extra_field1" in doc.metadata
        assert "extra_field2" in doc.metadata
        assert doc.metadata["extra_field1"].startswith("value")
        assert doc.metadata["extra_field2"].startswith("value")
        assert isinstance(score, float)
        assert 0 <= score <= 2  # Cosine distance is between 0 and 2

    # Test with_vectors=True
    results_with_vectors = vector_store.similarity_search_with_score_by_vector(
        query_embedding, k=2, with_vectors=True, return_all=True
    )

    assert len(results_with_vectors) == 2
    for result in results_with_vectors:
        doc, score, vector = result  # type: ignore[misc]
        assert doc.page_content in texts
        assert "indexed_metadata" in doc.metadata
        assert "extra_field1" in doc.metadata
        assert "extra_field2" in doc.metadata
        assert isinstance(score, float)
        assert 0 <= score <= 2  # Cosine distance is between 0 and 2
        assert isinstance(vector, list)
        assert len(vector) == 1536  # Assuming OpenAI embeddings
        assert all(isinstance(v, float) for v in vector)

    # Clean up
    vector_store.index.delete(drop=True)


def test_connect_to_redisvl_created_index_w_index_name(redis_url: str) -> None:
    # Create a unique index name for testing
    index_name = f"test_index_{str(ULID())}"

    # Create a Redis client
    redis_client = Redis.from_url(redis_url)

    # Define the index schema
    schema = IndexSchema.from_dict(
        {
            "index": {"name": index_name, "storage_type": "hash", "prefix": index_name},
            "fields": [
                {"name": "text", "type": "text"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 3072,
                        "distance_metric": "cosine",
                        "algorithm": "FLAT",
                    },
                },
            ],
        }
    )

    # Create the index using RedisVL
    redisvl_index = SearchIndex(schema, redis_client)
    redisvl_index.create(overwrite=True)

    # Define sentences with known similarities
    sentences = {
        "like": "I like dogs.",
        "love": "You love dogs.",
        "dont_like": "I hate dogs.",
    }

    # Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectors = embeddings.embed_documents(list(sentences.values()))

    # Prepare data for Redis
    data = []
    for (label, text), vector in zip(sentences.items(), vectors):
        data.append(
            {"text": text, "embedding": np.array(vector, dtype=np.float32).tobytes()}
        )

    # Load data into Redis
    redisvl_index.load(
        data, keys=[f"{index_name}:{label}" for label in sentences.keys()]
    )

    # Verify data was added
    for label, text in sentences.items():
        key = f"{index_name}:{label}"
        assert redis_client.exists(key), f"Document '{label}' was not added to Redis"

    # Create LangChain's RedisVectorStore
    config = RedisConfig(index_name=index_name, redis_url=redis_url)
    langchain_vector_store = RedisVectorStore(embeddings, config=config)

    # Perform a similarity search
    query_text = "I like dogs."
    query_embedding = embeddings.embed_query(query_text)

    # RedisVL search
    vector_query = VectorQuery(
        vector=query_embedding,
        vector_field_name="embedding",
        return_fields=["text"],
        num_results=3,
    )
    redisvl_results = redisvl_index.query(vector_query)
    assert len(redisvl_results) == 3, f"Expected 3 results, got {len(redisvl_results)}"

    # LangChain search
    langchain_results = langchain_vector_store.similarity_search(query_text, k=3)

    # Assertions
    assert (
        len(langchain_results) == 3
    ), f"Expected 3 results, got {len(langchain_results)}"
    assert (
        langchain_results[0].page_content == "I like dogs."
    ), "Expected 'I like dogs.' to be the top result"
    assert (
        langchain_results[1].page_content == "You love dogs."
    ), "Expected 'You love dogs.' to be the second result"
    assert (
        langchain_results[2].page_content == "I hate dogs."
    ), "Expected 'I hate dogs.' to be the third result"

    # Clean up
    redisvl_index.delete(drop=True)


def test_similarity_search_k(redis_url: str) -> None:
    """Test end-to-end construction and search with varying k values."""
    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create a unique index name for testing
    index_name = f"test_index_{str(ULID())}"

    texts = ["foo", "bar", "baz"]

    # Create the RedisVectorStore
    vector_store = RedisVectorStore.from_texts(
        texts,
        embeddings,
        index_name=index_name,
        key_prefix="tst11",
        redis_url=redis_url,
    )

    # Perform similarity search with k=1
    output_k1 = vector_store.similarity_search("foo", k=1)
    assert len(output_k1) == 1
    assert output_k1[0].page_content == "foo"

    # Perform similarity search with k=2
    output_k2 = vector_store.similarity_search("foo", k=2)
    assert len(output_k2) == 2
    assert set(doc.page_content for doc in output_k2) == {"baz", "foo"}

    # Perform similarity search with k=3
    output_k3 = vector_store.similarity_search("foo", k=3)
    assert len(output_k3) == 3
    assert set(doc.page_content for doc in output_k3) == {"foo", "bar", "baz"}

    # Clean up
    vector_store.index.delete(drop=True)
