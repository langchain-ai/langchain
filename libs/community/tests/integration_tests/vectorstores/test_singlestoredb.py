"""Test SingleStoreDB functionality."""

import math
import os
import tempfile
from typing import List, cast

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_community.vectorstores.singlestoredb import SingleStoreDB
from langchain_community.vectorstores.utils import DistanceStrategy
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

TEST_SINGLESTOREDB_URL = "root:pass@localhost:3306/db"
TEST_SINGLE_RESULT = [Document(page_content="foo")]
TEST_SINGLE_WITH_METADATA_RESULT = [Document(page_content="foo", metadata={"a": "b"})]
TEST_RESULT = [Document(page_content="foo"), Document(page_content="foo")]
TEST_IMAGES_DIR = ""

try:
    import singlestoredb as s2

    singlestoredb_installed = True
except ImportError:
    singlestoredb_installed = False

try:
    from langchain_experimental.open_clip import OpenCLIPEmbeddings

    langchain_experimental_installed = True
except ImportError:
    langchain_experimental_installed = False


def drop(table_name: str) -> None:
    with s2.connect(TEST_SINGLESTOREDB_URL) as conn:
        conn.autocommit(True)
        with conn.cursor() as cursor:
            cursor.execute(f"DROP TABLE IF EXISTS {table_name};")


class NormilizedFakeEmbeddings(FakeEmbeddings):
    """Fake embeddings with normalization. For testing purposes."""

    def normalize(self, vector: List[float]) -> List[float]:
        """Normalize vector."""
        return [float(v / np.linalg.norm(vector)) for v in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.normalize(v) for v in super().embed_documents(texts)]

    def embed_query(self, text: str) -> List[float]:
        return self.normalize(super().embed_query(text))


class RandomEmbeddings(Embeddings):
    """Fake embeddings with random vectors. For testing purposes."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [cast(list[float], np.random.rand(100).tolist()) for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return cast(list[float], np.random.rand(100).tolist())

    def embed_image(self, uris: List[str]) -> List[List[float]]:
        return [cast(list[float], np.random.rand(100).tolist()) for _ in uris]


class IncrementalEmbeddings(Embeddings):
    """Fake embeddings with incremental vectors. For testing purposes."""

    def __init__(self) -> None:
        self.counter = 0

    def set_counter(self, counter: int) -> None:
        self.counter = counter

    def embed_query(self, text: str) -> List[float]:
        self.counter += 1
        return [
            math.cos(self.counter * math.pi / 10),
            math.sin(self.counter * math.pi / 10),
        ]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]


@pytest.fixture
def texts() -> List[str]:
    return ["foo", "bar", "baz"]


@pytest.fixture
def snow_rain_docs() -> List[Document]:
    return [
        Document(
            page_content="""In the parched desert, a sudden rainstorm brought relief,
            as the droplets danced upon the thirsty earth, rejuvenating the landscape
            with the sweet scent of petrichor.""",
            metadata={"count": "1", "category": "rain", "group": "a"},
        ),
        Document(
            page_content="""Amidst the bustling cityscape, the rain fell relentlessly,
            creating a symphony of pitter-patter on the pavement, while umbrellas
            bloomed like colorful flowers in a sea of gray.""",
            metadata={"count": "2", "category": "rain", "group": "a"},
        ),
        Document(
            page_content="""High in the mountains, the rain transformed into a delicate
            mist, enveloping the peaks in a mystical veil, where each droplet seemed to
            whisper secrets to the ancient rocks below.""",
            metadata={"count": "3", "category": "rain", "group": "b"},
        ),
        Document(
            page_content="""Blanketing the countryside in a soft, pristine layer, the
            snowfall painted a serene tableau, muffling the world in a tranquil hush
            as delicate flakes settled upon the branches of trees like nature's own 
            lacework.""",
            metadata={"count": "1", "category": "snow", "group": "b"},
        ),
        Document(
            page_content="""In the urban landscape, snow descended, transforming
            bustling streets into a winter wonderland, where the laughter of
            children echoed amidst the flurry of snowballs and the twinkle of
            holiday lights.""",
            metadata={"count": "2", "category": "snow", "group": "a"},
        ),
        Document(
            page_content="""Atop the rugged peaks, snow fell with an unyielding
            intensity, sculpting the landscape into a pristine alpine paradise,
            where the frozen crystals shimmered under the moonlight, casting a
            spell of enchantment over the wilderness below.""",
            metadata={"count": "3", "category": "snow", "group": "a"},
        ),
    ]


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb(texts: List[str]) -> None:
    """Test end to end construction and search."""
    table_name = "test_singlestoredb"
    drop(table_name)
    docsearch = SingleStoreDB.from_texts(
        texts,
        NormilizedFakeEmbeddings(),
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == TEST_SINGLE_RESULT
    docsearch.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_new_vector(texts: List[str]) -> None:
    """Test adding a new document"""
    table_name = "test_singlestoredb_new_vector"
    drop(table_name)
    docsearch = SingleStoreDB.from_texts(
        texts,
        NormilizedFakeEmbeddings(),
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    docsearch.add_texts(["foo"])
    output = docsearch.similarity_search("foo", k=2)
    assert output == TEST_RESULT
    docsearch.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_euclidean_distance(texts: List[str]) -> None:
    """Test adding a new document"""
    table_name = "test_singlestoredb_euclidean_distance"
    drop(table_name)
    docsearch = SingleStoreDB.from_texts(
        texts,
        FakeEmbeddings(),
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    docsearch.add_texts(["foo"])
    output = docsearch.similarity_search("foo", k=2)
    assert output == TEST_RESULT
    docsearch.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_vector_index_1(texts: List[str]) -> None:
    """Test adding a new document"""
    table_name = "test_singlestoredb_vector_index_1"
    drop(table_name)
    docsearch = SingleStoreDB.from_texts(
        texts,
        FakeEmbeddings(),
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        table_name=table_name,
        use_vector_index=True,
        vector_size=10,
        host=TEST_SINGLESTOREDB_URL,
    )
    docsearch.add_texts(["foo"])
    output = docsearch.similarity_search("foo", k=2)
    assert output == TEST_RESULT
    docsearch.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_vector_index_2(texts: List[str]) -> None:
    """Test adding a new document"""
    table_name = "test_singlestoredb_vector_index_2"
    drop(table_name)
    docsearch = SingleStoreDB.from_texts(
        texts,
        FakeEmbeddings(),
        table_name=table_name,
        use_vector_index=True,
        vector_index_options={"index_type": "IVF_PQ", "nlist": 256},
        vector_size=10,
        host=TEST_SINGLESTOREDB_URL,
    )
    docsearch.add_texts(["foo"])
    output = docsearch.similarity_search("foo", k=1)
    output[0].page_content == "foo"
    docsearch.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_vector_index_large() -> None:
    """Test adding a new document"""
    table_name = "test_singlestoredb_vector_index_large"
    drop(table_name)
    docsearch = SingleStoreDB.from_texts(
        ["foo"] * 30,
        RandomEmbeddings(),
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        table_name=table_name,
        use_vector_index=True,
        vector_size=100,
        vector_index_name="vector_index_large",
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output[0].page_content == "foo"
    docsearch.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_from_existing(texts: List[str]) -> None:
    """Test adding a new document"""
    table_name = "test_singlestoredb_from_existing"
    drop(table_name)
    SingleStoreDB.from_texts(
        texts,
        NormilizedFakeEmbeddings(),
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    # Test creating from an existing
    docsearch2 = SingleStoreDB(
        NormilizedFakeEmbeddings(),
        table_name="test_singlestoredb_from_existing",
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch2.similarity_search("foo", k=1)
    assert output == TEST_SINGLE_RESULT
    docsearch2.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_from_documents(texts: List[str]) -> None:
    """Test from_documents constructor."""
    table_name = "test_singlestoredb_from_documents"
    drop(table_name)
    docs = [Document(page_content=t, metadata={"a": "b"}) for t in texts]
    docsearch = SingleStoreDB.from_documents(
        docs,
        NormilizedFakeEmbeddings(),
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == TEST_SINGLE_WITH_METADATA_RESULT
    docsearch.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_add_texts_to_existing(texts: List[str]) -> None:
    """Test adding a new document"""
    table_name = "test_singlestoredb_add_texts_to_existing"
    drop(table_name)
    # Test creating from an existing
    SingleStoreDB.from_texts(
        texts,
        NormilizedFakeEmbeddings(),
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    docsearch = SingleStoreDB(
        NormilizedFakeEmbeddings(),
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    docsearch.add_texts(["foo"])
    output = docsearch.similarity_search("foo", k=2)
    assert output == TEST_RESULT
    docsearch.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_filter_metadata(texts: List[str]) -> None:
    """Test filtering by metadata"""
    table_name = "test_singlestoredb_filter_metadata"
    drop(table_name)
    docs = [
        Document(page_content=t, metadata={"index": i}) for i, t in enumerate(texts)
    ]
    docsearch = SingleStoreDB.from_documents(
        docs,
        FakeEmbeddings(),
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch.similarity_search("foo", k=1, filter={"index": 2})
    assert output == [Document(page_content="baz", metadata={"index": 2})]
    docsearch.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_filter_metadata_2(texts: List[str]) -> None:
    """Test filtering by metadata field that is similar for each document"""
    table_name = "test_singlestoredb_filter_metadata_2"
    drop(table_name)
    docs = [
        Document(page_content=t, metadata={"index": i, "category": "budget"})
        for i, t in enumerate(texts)
    ]
    docsearch = SingleStoreDB.from_documents(
        docs,
        FakeEmbeddings(),
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch.similarity_search("foo", k=1, filter={"category": "budget"})
    assert output == [
        Document(page_content="foo", metadata={"index": 0, "category": "budget"})
    ]
    docsearch.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_filter_metadata_3(texts: List[str]) -> None:
    """Test filtering by two metadata fields"""
    table_name = "test_singlestoredb_filter_metadata_3"
    drop(table_name)
    docs = [
        Document(page_content=t, metadata={"index": i, "category": "budget"})
        for i, t in enumerate(texts)
    ]
    docsearch = SingleStoreDB.from_documents(
        docs,
        FakeEmbeddings(),
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch.similarity_search(
        "foo", k=1, filter={"category": "budget", "index": 1}
    )
    assert output == [
        Document(page_content="bar", metadata={"index": 1, "category": "budget"})
    ]
    docsearch.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_filter_metadata_4(texts: List[str]) -> None:
    """Test no matches"""
    table_name = "test_singlestoredb_filter_metadata_4"
    drop(table_name)
    docs = [
        Document(page_content=t, metadata={"index": i, "category": "budget"})
        for i, t in enumerate(texts)
    ]
    docsearch = SingleStoreDB.from_documents(
        docs,
        FakeEmbeddings(),
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch.similarity_search("foo", k=1, filter={"category": "vacation"})
    assert output == []
    docsearch.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_filter_metadata_5(texts: List[str]) -> None:
    """Test complex metadata path"""
    table_name = "test_singlestoredb_filter_metadata_5"
    drop(table_name)
    docs = [
        Document(
            page_content=t,
            metadata={
                "index": i,
                "category": "budget",
                "subfield": {"subfield": {"idx": i, "other_idx": i + 1}},
            },
        )
        for i, t in enumerate(texts)
    ]
    docsearch = SingleStoreDB.from_documents(
        docs,
        FakeEmbeddings(),
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch.similarity_search(
        "foo", k=1, filter={"category": "budget", "subfield": {"subfield": {"idx": 2}}}
    )
    assert output == [
        Document(
            page_content="baz",
            metadata={
                "index": 2,
                "category": "budget",
                "subfield": {"subfield": {"idx": 2, "other_idx": 3}},
            },
        )
    ]
    docsearch.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_filter_metadata_6(texts: List[str]) -> None:
    """Test filtering by other bool"""
    table_name = "test_singlestoredb_filter_metadata_6"
    drop(table_name)
    docs = [
        Document(
            page_content=t,
            metadata={"index": i, "category": "budget", "is_good": i == 1},
        )
        for i, t in enumerate(texts)
    ]
    docsearch = SingleStoreDB.from_documents(
        docs,
        FakeEmbeddings(),
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch.similarity_search(
        "foo", k=1, filter={"category": "budget", "is_good": True}
    )
    assert output == [
        Document(
            page_content="bar",
            metadata={"index": 1, "category": "budget", "is_good": True},
        )
    ]
    docsearch.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_filter_metadata_7(texts: List[str]) -> None:
    """Test filtering by float"""
    table_name = "test_singlestoredb_filter_metadata_7"
    drop(table_name)
    docs = [
        Document(
            page_content=t,
            metadata={"index": i, "category": "budget", "score": i + 0.5},
        )
        for i, t in enumerate(texts)
    ]
    docsearch = SingleStoreDB.from_documents(
        docs,
        FakeEmbeddings(),
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch.similarity_search(
        "bar", k=1, filter={"category": "budget", "score": 2.5}
    )
    assert output == [
        Document(
            page_content="baz",
            metadata={"index": 2, "category": "budget", "score": 2.5},
        )
    ]
    docsearch.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_as_retriever(texts: List[str]) -> None:
    table_name = "test_singlestoredb_8"
    drop(table_name)
    docsearch = SingleStoreDB.from_texts(
        texts,
        FakeEmbeddings(),
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    retriever = docsearch.as_retriever(search_kwargs={"k": 2})
    output = retriever.invoke("foo")
    assert output == [
        Document(
            page_content="foo",
        ),
        Document(
            page_content="bar",
        ),
    ]
    docsearch.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_add_image(texts: List[str]) -> None:
    """Test adding images"""
    table_name = "test_singlestoredb_add_image"
    drop(table_name)
    docsearch = SingleStoreDB(
        RandomEmbeddings(),
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    temp_files = []
    for _ in range(3):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(b"foo")
        temp_file.close()
        temp_files.append(temp_file.name)

    docsearch.add_images(temp_files)
    output = docsearch.similarity_search("foo", k=1)
    assert output[0].page_content in temp_files
    docsearch.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
@pytest.mark.skipif(
    not langchain_experimental_installed, reason="langchain_experimental not installed"
)
def test_singestoredb_add_image2() -> None:
    table_name = "test_singlestoredb_add_images"
    drop(table_name)
    docsearch = SingleStoreDB(
        OpenCLIPEmbeddings(),  # type: ignore[call-arg, call-arg, call-arg]
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    image_uris = sorted(
        [
            os.path.join(TEST_IMAGES_DIR, image_name)
            for image_name in os.listdir(TEST_IMAGES_DIR)
            if image_name.endswith(".jpg")
        ]
    )
    docsearch.add_images(image_uris)
    output = docsearch.similarity_search("horse", k=1)
    assert "horse" in output[0].page_content
    docsearch.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_text_only_search(snow_rain_docs: List[Document]) -> None:
    table_name = "test_singlestoredb_text_only_search"
    drop(table_name)
    docsearch = SingleStoreDB(
        RandomEmbeddings(),
        table_name=table_name,
        use_full_text_search=True,
        host=TEST_SINGLESTOREDB_URL,
    )
    docsearch.add_documents(snow_rain_docs)
    output = docsearch.similarity_search(
        "rainstorm in parched desert",
        k=3,
        filter={"count": "1"},
        search_strategy=SingleStoreDB.SearchStrategy.TEXT_ONLY,
    )
    assert len(output) == 2
    assert (
        "In the parched desert, a sudden rainstorm brought relief,"
        in output[0].page_content
    )
    assert (
        "Blanketing the countryside in a soft, pristine layer" in output[1].page_content
    )

    output = docsearch.similarity_search(
        "snowfall in countryside",
        k=3,
        search_strategy=SingleStoreDB.SearchStrategy.TEXT_ONLY,
    )
    assert len(output) == 3
    assert (
        "Blanketing the countryside in a soft, pristine layer,"
        in output[0].page_content
    )
    docsearch.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_filter_by_text_search(snow_rain_docs: List[Document]) -> None:
    table_name = "test_singlestoredb_filter_by_text_search"
    drop(table_name)
    embeddings = IncrementalEmbeddings()
    docsearch = SingleStoreDB.from_documents(
        snow_rain_docs,
        embeddings,
        table_name=table_name,
        use_full_text_search=True,
        use_vector_index=True,
        vector_size=2,
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch.similarity_search(
        "rainstorm in parched desert",
        k=1,
        search_strategy=SingleStoreDB.SearchStrategy.FILTER_BY_TEXT,
        filter_threshold=0,
    )
    assert len(output) == 1
    assert (
        "In the parched desert, a sudden rainstorm brought relief"
        in output[0].page_content
    )
    drop(table_name)


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_filter_by_vector_search1(snow_rain_docs: List[Document]) -> None:
    table_name = "test_singlestoredb_filter_by_vector_search1"
    drop(table_name)
    embeddings = IncrementalEmbeddings()
    docsearch = SingleStoreDB.from_documents(
        snow_rain_docs,
        embeddings,
        table_name=table_name,
        use_full_text_search=True,
        use_vector_index=True,
        vector_size=2,
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch.similarity_search(
        "rainstorm in parched desert, rain",
        k=1,
        filter={"category": "rain"},
        search_strategy=SingleStoreDB.SearchStrategy.FILTER_BY_VECTOR,
        filter_threshold=-0.2,
    )
    assert len(output) == 1
    assert (
        "High in the mountains, the rain transformed into a delicate"
        in output[0].page_content
    )
    docsearch.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_filter_by_vector_search2(snow_rain_docs: List[Document]) -> None:
    table_name = "test_singlestoredb_filter_by_vector_search2"
    drop(table_name)
    embeddings = IncrementalEmbeddings()
    docsearch = SingleStoreDB.from_documents(
        snow_rain_docs,
        embeddings,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        table_name=table_name,
        use_full_text_search=True,
        use_vector_index=True,
        vector_size=2,
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch.similarity_search(
        "rainstorm in parched desert, rain",
        k=1,
        filter={"group": "a"},
        search_strategy=SingleStoreDB.SearchStrategy.FILTER_BY_VECTOR,
        filter_threshold=1.5,
    )
    assert len(output) == 1
    assert (
        "Amidst the bustling cityscape, the rain fell relentlessly"
        in output[0].page_content
    )
    docsearch.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_weighted_sum_search_unsupported_strategy(
    snow_rain_docs: List[Document],
) -> None:
    table_name = "test_singlestoredb_waighted_sum_search_unsupported_strategy"
    drop(table_name)
    embeddings = IncrementalEmbeddings()
    docsearch = SingleStoreDB.from_documents(
        snow_rain_docs,
        embeddings,
        table_name=table_name,
        use_full_text_search=True,
        use_vector_index=True,
        vector_size=2,
        host=TEST_SINGLESTOREDB_URL,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )
    try:
        docsearch.similarity_search(
            "rainstorm in parched desert, rain",
            k=1,
            search_strategy=SingleStoreDB.SearchStrategy.WEIGHTED_SUM,
        )
    except ValueError as e:
        assert "Search strategy WEIGHTED_SUM is not" in str(e)
    docsearch.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_weighted_sum_search(snow_rain_docs: List[Document]) -> None:
    table_name = "test_singlestoredb_waighted_sum_search"
    drop(table_name)
    embeddings = IncrementalEmbeddings()
    docsearch = SingleStoreDB.from_documents(
        snow_rain_docs,
        embeddings,
        table_name=table_name,
        use_full_text_search=True,
        use_vector_index=True,
        vector_size=2,
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch.similarity_search(
        "rainstorm in parched desert, rain",
        k=1,
        search_strategy=SingleStoreDB.SearchStrategy.WEIGHTED_SUM,
        filter={"category": "snow"},
    )
    assert len(output) == 1
    assert (
        "Atop the rugged peaks, snow fell with an unyielding" in output[0].page_content
    )
    docsearch.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_insert(snow_rain_docs: List[Document]) -> None:
    table_name = "test_singlestoredb_insert"
    drop(table_name)
    embeddings = IncrementalEmbeddings()
    docsearch = SingleStoreDB(
        embeddings,
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    ids = docsearch.add_documents(snow_rain_docs, return_ids=True)
    assert len(ids) == len(snow_rain_docs)
    for i, id1 in enumerate(ids):
        for j, id2 in enumerate(ids):
            if i != j:
                assert id1 != id2
    docsearch.drop()


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_delete(snow_rain_docs: List[Document]) -> None:
    table_name = "test_singlestoredb_delete"
    drop(table_name)
    embeddings = IncrementalEmbeddings()
    docsearch = SingleStoreDB(
        embeddings,
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    ids = docsearch.add_documents(snow_rain_docs, return_ids=True)
    output = docsearch.similarity_search(
        "rainstorm in parched desert",
        k=3,
        filter={"count": "1"},
    )
    assert len(output) == 2
    docsearch.delete(ids)
    output = docsearch.similarity_search(
        "rainstorm in parched desert",
        k=3,
    )
    assert len(output) == 0
    docsearch.drop()
