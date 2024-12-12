"""
Test CrateDB `FLOAT_VECTOR` / `KNN_MATCH` functionality.

cd tests/integration_tests/vectorstores/docker-compose
docker-compose -f cratedb.yml up
"""

import os
import re
from typing import Dict, Generator, List

import pytest
import sqlalchemy as sa
import sqlalchemy.orm
from langchain.docstore.document import Document
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session

from langchain_community.vectorstores.cratedb import CrateDBVectorStore
from langchain_community.vectorstores.cratedb.extended import (
    CrateDBVectorStoreMultiCollection,
)
from langchain_community.vectorstores.cratedb.model import ModelFactory
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
    FakeEmbeddings,
)

SCHEMA_NAME = os.environ.get("TEST_CRATEDB_DATABASE", "testdrive")

CONNECTION_STRING = CrateDBVectorStore.connection_string_from_db_params(
    driver=os.environ.get("TEST_CRATEDB_DRIVER", "crate"),
    host=os.environ.get("TEST_CRATEDB_HOST", "localhost"),
    port=int(os.environ.get("TEST_CRATEDB_PORT", "4200")),
    database=SCHEMA_NAME,
    user=os.environ.get("TEST_CRATEDB_USER", "crate"),
    password=os.environ.get("TEST_CRATEDB_PASSWORD", ""),
)

ADA_TOKEN_COUNT = 1536


@pytest.fixture
def engine() -> sa.Engine:
    """
    Return an SQLAlchemy engine object.
    """
    return sa.create_engine(CONNECTION_STRING, echo=False)


@pytest.fixture
def session(engine: sa.Engine) -> Generator[sa.orm.Session, None, None]:
    with engine.connect() as conn:
        with Session(conn) as session:
            yield session


@pytest.fixture(autouse=True)
def drop_tables(engine: sa.Engine) -> None:
    """
    Drop database tables.
    """
    try:
        mf = ModelFactory()
        mf.BaseModel.metadata.drop_all(engine, checkfirst=False)
    except Exception as ex:
        if "RelationUnknown" not in str(ex):
            raise


@pytest.fixture
def prune_tables(engine: sa.Engine) -> None:
    """
    Delete data from database tables.
    """
    with engine.connect() as conn:
        with Session(conn) as session:
            mf = ModelFactory()
            try:
                session.query(mf.CollectionStore).delete()
            except ProgrammingError:
                pass
            try:
                session.query(mf.EmbeddingStore).delete()
            except ProgrammingError:
                pass


def ensure_collection(session: sa.orm.Session, name: str) -> None:
    """
    Create a (fake) collection item.
    """
    session.execute(
        sa.text(
            """
            CREATE TABLE IF NOT EXISTS collection (
                uuid TEXT,
                name TEXT,
                cmetadata OBJECT
            );
            """
        )
    )
    session.execute(
        sa.text(
            """
            CREATE TABLE IF NOT EXISTS embedding (
                uuid TEXT,
                collection_id TEXT,
                embedding FLOAT_VECTOR(123),
                document TEXT,
                cmetadata OBJECT,
                custom_id TEXT
            );
            """
        )
    )
    try:
        session.execute(
            sa.text(
                f"INSERT INTO collection (uuid, name, cmetadata) "
                f"VALUES ('uuid-{name}', '{name}', {{}});"
            )
        )
        session.execute(sa.text("REFRESH TABLE collection"))
    except sa.exc.IntegrityError:
        pass


class FakeEmbeddingsWithAdaDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (ADA_TOKEN_COUNT - 1) + [float(i)] for i in range(len(texts))
        ]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (ADA_TOKEN_COUNT - 1) + [float(0.0)]


class ConsistentFakeEmbeddingsWithAdaDimension(ConsistentFakeEmbeddings):
    """
    Fake embeddings which remember all the texts seen so far to return
    consistent vectors for the same texts.

    Other than this, they also have a fixed dimensionality, which is
    important in this case.
    """

    def __init__(self, *args: List, **kwargs: Dict) -> None:
        super().__init__(dimensionality=ADA_TOKEN_COUNT)


def test_cratedb_texts() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_cratedb_embedding_dimension() -> None:
    """Verify the `embedding` column uses the correct vector dimensionality."""
    texts = ["foo", "bar", "baz"]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=ConsistentFakeEmbeddingsWithAdaDimension(),
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    with docsearch.Session() as session:
        result = session.execute(sa.text(f"SHOW CREATE TABLE {SCHEMA_NAME}.embedding"))
        record = result.first()
        if not record:
            raise ValueError("No data found")
        ddl = record[0]
        assert f'"embedding" FLOAT_VECTOR({ADA_TOKEN_COUNT})' in ddl


def test_cratedb_embeddings() -> None:
    """Test end to end construction with embeddings and search."""
    texts = ["foo", "bar", "baz"]
    text_embeddings = FakeEmbeddingsWithAdaDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = CrateDBVectorStore.from_embeddings(
        text_embeddings=text_embedding_pairs,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_cratedb_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


def test_cratedb_with_metadatas_with_scores() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score("foo", k=1)
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 1.0)]


def test_cratedb_with_filter_match() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    # TODO: Original:
    #       assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]  # noqa: E501
    output = docsearch.similarity_search_with_score("foo", k=1, filter={"page": "0"})
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 1.0)]


def test_cratedb_with_filter_distant_match() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score("foo", k=2, filter={"page": "2"})
    # Original score value: 0.0013003906671379406
    assert output == [(Document(page_content="baz", metadata={"page": "2"}), 0.2)]


def test_cratedb_with_filter_no_match() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score("foo", k=1, filter={"page": "5"})
    assert output == []


def test_cratedb_collection_delete() -> None:
    """
    Test end to end collection construction and deletion.
    Uses two different collections of embeddings.
    """

    store_foo = CrateDBVectorStore.from_texts(
        texts=["foo"],
        collection_name="test_collection_foo",
        collection_metadata={"category": "foo"},
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=[{"document": "foo"}],
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    store_bar = CrateDBVectorStore.from_texts(
        texts=["bar"],
        collection_name="test_collection_bar",
        collection_metadata={"category": "bar"},
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=[{"document": "bar"}],
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    session = store_foo.Session()

    # Verify data in database.
    collection_foo = store_foo.get_collection(session)
    collection_bar = store_bar.get_collection(session)
    if collection_foo is None or collection_bar is None:
        assert False, "Expected CollectionStore objects but received None"
    assert collection_foo.embeddings[0].cmetadata == {"document": "foo"}
    assert collection_bar.embeddings[0].cmetadata == {"document": "bar"}

    # Delete first collection.
    store_foo.delete_collection()

    # Verify that the "foo" collection has been deleted.
    collection_foo = store_foo.get_collection(session)
    collection_bar = store_bar.get_collection(session)
    if collection_bar is None:
        assert False, "Expected CollectionStore object but received None"
    assert collection_foo is None
    assert collection_bar.embeddings[0].cmetadata == {"document": "bar"}

    # Verify that associated embeddings also have been deleted.
    embeddings_count = session.query(store_foo.EmbeddingStore).count()
    assert embeddings_count == 1


def test_cratedb_collection_with_metadata() -> None:
    """Test end to end collection construction"""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    cratedb_vector = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection",
        collection_metadata={"foo": "bar"},
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    collection = cratedb_vector.get_collection(cratedb_vector.Session())
    if collection is None:
        assert False, "Expected a CollectionStore object but received None"
    else:
        assert collection.name == "test_collection"
        assert collection.cmetadata == {"foo": "bar"}


def test_cratedb_collection_no_embedding_dimension() -> None:
    """
    Verify that addressing collections fails when not specifying dimensions.
    """
    cratedb_vector = CrateDBVectorStore(
        embedding_function=None,  # type: ignore[arg-type]
        connection_string=CONNECTION_STRING,
    )
    session = cratedb_vector.Session()
    with pytest.raises(RuntimeError) as ex:
        cratedb_vector.get_collection(session)
    assert ex.match(
        "Collection can't be accessed without specifying "
        "dimension size of embedding vectors"
    )


def test_cratedb_collection_read_only(session: Session) -> None:
    """
    Test using a collection, without adding any embeddings upfront.

    This happens when just invoking the "retrieval" case.

    In this scenario, embedding dimensionality needs to be figured out
    from the supplied `embedding_function`.
    """

    # Create a fake collection item.
    ensure_collection(session, "baz2")

    # This test case needs an embedding _with_ dimensionality.
    # Otherwise, the data access layer is unable to figure it
    # out at runtime.
    embedding = ConsistentFakeEmbeddingsWithAdaDimension()

    vectorstore = CrateDBVectorStore(
        collection_name="baz2",
        connection_string=CONNECTION_STRING,
        embedding_function=embedding,
    )
    output = vectorstore.similarity_search("foo", k=1)

    # No documents/embeddings have been loaded, the collection is empty.
    # This is why there are also no results.
    assert output == []


def test_cratedb_with_filter_in_set() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score(
        "foo", k=2, filter={"page": {"IN": ["0", "2"]}}
    )
    # Original score values: 0.0, 0.0013003906671379406
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0),
        (Document(page_content="baz", metadata={"page": "2"}), 0.2),
    ]


def test_cratedb_delete_docs() -> None:
    """Add and delete documents."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        ids=["1", "2", "3"],
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    docsearch.delete(["1", "2"])
    with docsearch._make_session() as session:
        records = list(session.query(docsearch.EmbeddingStore).all())
        # ignoring type error since mypy cannot determine whether
        # the list is sortable
        assert sorted(record.custom_id for record in records) == ["3"]  # type: ignore

    docsearch.delete(["2", "3"])  # Should not raise on missing ids
    with docsearch._make_session() as session:
        records = list(session.query(docsearch.EmbeddingStore).all())
        # ignoring type error since mypy cannot determine whether
        # the list is sortable
        assert sorted(record.custom_id for record in records) == []  # type: ignore


def test_cratedb_relevance_score() -> None:
    """Test to make sure the relevance score is scaled to 0-1."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )

    output = docsearch.similarity_search_with_relevance_scores("foo", k=3)
    # Original score values: 1.0, 0.9996744261675065, 0.9986996093328621
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0),
        (Document(page_content="bar", metadata={"page": "1"}), 0.5),
        (Document(page_content="baz", metadata={"page": "2"}), 0.2),
    ]


def test_cratedb_retriever_search_threshold() -> None:
    """Test using retriever for searching with threshold."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.35},  # Original value: 0.999
    )
    output = retriever.invoke("summer")
    assert output == [
        Document(page_content="foo", metadata={"page": "0"}),
        Document(page_content="bar", metadata={"page": "1"}),
    ]


def test_cratedb_retriever_search_threshold_custom_normalization_fn() -> None:
    """Test searching with threshold and custom normalization function"""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
        relevance_score_fn=lambda d: d * 0,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )
    output = retriever.invoke("foo")
    assert output == []


def test_cratedb_max_marginal_relevance_search() -> None:
    """Test max marginal relevance search."""
    texts = ["foo", "bar", "baz"]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.max_marginal_relevance_search("foo", k=1, fetch_k=3)
    assert output == [Document(page_content="foo")]


def test_cratedb_max_marginal_relevance_search_with_score() -> None:
    """Test max marginal relevance search with relevance scores."""
    texts = ["foo", "bar", "baz"]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.max_marginal_relevance_search_with_score("foo", k=1, fetch_k=3)
    assert output == [(Document(page_content="foo"), 1.0)]


def test_cratedb_multicollection_search_success() -> None:
    """
    `CrateDBVectorStoreMultiCollection` provides functionality for
    searching multiple collections.
    """

    store_1 = CrateDBVectorStore.from_texts(
        texts=["Räuber", "Hotzenplotz"],
        collection_name="test_collection_1",
        embedding=ConsistentFakeEmbeddingsWithAdaDimension(),
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    _ = CrateDBVectorStore.from_texts(
        texts=["John", "Doe"],
        collection_name="test_collection_2",
        embedding=ConsistentFakeEmbeddingsWithAdaDimension(),
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )

    # Probe the first store.
    output = store_1.similarity_search("Räuber", k=1)
    assert Document(page_content="Räuber") in output[:2]
    output = store_1.similarity_search("Hotzenplotz", k=1)
    assert Document(page_content="Hotzenplotz") in output[:2]
    output = store_1.similarity_search("John Doe", k=1)
    assert Document(page_content="Hotzenplotz") in output[:2]

    # Probe the multi-store.
    multisearch = CrateDBVectorStoreMultiCollection(
        collection_names=["test_collection_1", "test_collection_2"],
        embedding_function=ConsistentFakeEmbeddingsWithAdaDimension(),
        connection_string=CONNECTION_STRING,
    )
    output = multisearch.similarity_search("Räuber Hotzenplotz", k=2)
    assert Document(page_content="Räuber") in output[:2]
    output = multisearch.similarity_search("John Doe", k=2)
    assert Document(page_content="Doe") in output[:2]


def test_cratedb_multicollection_fail_indexing_not_permitted() -> None:
    """
    `CrateDBVectorStoreMultiCollection` does not provide functionality for
    indexing documents.
    """

    with pytest.raises(NotImplementedError) as ex:
        CrateDBVectorStoreMultiCollection.from_texts(
            texts=["foo"],
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            connection_string=CONNECTION_STRING,
        )
    assert ex.match("This adapter can not be used for indexing documents")


def test_cratedb_multicollection_search_table_does_not_exist() -> None:
    """
    `CrateDBVectorStoreMultiCollection` will fail when the `collection`
    table does not exist.
    """

    store = CrateDBVectorStoreMultiCollection(
        collection_names=["unknown"],
        embedding_function=ConsistentFakeEmbeddingsWithAdaDimension(),
        connection_string=CONNECTION_STRING,
    )
    with pytest.raises(ProgrammingError) as ex:
        store.similarity_search("foo")
    assert ex.match(re.escape("RelationUnknown[Relation 'collection' unknown]"))


def test_cratedb_multicollection_search_unknown_collection() -> None:
    """
    `CrateDBVectorStoreMultiCollection` will fail when not able to identify
    collections to search in.
    """

    CrateDBVectorStore.from_texts(
        texts=["Räuber", "Hotzenplotz"],
        collection_name="test_collection",
        embedding=ConsistentFakeEmbeddingsWithAdaDimension(),
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )

    store = CrateDBVectorStoreMultiCollection(
        collection_names=["unknown"],
        embedding_function=ConsistentFakeEmbeddingsWithAdaDimension(),
        connection_string=CONNECTION_STRING,
    )
    with pytest.raises(ValueError) as ex:
        store.similarity_search("foo")
    assert ex.match("No collections found")


def test_cratedb_multicollection_no_embedding_dimension() -> None:
    """
    Verify that addressing collections fails when not specifying dimensions.
    """
    store = CrateDBVectorStoreMultiCollection(
        embedding_function=None,  # type: ignore[arg-type]
        connection_string=CONNECTION_STRING,
    )
    session = store.Session()
    with pytest.raises(RuntimeError) as ex:
        store.get_collection(session)
    assert ex.match(
        "Collection can't be accessed without specifying "
        "dimension size of embedding vectors"
    )
