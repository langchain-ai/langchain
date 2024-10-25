import os
from typing import Generator

import pytest
from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore, InMemoryStore
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from objectbox.c import c_str, obx_remove_db_files
from objectbox.model.properties import HnswDistanceType

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.objectbox import ObjectBox
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
    FakeEmbeddings,
)


def remove_test_dir(test_dir: str) -> None:
    obx_remove_db_files(c_str(test_dir))


@pytest.fixture(autouse=True)
def auto_cleanup() -> Generator[None, None, None]:
    remove_test_dir("data")
    try:
        yield  # run the test function
    finally:
        remove_test_dir("data")


def test_objectbox_db_initialisation() -> None:
    ObjectBox(embedding=FakeEmbeddings(), embedding_dimensions=10)
    folder_path = "data"

    assert os.path.exists(folder_path), f"Folder '{folder_path}' does not exist."

    filepath = os.path.join(folder_path, "data.mdb")
    assert os.path.isfile(
        filepath
    ), f"File '{folder_path}' not found in '{folder_path}'"


def test_similarity_search() -> None:
    ob = ObjectBox(
        embedding=ConsistentFakeEmbeddings(dimensionality=10),
        embedding_dimensions=10,
        distance_type=HnswDistanceType.EUCLIDEAN,
    )

    objects = [
        {
            "title": "Inception",
            "year": 2010,
            "director": "Christopher Nolan",
            "genre": ["Science Fiction", "Action", "Thriller"],
        },
        {
            "title": "Spirited Away",
            "year": 2001,
            "director": "Hayao Miyazaki",
            "genre": ["Animation", "Fantasy", "Adventure"],
        },
        {
            "title": "The Shawshank Redemption",
            "year": 1994,
            "director": "Frank Darabont",
            "genre": ["Drama", "Crime"],
        },
        {
            "title": "Pan's Labyrinth",
            "year": 2006,
            "director": "Guillermo del Toro",
            "genre": ["Fantasy", "Drama", "War"],
        },
        {
            "title": "Pulp Fiction",
            "year": 1994,
            "director": "Quentin Tarantino",
            "genre": ["Crime", "Drama"],
        },
    ]
    texts = [str(obj["title"]) for obj in objects]  # text is the title
    metadatas = [
        obj for obj in objects
    ]  # Use all object attributes as its metadata (including the text)

    ob.add_texts(texts=texts, metadatas=metadatas)

    results = ob.similarity_search("Inception", k=1)
    assert len(results) == 1
    assert results[0].page_content == "Inception"
    assert results[0].metadata["year"] == 2010
    assert results[0].metadata["director"] == "Christopher Nolan"
    assert results[0].metadata["genre"] == ["Science Fiction", "Action", "Thriller"]

    results = ob.similarity_search("Spirited Away", k=2)
    assert len(results) == 2
    assert results[0].page_content == "Spirited Away"
    assert results[0].metadata["year"] == 2001
    assert results[1].page_content == "The Shawshank Redemption"
    assert results[1].metadata["year"] == 1994

    results = ob.similarity_search("The Shawshank Redemption", k=3)
    assert len(results) == 3
    assert results[0].page_content == "The Shawshank Redemption"
    assert results[0].metadata["year"] == 1994
    assert results[1].page_content == "Pan's Labyrinth"
    assert results[1].metadata["year"] == 2006
    assert results[2].page_content == "Spirited Away"
    assert results[2].metadata["year"] == 2001


def test_similarity_search_distance_types() -> None:
    """
    Test similarity search with different distance
    types (euclidean, cosine, dot product, ...).
    """

    texts = ["Apple", "Banana", "Carrot", "Mango", "Onion"]

    # Check embeddings values
    embedder = ConsistentFakeEmbeddings(dimensionality=2)
    embeddings = embedder.embed_documents(texts)
    assert embeddings == [
        [1.0, 0.0],  # Apple
        [1.0, 1.0],  # Banana
        [1.0, 2.0],  # Carrot
        [1.0, 3.0],  # Mango
        [1.0, 4.0],  # Onion
    ]

    # Test EUCLIDEAN
    embedder.known_texts = []  # To produce same embeddings
    ob = ObjectBox(
        embedding=embedder,
        embedding_dimensions=2,
        distance_type=HnswDistanceType.EUCLIDEAN,
        clear_db=True,
    )
    ob.add_texts(texts)

    results = ob.similarity_search_by_vector([1.0, 2.7], k=3)
    assert results[0].page_content == "Mango"
    assert results[1].page_content == "Carrot"
    assert results[2].page_content == "Onion"

    del ob

    # Test COSINE
    embedder.known_texts = []  # To produce same embeddings
    ob = ObjectBox(
        embedding=embedder,
        embedding_dimensions=2,
        distance_type=HnswDistanceType.COSINE,
        clear_db=True,
    )
    ob.add_texts(texts)

    results = ob.similarity_search_by_vector([3.0, -2.7], k=5)
    assert results[0].page_content == "Apple"  # Cosine distance: 0.740
    assert results[1].page_content == "Banana"  # Cosine distance: 0.525
    assert results[2].page_content == "Carrot"  # Cosine distance: -0.266
    assert results[3].page_content == "Mango"  # Cosine distance: -0.400
    assert results[4].page_content == "Onion"  # Cosine distance: -0.469

    del ob

    # Test DOT_PRODUCT
    # TODO we need *Embeddings to produce normalized vectors

    # Test DOT_PRODUCT_NON_NORMALIZED
    # TODO Core error: Argument condition "value <= OBXHnswDistanceType_Hamming" not met
    # embedder.known_texts = []  # To produce same embeddings
    ob = ObjectBox(
        embedding=embedder,
        embedding_dimensions=2,
        distance_type=HnswDistanceType.COSINE,
        clear_db=True,
    )
    ob.add_texts(texts)

    results = ob.similarity_search_by_vector(
        [2.0, 2.0], k=3
    )  # Same direction as Banana

    assert results[0].page_content == "Banana"
    assert results[1].page_content == "Carrot"
    assert results[2].page_content == "Mango"

    # This uses ObjectBox distance
    results1 = ob.similarity_search_with_score("Banana", k=3)
    assert results1[0][0].page_content == "Banana"
    assert results1[1][0].page_content == "Carrot"
    assert results1[2][0].page_content == "Mango"
    assert results1[0][1] < results1[1][1]
    assert results1[1][1] < results1[2][1]
    assert results1[0][1] < 1.0
    assert results1[0][1] == 0.0  # perfect match
    assert results1[1][1] > 0.0
    assert results1[1][1] < 1.0
    assert results1[2][1] > 0.0
    assert results1[2][1] < 1.0
    # TODO create vectors that result in a distance > 1.0

    # With LangChain relevance scores, higher scores indicate a higher
    # similarity (0 is dissimilar, 1 is most similar)
    results2 = ob.similarity_search_with_relevance_scores("Banana", k=3)
    assert results2[0][0].page_content == "Banana"
    assert results2[1][0].page_content == "Carrot"
    assert results2[2][0].page_content == "Mango"
    assert results2[0][1] > results2[1][1]
    assert results2[1][1] > results2[2][1]
    assert results2[0][1] > 0.0
    assert results2[0][1] == 1.0  # perfect match
    assert results2[1][1] > 0.0
    assert results2[1][1] < 1.0
    assert results2[2][1] > 0.0
    assert results2[2][1] < 1.0

    del ob


def test_similarity_search_with_relevance_scores() -> None:
    """
    Tests that similarity_search_with_relevance_scores returns
    values in range [0, 1] regardless the distance type.
    """

    texts = ["Apple", "Banana", "Carrot", "Mango", "Onion"]

    # Test euclidean
    # TODO currently relevance score for euclidean distance isn't implemented
    # ob = ObjectBox(
    #     embedding=ConsistentFakeEmbeddings(dimensionality=2),
    #     embedding_dimensions=2,
    #     distance_type=HnswDistanceType.EUCLIDEAN,
    #     clear_db=True
    # )
    # ob.add_texts(texts)
    #
    # results = ob.similarity_search_with_relevance_scores("Strawberry", k=3)
    # assert 0.0 <= results[0][1] <= 1.0
    # assert 0.0 <= results[1][1] <= 1.0
    # assert 0.0 <= results[2][1] <= 1.0
    #
    # del ob

    # Test cosine
    ob = ObjectBox(
        embedding=ConsistentFakeEmbeddings(dimensionality=2),
        embedding_dimensions=2,
        distance_type=HnswDistanceType.COSINE,
        clear_db=True,
    )
    ob.add_texts(texts)

    results = ob.similarity_search_with_relevance_scores("Strawberry", k=3)
    assert 0.0 <= results[0][1] <= 1.0
    assert 0.0 <= results[1][1] <= 1.0
    assert 0.0 <= results[2][1] <= 1.0

    # TODO test DOT_PRODUCT
    # TODO test DOT_PRODUCT_UNNORMALIZED


def test_from_texts() -> None:
    texts = ["foo", "bar", "baz"]
    ob = ObjectBox.from_texts(
        embedding=FakeEmbeddings(), embedding_dimensions=10, texts=texts
    )

    # positive test
    query = ob.similarity_search("foo", k=2)
    assert len(query) == 2


def test_similarity_search_with_score() -> None:
    ob = ObjectBox(embedding=FakeEmbeddings(), embedding_dimensions=10)
    texts = ["foo", "bar", "baz"]
    ob.add_texts(texts=texts)

    query = ob.similarity_search_with_score("foo", k=1)
    assert len(query) == 1

    query = ob.similarity_search_with_score("foo", k=2)
    assert len(query) == 2

    query = ob.similarity_search_with_score("foo", k=3)
    assert len(query) == 3


def test_similarity_search_by_vector() -> None:
    ob = ObjectBox(embedding=FakeEmbeddings(), embedding_dimensions=10)
    texts = ["foo", "bar", "baz"]
    ob.add_texts(texts=texts)

    query_embedding = FakeEmbeddings().embed_query("foo")
    query = ob.similarity_search_by_vector(query_embedding, k=1)
    assert len(query) == 1

    query = ob.similarity_search_by_vector(query_embedding, k=2)
    assert len(query) == 2

    query = ob.similarity_search_by_vector(query_embedding, k=3)
    assert len(query) == 3


def test_delete_vector_by_ids() -> None:
    ob = ObjectBox(embedding=FakeEmbeddings(), embedding_dimensions=10)
    texts = ["foo", "bar", "baz"]
    ob.add_texts(texts=texts)

    bool = ob.delete(["1", "2"])
    assert bool is True

    assert ob._vector_box.count() == 1


def test_parent_document_retriever() -> None:
    loaders = [
        TextLoader("../../docs/docs/modules/paul_graham_essay.txt"),
        TextLoader("../../docs/docs/modules/state_of_the_union.txt"),
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    # This text splitter is used to create the child documents
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    # The vectorstore to use to index the child chunks
    vectorstore = ObjectBox(
        embedding=HuggingFaceEmbeddings(),
        embedding_dimensions=768,
    )
    # The storage layer for the parent documents
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        id_key="id",
    )

    retriever.add_documents(docs, ids=None)

    assert len(list(store.yield_keys())) == 2

    _ = vectorstore.similarity_search("justice breyer")

    retrieved_docs = retriever.invoke("justice breyer")
    assert len(retrieved_docs[0].page_content) == 38540


def test_multi_vector_retriever() -> None:
    loaders = [
        TextLoader("../../docs/docs/modules/paul_graham_essay.txt"),
        TextLoader("../../docs/docs/modules/state_of_the_union.txt"),
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
    docs = text_splitter.split_documents(docs)
    # The vectorstore to use to index the child chunks
    vectorstore = ObjectBox(embedding=HuggingFaceEmbeddings(), embedding_dimensions=768)
    # The storage layer for the parent documents
    store = InMemoryByteStore()
    id_key = "id"

    import uuid

    doc_ids = [str(uuid.uuid4()) for _ in docs]
    # The splitter to use to create smaller chunks
    child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    sub_docs = []
    for i, doc in enumerate(docs):
        _id = doc_ids[i]
        _sub_docs = child_text_splitter.split_documents([doc])
        for _doc in _sub_docs:
            _doc.metadata[id_key] = _id
        sub_docs.extend(_sub_docs)

    # The retriever (empty to start)
    docstore = InMemoryStore()
    docstore.mset(list(zip(doc_ids, docs)))
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore, byte_store=store, id_key=id_key, docstore=docstore
    )
    retriever.vectorstore.add_documents(sub_docs)

    # Vectorstore alone retrieves the small chunks
    _ = retriever.vectorstore.similarity_search("justice breyer")[0]

    # Retriever returns larger chunks
    assert len(retriever.invoke("justice breyer")[0].page_content) == 9875

    docs = text_splitter.split_documents(docs)
