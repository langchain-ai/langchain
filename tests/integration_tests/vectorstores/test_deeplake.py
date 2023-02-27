"""Test Deep Lake functionality."""
from langchain.docstore.document import Document
from langchain.vectorstores import DeepLake
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


def test_deeplake() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = DeepLake.from_texts(
        dataset_path="mem://test_path", texts=texts, embedding=FakeEmbeddings()
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_deeplake_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = DeepLake.from_texts(
        dataset_path="mem://test_path",
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


def test_deeplakewith_persistence() -> None:
    """Test end to end construction and search, with persistence."""
    dataset_path = "./tests/persist_dir"
    texts = ["foo", "bar", "baz"]
    docsearch = DeepLake.from_texts(
        dataset_path=dataset_path,
        texts=texts,
        embedding=FakeEmbeddings(),
    )

    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    docsearch.persist()

    # Get a new VectorStore from the persisted directory
    docsearch = DeepLake(
        dataset_path=dataset_path,
        embedding_function=FakeEmbeddings(),
    )
    output = docsearch.similarity_search("foo", k=1)

    # Clean up
    docsearch.delete_dataset()

    # Persist doesn't need to be called again
    # Data will be automatically persisted on object deletion
    # Or on program exit
