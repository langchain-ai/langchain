"""Test Atlas functionality."""
import time

from langchain.vectorstores import AtlasDB
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

ATLAS_TEST_API_KEY = "7xDPkYXSYDc1_ErdTPIcoAR9RNd8YDlkS3nVNXcVoIMZ6"


def test_atlas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = AtlasDB.from_texts(
        name="langchain_test_project" + str(time.time()),
        texts=texts,
        api_key=ATLAS_TEST_API_KEY,
        embedding=FakeEmbeddings(),
    )
    output = docsearch.similarity_search("foo", k=1)
    assert len(output) == 1
    assert output[0].page_content == "foo"


def test_atlas_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = AtlasDB.from_texts(
        name="langchain_test_project" + str(time.time()),
        texts=texts,
        api_key=ATLAS_TEST_API_KEY,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        reset_project_if_exists=True,
    )

    output = docsearch.similarity_search("foo", k=1)
    assert len(output) == 1
    assert output[0].page_content == "foo"
    assert output[0].metadata["page"] == "0"
