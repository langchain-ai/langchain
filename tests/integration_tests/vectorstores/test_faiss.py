"""Test FAISS functionality."""
import datetime
import math
import tempfile

import pytest

from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.docstore.wikipedia import Wikipedia
from langchain.vectorstores.faiss import FAISS
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


def test_faiss() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings())
    index_to_id = docsearch.index_to_docstore_id
    expected_docstore = InMemoryDocstore(
        {
            index_to_id[0]: Document(page_content="foo"),
            index_to_id[1]: Document(page_content="bar"),
            index_to_id[2]: Document(page_content="baz"),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_faiss_vector_sim() -> None:
    """Test vector similarity."""
    texts = ["foo", "bar", "baz"]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings())
    index_to_id = docsearch.index_to_docstore_id
    expected_docstore = InMemoryDocstore(
        {
            index_to_id[0]: Document(page_content="foo"),
            index_to_id[1]: Document(page_content="bar"),
            index_to_id[2]: Document(page_content="baz"),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.similarity_search_by_vector(query_vec, k=1)
    assert output == [Document(page_content="foo")]


def test_faiss_mmr() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings())
    query_vec = FakeEmbeddings().embed_query(text="foo")
    # make sure we can have k > docstore size
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1
    )
    assert len(output) == len(texts)
    assert output[0][0] == Document(page_content="foo")
    assert output[0][1] == 0.0
    assert output[1][0] != Document(page_content="foo")


def test_faiss_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    expected_docstore = InMemoryDocstore(
        {
            docsearch.index_to_docstore_id[0]: Document(
                page_content="foo", metadata={"page": 0}
            ),
            docsearch.index_to_docstore_id[1]: Document(
                page_content="bar", metadata={"page": 1}
            ),
            docsearch.index_to_docstore_id[2]: Document(
                page_content="baz", metadata={"page": 2}
            ),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": 0})]


def test_faiss_with_metadatas_and_filter() -> None:
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    expected_docstore = InMemoryDocstore(
        {
            docsearch.index_to_docstore_id[0]: Document(
                page_content="foo", metadata={"page": 0}
            ),
            docsearch.index_to_docstore_id[1]: Document(
                page_content="bar", metadata={"page": 1}
            ),
            docsearch.index_to_docstore_id[2]: Document(
                page_content="baz", metadata={"page": 2}
            ),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    output = docsearch.similarity_search("foo", k=1, filter={"page": 1})
    assert output == [Document(page_content="bar", metadata={"page": 1})]


def test_faiss_with_metadatas_and_list_filter() -> None:
    texts = ["foo", "bar", "baz", "foo", "qux"]
    metadatas = [{"page": i} if i <= 3 else {"page": 3} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    expected_docstore = InMemoryDocstore(
        {
            docsearch.index_to_docstore_id[0]: Document(
                page_content="foo", metadata={"page": 0}
            ),
            docsearch.index_to_docstore_id[1]: Document(
                page_content="bar", metadata={"page": 1}
            ),
            docsearch.index_to_docstore_id[2]: Document(
                page_content="baz", metadata={"page": 2}
            ),
            docsearch.index_to_docstore_id[3]: Document(
                page_content="foo", metadata={"page": 3}
            ),
            docsearch.index_to_docstore_id[4]: Document(
                page_content="qux", metadata={"page": 3}
            ),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    output = docsearch.similarity_search("foor", k=1, filter={"page": [0, 1, 2]})
    assert output == [Document(page_content="foo", metadata={"page": 0})]


def test_faiss_search_not_found() -> None:
    """Test what happens when document is not found."""
    texts = ["foo", "bar", "baz"]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings())
    # Get rid of the docstore to purposefully induce errors.
    docsearch.docstore = InMemoryDocstore({})
    with pytest.raises(ValueError):
        docsearch.similarity_search("foo")


def test_faiss_add_texts() -> None:
    """Test end to end adding of texts."""
    # Create initial doc store.
    texts = ["foo", "bar", "baz"]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings())
    # Test adding a similar document as before.
    docsearch.add_texts(["foo"])
    output = docsearch.similarity_search("foo", k=2)
    assert output == [Document(page_content="foo"), Document(page_content="foo")]


def test_faiss_add_texts_not_supported() -> None:
    """Test adding of texts to a docstore that doesn't support it."""
    docsearch = FAISS(FakeEmbeddings().embed_query, None, Wikipedia(), {})
    with pytest.raises(ValueError):
        docsearch.add_texts(["foo"])


def test_faiss_local_save_load() -> None:
    """Test end to end serialization."""
    texts = ["foo", "bar", "baz"]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings())
    temp_timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    with tempfile.TemporaryDirectory(suffix="_" + temp_timestamp + "/") as temp_folder:
        docsearch.save_local(temp_folder)
        new_docsearch = FAISS.load_local(temp_folder, FakeEmbeddings())
    assert new_docsearch.index is not None


def test_faiss_similarity_search_with_relevance_scores() -> None:
    """Test the similarity search with normalized similarities."""
    texts = ["foo", "bar", "baz"]
    docsearch = FAISS.from_texts(
        texts,
        FakeEmbeddings(),
        relevance_score_fn=lambda score: 1.0 - score / math.sqrt(2),
    )
    outputs = docsearch.similarity_search_with_relevance_scores("foo", k=1)
    output, score = outputs[0]
    assert output == Document(page_content="foo")
    assert score == 1.0


def test_faiss_invalid_normalize_fn() -> None:
    """Test the similarity search with normalized similarities."""
    texts = ["foo", "bar", "baz"]
    docsearch = FAISS.from_texts(
        texts, FakeEmbeddings(), relevance_score_fn=lambda _: 2.0
    )
    with pytest.warns(Warning, match="scores must be between"):
        docsearch.similarity_search_with_relevance_scores("foo", k=1)


def test_missing_normalize_score_fn() -> None:
    """Test doesn't perform similarity search without a normalize score function."""
    with pytest.raises(ValueError):
        texts = ["foo", "bar", "baz"]
        faiss_instance = FAISS.from_texts(texts, FakeEmbeddings())
        faiss_instance.relevance_score_fn = None
        faiss_instance.similarity_search_with_relevance_scores("foo", k=2)
