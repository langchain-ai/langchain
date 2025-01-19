"""Test FAISS functionality."""

import datetime
import math
import tempfile
from typing import Union

import pytest
from langchain_core.documents import Document

from langchain_community.docstore.base import Docstore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

_PAGE_CONTENT = """This is a page about LangChain.

It is a really cool framework.

What isn't there to love about langchain?

Made in 2022."""


class FakeDocstore(Docstore):
    """Fake docstore for testing purposes."""

    def search(self, search: str) -> Union[str, Document]:
        """Return the fake document."""
        document = Document(page_content=_PAGE_CONTENT)
        return document


@pytest.mark.requires("faiss")
def test_faiss() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings())
    index_to_id = docsearch.index_to_docstore_id
    expected_docstore = InMemoryDocstore(
        {
            index_to_id[0]: Document(id=index_to_id[0], page_content="foo"),
            index_to_id[1]: Document(id=index_to_id[1], page_content="bar"),
            index_to_id[2]: Document(id=index_to_id[2], page_content="baz"),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(id=output[0].id, page_content="foo")]

    # Retriever standard params
    retriever = docsearch.as_retriever()
    ls_params = retriever._get_ls_params()
    assert ls_params == {
        "ls_retriever_name": "vectorstore",
        "ls_vector_store_provider": "FAISS",
        "ls_embedding_provider": "FakeEmbeddings",
    }


@pytest.mark.requires("faiss")
async def test_faiss_afrom_texts() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = await FAISS.afrom_texts(texts, FakeEmbeddings())
    index_to_id = docsearch.index_to_docstore_id
    expected_docstore = InMemoryDocstore(
        {
            index_to_id[0]: Document(id=index_to_id[0], page_content="foo"),
            index_to_id[1]: Document(id=index_to_id[1], page_content="bar"),
            index_to_id[2]: Document(id=index_to_id[2], page_content="baz"),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    output = await docsearch.asimilarity_search("foo", k=1)
    assert output == [Document(id=output[0].id, page_content="foo")]


@pytest.mark.requires("faiss")
def test_faiss_vector_sim() -> None:
    """Test vector similarity."""
    texts = ["foo", "bar", "baz"]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings())
    index_to_id = docsearch.index_to_docstore_id
    expected_docstore = InMemoryDocstore(
        {
            index_to_id[0]: Document(id=index_to_id[0], page_content="foo"),
            index_to_id[1]: Document(id=index_to_id[1], page_content="bar"),
            index_to_id[2]: Document(id=index_to_id[2], page_content="baz"),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.similarity_search_by_vector(query_vec, k=1)
    assert output == [Document(id=output[0].id, page_content="foo")]


@pytest.mark.requires("faiss")
async def test_faiss_async_vector_sim() -> None:
    """Test vector similarity."""
    texts = ["foo", "bar", "baz"]
    docsearch = await FAISS.afrom_texts(texts, FakeEmbeddings())
    index_to_id = docsearch.index_to_docstore_id
    expected_docstore = InMemoryDocstore(
        {
            index_to_id[0]: Document(id=index_to_id[0], page_content="foo"),
            index_to_id[1]: Document(id=index_to_id[1], page_content="bar"),
            index_to_id[2]: Document(id=index_to_id[2], page_content="baz"),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    query_vec = await FakeEmbeddings().aembed_query(text="foo")
    output = await docsearch.asimilarity_search_by_vector(query_vec, k=1)
    assert output == [Document(id=output[0].id, page_content="foo")]


@pytest.mark.requires("faiss")
def test_faiss_vector_sim_with_score_threshold() -> None:
    """Test vector similarity."""
    texts = ["foo", "bar", "baz"]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings())
    index_to_id = docsearch.index_to_docstore_id
    expected_docstore = InMemoryDocstore(
        {
            index_to_id[0]: Document(id=index_to_id[0], page_content="foo"),
            index_to_id[1]: Document(id=index_to_id[1], page_content="bar"),
            index_to_id[2]: Document(id=index_to_id[2], page_content="baz"),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.similarity_search_by_vector(query_vec, k=2, score_threshold=0.2)
    assert output == [Document(id=output[0].id, page_content="foo")]


@pytest.mark.requires("faiss")
async def test_faiss_vector_async_sim_with_score_threshold() -> None:
    """Test vector similarity."""
    texts = ["foo", "bar", "baz"]
    docsearch = await FAISS.afrom_texts(texts, FakeEmbeddings())
    index_to_id = docsearch.index_to_docstore_id
    expected_docstore = InMemoryDocstore(
        {
            index_to_id[0]: Document(id=index_to_id[0], page_content="foo"),
            index_to_id[1]: Document(id=index_to_id[1], page_content="bar"),
            index_to_id[2]: Document(id=index_to_id[2], page_content="baz"),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    query_vec = await FakeEmbeddings().aembed_query(text="foo")
    output = await docsearch.asimilarity_search_by_vector(
        query_vec, k=2, score_threshold=0.2
    )
    assert output == [Document(id=output[0].id, page_content="foo")]


@pytest.mark.requires("faiss")
def test_similarity_search_with_score_by_vector() -> None:
    """Test vector similarity with score by vector."""
    texts = ["foo", "bar", "baz"]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings())
    index_to_id = docsearch.index_to_docstore_id
    expected_docstore = InMemoryDocstore(
        {
            index_to_id[0]: Document(id=index_to_id[0], page_content="foo"),
            index_to_id[1]: Document(id=index_to_id[1], page_content="bar"),
            index_to_id[2]: Document(id=index_to_id[2], page_content="baz"),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.similarity_search_with_score_by_vector(query_vec, k=1)
    assert len(output) == 1
    assert output[0][0] == Document(id=output[0][0].id, page_content="foo")


@pytest.mark.requires("faiss")
async def test_similarity_async_search_with_score_by_vector() -> None:
    """Test vector similarity with score by vector."""
    texts = ["foo", "bar", "baz"]
    docsearch = await FAISS.afrom_texts(texts, FakeEmbeddings())
    index_to_id = docsearch.index_to_docstore_id
    expected_docstore = InMemoryDocstore(
        {
            index_to_id[0]: Document(id=index_to_id[0], page_content="foo"),
            index_to_id[1]: Document(id=index_to_id[1], page_content="bar"),
            index_to_id[2]: Document(id=index_to_id[2], page_content="baz"),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    query_vec = await FakeEmbeddings().aembed_query(text="foo")
    output = await docsearch.asimilarity_search_with_score_by_vector(query_vec, k=1)
    assert len(output) == 1
    assert output[0][0] == Document(id=output[0][0].id, page_content="foo")


@pytest.mark.requires("faiss")
def test_similarity_search_with_score_by_vector_with_score_threshold() -> None:
    """Test vector similarity with score by vector."""
    texts = ["foo", "bar", "baz"]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings())
    index_to_id = docsearch.index_to_docstore_id
    expected_docstore = InMemoryDocstore(
        {
            index_to_id[0]: Document(id=index_to_id[0], page_content="foo"),
            index_to_id[1]: Document(id=index_to_id[1], page_content="bar"),
            index_to_id[2]: Document(id=index_to_id[2], page_content="baz"),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.similarity_search_with_score_by_vector(
        query_vec,
        k=2,
        score_threshold=0.2,
    )
    assert len(output) == 1
    assert output[0][0] == Document(id=output[0][0].id, page_content="foo")
    assert output[0][1] < 0.2


@pytest.mark.requires("faiss")
async def test_sim_asearch_with_score_by_vector_with_score_threshold() -> None:
    """Test vector similarity with score by vector."""
    texts = ["foo", "bar", "baz"]
    docsearch = await FAISS.afrom_texts(texts, FakeEmbeddings())
    index_to_id = docsearch.index_to_docstore_id
    expected_docstore = InMemoryDocstore(
        {
            index_to_id[0]: Document(id=index_to_id[0], page_content="foo"),
            index_to_id[1]: Document(id=index_to_id[1], page_content="bar"),
            index_to_id[2]: Document(id=index_to_id[2], page_content="baz"),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    query_vec = await FakeEmbeddings().aembed_query(text="foo")
    output = await docsearch.asimilarity_search_with_score_by_vector(
        query_vec,
        k=2,
        score_threshold=0.2,
    )
    assert len(output) == 1
    assert output[0][0] == Document(id=output[0][0].id, page_content="foo")
    assert output[0][1] < 0.2


@pytest.mark.requires("faiss")
def test_faiss_mmr() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings())
    query_vec = FakeEmbeddings().embed_query(text="foo")
    # make sure we can have k > docstore size
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1
    )
    assert len(output) == len(texts)
    assert output[0][0] == Document(id=output[0][0].id, page_content="foo")
    assert output[0][1] == 0.0
    assert output[1][0] != Document(id=output[1][0].id, page_content="foo")


@pytest.mark.requires("faiss")
async def test_faiss_async_mmr() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    docsearch = await FAISS.afrom_texts(texts, FakeEmbeddings())
    query_vec = await FakeEmbeddings().aembed_query(text="foo")
    # make sure we can have k > docstore size
    output = await docsearch.amax_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1
    )
    assert len(output) == len(texts)
    assert output[0][0] == Document(id=output[0][0].id, page_content="foo")
    assert output[0][1] == 0.0
    assert output[1][0] != Document(id=output[1][0].id, page_content="foo")


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1
    )
    assert len(output) == len(texts)
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="foo", metadata={"page": 0}
    )
    assert output[0][1] == 0.0
    assert output[1][0] != Document(
        id=output[1][0].id, page_content="foo", metadata={"page": 0}
    )


@pytest.mark.requires("faiss")
async def test_faiss_async_mmr_with_metadatas() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = await FAISS.afrom_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = await FakeEmbeddings().aembed_query(text="foo")
    output = await docsearch.amax_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1
    )
    assert len(output) == len(texts)
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="foo", metadata={"page": 0}
    )
    assert output[0][1] == 0.0
    assert output[1][0] != Document(
        id=output[1][0].id, page_content="foo", metadata={"page": 0}
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_filter() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={"page": 1}
    )
    assert len(output) == 1
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="foo", metadata={"page": 1}
    )
    assert output[0][1] == 0.0
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter=lambda di: di["page"] == 1
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_comparison_operators_filter_eq() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={"page": {"$eq": 1}}
    )
    assert len(output) == 1
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="foo", metadata={"page": 1}
    )
    assert output[0][1] == 0.0
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter=lambda di: di["page"] == 1
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_comparison_operators_filter_neq() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={"page": {"$neq": 1}}
    )
    assert len(output) == 3
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="foo", metadata={"page": 0}
    )
    assert output[0][1] == 0.0
    assert output[1][0] != Document(
        id=output[1][0].id, page_content="foo", metadata={"page": 0}
    )
    assert output[2][0] != Document(
        id=output[2][0].id, page_content="foo", metadata={"page": 0}
    )
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter=lambda di: di["page"] != 1
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_comparison_operators_filter_gt() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={"page": {"$gt": 0}}
    )
    assert len(output) == 3
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="foo", metadata={"page": 1}
    )
    assert output[0][1] == 0.0
    assert output[1][0] != Document(
        id=output[1][0].id, page_content="foo", metadata={"page": 1}
    )
    assert output[2][0] != Document(
        id=output[2][0].id, page_content="foo", metadata={"page": 1}
    )
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter=lambda di: di["page"] > 0
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_comparison_operators_filter_lt() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={"page": {"$lt": 2}}
    )
    assert len(output) == 2
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="foo", metadata={"page": 0}
    )
    assert output[0][1] == 0.0
    assert output[1][0] == Document(
        id=output[1][0].id, page_content="foo", metadata={"page": 1}
    )
    assert output[1][1] == 1.0
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter=lambda di: di["page"] < 2
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_comparison_operators_filter_gte() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={"page": {"$gte": 1}}
    )
    assert len(output) == 3
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="foo", metadata={"page": 1}
    )
    assert output[0][1] == 0.0
    assert output[1][0] != Document(
        id=output[1][0].id, page_content="foo", metadata={"page": 1}
    )
    assert output[2][0] != Document(
        id=output[2][0].id, page_content="foo", metadata={"page": 1}
    )
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter=lambda di: di["page"] >= 1
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_comparison_operators_filter_lte() -> None:
    texts = ["fou", "fou", "fouu", "fouuu"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={"page": {"$lte": 0}}
    )
    assert len(output) == 1
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="fou", metadata={"page": 0}
    )
    assert output[0][1] == 0.0
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter=lambda di: di["page"] <= 0
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_comparison_operators_filter_in_1() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={"page": {"$in": [0]}}
    )
    assert len(output) == 1
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="foo", metadata={"page": 0}
    )
    assert output[0][1] == 0.0
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter=lambda di: di["page"] in [0]
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_comparison_operators_filter_in_2() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={"page": {"$in": [1, 2]}}
    )
    assert len(output) == 2
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="foo", metadata={"page": 1}
    )
    assert output[0][1] == 0.0
    assert output[1][0] == Document(
        id=output[1][0].id, page_content="fou", metadata={"page": 2}
    )
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter=lambda di: di["page"] in [1, 2]
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_comparison_operators_filter_nin_1() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={"page": {"$nin": [0, 1]}}
    )
    assert len(output) == 2
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="fou", metadata={"page": 2}
    )
    assert output[0][1] == 0.0
    assert output[1][0] == Document(
        id=output[1][0].id, page_content="foy", metadata={"page": 3}
    )
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter=lambda di: di["page"] not in [0, 1]
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_comparison_operators_filter_nin_2() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={"page": {"$nin": [0, 1, 2]}}
    )
    assert len(output) == 1
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="foy", metadata={"page": 3}
    )
    assert output[0][1] == 0.0
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter=lambda di: di["page"] not in [0, 1, 2]
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_logical_operators_filter_not() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={"$not": {"page": 1}}
    )
    assert len(output) == 3
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="foo", metadata={"page": 0}
    )
    assert output[0][1] == 0.0
    assert output[1][0] == Document(
        id=output[1][0].id, page_content="foy", metadata={"page": 3}
    )
    assert output[2][0] == Document(
        id=output[2][0].id, page_content="fou", metadata={"page": 2}
    )
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter=lambda di: not di["page"] == 1
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_logical_operators_filter_or_1() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={"$or": [{"page": 0}]}
    )
    assert len(output) == 1
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="foo", metadata={"page": 0}
    )
    assert output[0][1] == 0.0
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter=lambda di: (di["page"] == 0)
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_logical_operators_filter_or_2() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={"$or": [{"page": 0}, {"page": 1}]}
    )
    assert len(output) == 2
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="foo", metadata={"page": 0}
    )
    assert output[0][1] == 0.0
    assert output[1][0] == Document(
        id=output[1][0].id, page_content="foo", metadata={"page": 1}
    )
    assert output[1][1] == 1.0
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter=lambda di: (di["page"] == 0) or (di["page"] == 1),
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_logical_operators_filter_or_3() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter={"$or": [{"page": 0}, {"page": 1}, {"page": 2}]},
    )
    assert len(output) == 3
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="foo", metadata={"page": 0}
    )
    assert output[0][1] == 0.0
    assert output[1][0] != Document(
        id=output[1][0].id, page_content="foo", metadata={"page": 0}
    )
    assert output[2][0] != Document(
        id=output[2][0].id, page_content="foo", metadata={"page": 0}
    )
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter=lambda di: (di["page"] == 0) or (di["page"] == 1) or (di["page"] == 2),
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_logical_operators_filter_and_1() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={"$and": [{"page": 0}]}
    )
    assert len(output) == 1
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="foo", metadata={"page": 0}
    )
    assert output[0][1] == 0.0
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter=lambda di: (di["page"] == 0)
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_logical_operators_filter_and_2() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={"$and": [{"page": 0}, {"page": 1}]}
    )
    assert len(output) == 0
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter=lambda di: (di["page"] == 0) and (di["page"] == 1),
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_logical_operators_filter_and_3() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter={"$and": [{"page": 0}, {"page": 1}, {"page": 2}]},
    )
    assert len(output) == 0
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter=lambda di: (di["page"] == 0) and (di["page"] == 1) and (di["page"] == 2),
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_logical_operators_filter_and_4() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter={"$and": [{"page": 0}, {"page": 0}, {"page": 0}]},
    )
    assert len(output) == 1
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="foo", metadata={"page": 0}
    )
    assert output[0][1] == 0.0
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter=lambda di: (di["page"] == 0) and (di["page"] == 0) and (di["page"] == 0),
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_nested_logical_operators_filter_1() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter={"$and": [{"$or": [{"page": 1}, {"page": 2}]}, {"$not": {"page": 1}}]},
    )
    assert len(output) == 1
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="fou", metadata={"page": 2}
    )
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter=lambda di: (di["page"] == 1 or di["page"] == 2)
        and (not di["page"] == 1),
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_nested_logical_operators_filter_2() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter={
            "$and": [
                {"$or": [{"page": 1}, {"page": 2}]},
                {"$or": [{"page": 3}, {"page": 2}, {"page": 0}]},
            ]
        },
    )
    assert len(output) == 1
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="fou", metadata={"page": 2}
    )
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter=lambda di: (di["page"] == 1 or di["page"] == 2)
        and (di["page"] == 3 or di["page"] == 2 or di["page"] == 0),
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_nested_logical_operators_filter_3() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter={
            "$or": [
                {"$and": [{"page": 1}, {"page": 2}]},
                {"$and": [{"page": 0}, {"page": 2}]},
            ]
        },
    )
    assert len(output) == 0
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter=lambda di: (di["page"] == 1 and di["page"] == 2)
        or (di["page"] == 0 and di["page"] == 2),
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_logical_comparsion_operators_filter_1() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter={"$or": [{"page": {"$lt": 1}}, {"page": {"$gt": 2}}]},
    )
    assert len(output) == 2
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="foo", metadata={"page": 0}
    )
    assert output[0][1] == 0.0
    assert output[1][0] == Document(
        id=output[1][0].id, page_content="foy", metadata={"page": 3}
    )
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter=lambda di: di["page"] < 1 or di["page"] > 2,
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_logical_comparsion_operators_filter_2() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={"$not": {"page": {"$lt": 1}}}
    )
    assert len(output) == 3
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="foo", metadata={"page": 1}
    )
    assert output[0][1] == 0.0
    assert output[1][0] == Document(
        id=output[1][0].id, page_content="foy", metadata={"page": 3}
    )
    assert output[2][0] == Document(
        id=output[2][0].id, page_content="fou", metadata={"page": 2}
    )
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter=lambda di: not di["page"] < 1
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_nested_logical_comparsion_ops_filter_1() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter={
            "$and": [
                {"$or": [{"page": {"$lt": 1}}, {"page": {"$gt": 2}}]},
                {"$or": [{"page": {"$eq": 0}}, {"page": {"$eq": 1}}]},
            ]
        },
    )
    assert len(output) == 1
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="foo", metadata={"page": 0}
    )
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter=lambda di: (di["page"] < 1 or di["page"] > 2)
        and (di["page"] == 0 or di["page"] == 1),
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_nested_logical_comparsion_ops_filter_2() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter={
            "$and": [
                {"$or": [{"page": {"$lt": 1}}, {"page": {"$gt": 2}}]},
                {"$not": {"page": {"$in": [0]}}},
                {"page": {"$neq": 3}},
            ]
        },
    )
    assert len(output) == 0
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter=lambda di: (di["page"] < 1 or di["page"] > 2)
        and (di["page"] not in [0])
        and (di["page"] != 3),
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_nested_logical_comparsion_ops_filter_3() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter={
            "$or": [
                {"$and": [{"page": {"$lt": 1}}, {"page": {"$gt": 2}}]},
                {"$not": {"page": {"$nin": [0]}}},
                {"page": {"$eq": 3}},
            ]
        },
    )
    assert len(output) == 2
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="foo", metadata={"page": 0}
    )
    assert output[0][1] == 0.0
    assert output[1][0] == Document(
        id=output[1][0].id, page_content="foy", metadata={"page": 3}
    )
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter=lambda di: (di["page"] < 1 and di["page"] > 2)
        or (not di["page"] not in [0])
        or (di["page"] == 3),
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_empty_conditions() -> None:
    """Test with an empty filter condition."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")

    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={}
    )

    assert len(output) == 3
    assert all(doc[0].page_content in ["foo", "bar", "baz"] for doc in output)


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadates_and_empty_and_operator() -> None:
    """Test with an empty $and operator."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")

    # Using an empty $and filter
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={"$and": []}
    )

    assert len(output) == 3
    assert all(doc[0].page_content in ["foo", "bar", "baz"] for doc in output)


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_empty_or_operator() -> None:
    """Test with an empty $or operator."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")

    # Using an empty $or filter
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={"$or": []}
    )

    assert len(output) == 0


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_nonexistent_field() -> None:
    """Test with a non-existent field in the metadata."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")

    # Using a filter with a non-existent field
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={"nonexistent_field": {"$eq": 1}}
    )

    assert len(output) == 0  # Expecting no documents to match


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_invalid_logical_operator() -> None:
    """Test with an invalid logical operator key."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")

    # Using a filter with an invalid logical operator
    with pytest.raises(ValueError, match="unsupported operator"):
        docsearch.max_marginal_relevance_search_with_score_by_vector(
            query_vec, k=10, lambda_mult=0.1, filter={"$unknown": [{"page": 1}]}
        )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_invalid_comparison_operator() -> None:
    """Test with an invalid comparison operator key."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")

    # Using a filter with an invalid comparison operator
    with pytest.raises(ValueError, match="unsupported operator"):
        docsearch.max_marginal_relevance_search_with_score_by_vector(
            query_vec, k=10, lambda_mult=0.1, filter={"page": {"$invalid_operator": 1}}
        )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_valid_invalid_fields() -> None:
    """Test with logical operators combining valid and invalid field."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")

    # Using a filter with $and combining valid and invalid fields
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter={
            "$and": [
                {"page": {"$eq": 1}},  # Valid field
                {"invalid_field": {"$eq": 1}},  # Invalid field
            ]
        },
    )
    # Expecting no documents to match due to the invalid field
    assert len(output) == 0


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_valid_and_invalid_operators() -> None:
    """Test with logical operators combining valid and invalid operators."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")

    # Using a filter with $and combining valid and invalid operators
    with pytest.raises(ValueError, match="unsupported operator"):
        docsearch.max_marginal_relevance_search_with_score_by_vector(
            query_vec,
            k=10,
            lambda_mult=0.1,
            filter={
                "$and": [
                    {"page": {"$eq": 1}},  # Valid condition
                    {"page": {"$unknown": 2}},  # Invalid operator
                ]
            },
        )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_multiple_nested_logical_operators() -> None:
    """Test with multiple nested logical operators."""
    texts = ["foo", "bar", "baz", "qux", "quux"]
    metadatas = [
        {"page": 1, "chapter": 1, "section": 3},
        {"page": 2, "chapter": 2, "section": 4},
        {"page": 1, "chapter": 3, "section": 6},
        {"page": 3, "chapter": 2, "section": 5},
        {"page": 4, "chapter": 1, "section": 2},
    ]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")

    # Using a filter with multiple nested logical operators
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter={
            "$and": [
                {"$or": [{"page": {"$eq": 1}}, {"chapter": {"$gt": 2}}]},
                {"$not": {"section": {"$lte": 5}}},
            ]
        },
    )
    assert len(output) > 0
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter=lambda di: (
            (di["page"] == 1 or di["chapter"] > 2) and di["section"] > 5
        ),
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_mixed_data_types() -> None:
    """Test with metadata containing mixed data types (numbers, strings, booleans)."""
    texts = ["foo", "bar", "baz", "qux", "quux"]
    metadatas: list[dict] = [
        {"page": "1", "isActive": True, "priority": 2.5},
        {"page": 2, "isActive": False, "priority": 3.0},
        {"page": 3, "isActive": True, "priority": 1.5},
        {"page": 1, "isActive": True, "priority": 4.0},
        {"page": 4, "isActive": False, "priority": 2.0},
    ]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")

    # Using a filter with mixed data types
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter={
            "page": {"$eq": "1"},  # String comparison
            "isActive": {"$eq": True},  # Boolean comparison
            "priority": {"$gt": 2.0},  # Numeric comparison
        },
    )
    # Assert output matches expected results based on the filter conditions
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter=lambda di: (
            di["page"] == "1" and di["isActive"] is True and di["priority"] > 2.0
        ),
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_conflicting_conditions() -> None:
    """Test with conflicting conditions in filters."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")

    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=10,
        lambda_mult=0.1,
        filter={"$and": [{"page": {"$eq": 1}}, {"page": {"$eq": 2}}]},
    )
    # Assert that the output is empty due to conflicting conditions
    assert len(output) == 0


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_null_field_values() -> None:
    """Test with fields that have null or undefined values."""
    texts = ["foo", "bar", "baz", "qux"]
    metadatas: list[dict] = [{"page": 1}, {"page": None}, {"page": 2}, {"page": None}]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    # Using a filter to find documents where page is null
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={"page": {"$eq": None}}
    )
    assert len(output) == 2
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter=lambda di: di["page"] is None
    )


@pytest.mark.requires("faiss")
async def test_faiss_async_mmr_with_metadatas_and_filter() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = await FAISS.afrom_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = await FakeEmbeddings().aembed_query(text="foo")
    output = await docsearch.amax_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={"page": 1}
    )
    assert len(output) == 1
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="foo", metadata={"page": 1}
    )
    assert output[0][1] == 0.0
    assert (
        output
        == await docsearch.amax_marginal_relevance_search_with_score_by_vector(
            query_vec, k=10, lambda_mult=0.1, filter=lambda di: di["page"] == 1
        )
    )


@pytest.mark.requires("faiss")
def test_faiss_mmr_with_metadatas_and_list_filter() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} if i <= 3 else {"page": 3} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={"page": [0, 1, 2]}
    )
    assert len(output) == 3
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="foo", metadata={"page": 0}
    )
    assert output[0][1] == 0.0
    assert output[1][0] != Document(
        id=output[1][0].id, page_content="foo", metadata={"page": 0}
    )
    assert output == docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter=lambda di: di["page"] in [0, 1, 2]
    )


@pytest.mark.requires("faiss")
async def test_faiss_async_mmr_with_metadatas_and_list_filter() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} if i <= 3 else {"page": 3} for i in range(len(texts))]
    docsearch = await FAISS.afrom_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    query_vec = await FakeEmbeddings().aembed_query(text="foo")
    output = await docsearch.amax_marginal_relevance_search_with_score_by_vector(
        query_vec, k=10, lambda_mult=0.1, filter={"page": [0, 1, 2]}
    )
    assert len(output) == 3
    assert output[0][0] == Document(
        id=output[0][0].id, page_content="foo", metadata={"page": 0}
    )
    assert output[0][1] == 0.0
    assert output[1][0] != Document(
        id=output[1][0].id, page_content="foo", metadata={"page": 0}
    )
    assert output == (
        await docsearch.amax_marginal_relevance_search_with_score_by_vector(
            query_vec, k=10, lambda_mult=0.1, filter=lambda di: di["page"] in [0, 1, 2]
        )
    )


@pytest.mark.requires("faiss")
def test_faiss_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    expected_docstore = InMemoryDocstore(
        {
            docsearch.index_to_docstore_id[0]: Document(
                id=docsearch.index_to_docstore_id[0],
                page_content="foo",
                metadata={"page": 0},
            ),
            docsearch.index_to_docstore_id[1]: Document(
                id=docsearch.index_to_docstore_id[1],
                page_content="bar",
                metadata={"page": 1},
            ),
            docsearch.index_to_docstore_id[2]: Document(
                id=docsearch.index_to_docstore_id[2],
                page_content="baz",
                metadata={"page": 2},
            ),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    output = docsearch.similarity_search("foo", k=1)
    assert output == [
        Document(id=output[0].id, page_content="foo", metadata={"page": 0})
    ]


@pytest.mark.requires("faiss")
async def test_faiss_async_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = await FAISS.afrom_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    expected_docstore = InMemoryDocstore(
        {
            docsearch.index_to_docstore_id[0]: Document(
                id=docsearch.index_to_docstore_id[0],
                page_content="foo",
                metadata={"page": 0},
            ),
            docsearch.index_to_docstore_id[1]: Document(
                id=docsearch.index_to_docstore_id[1],
                page_content="bar",
                metadata={"page": 1},
            ),
            docsearch.index_to_docstore_id[2]: Document(
                id=docsearch.index_to_docstore_id[2],
                page_content="baz",
                metadata={"page": 2},
            ),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    output = await docsearch.asimilarity_search("foo", k=1)
    assert output == [
        Document(id=output[0].id, page_content="foo", metadata={"page": 0})
    ]


@pytest.mark.requires("faiss")
def test_faiss_with_metadatas_and_filter() -> None:
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    expected_docstore = InMemoryDocstore(
        {
            docsearch.index_to_docstore_id[0]: Document(
                id=docsearch.index_to_docstore_id[0],
                page_content="foo",
                metadata={"page": 0},
            ),
            docsearch.index_to_docstore_id[1]: Document(
                id=docsearch.index_to_docstore_id[1],
                page_content="bar",
                metadata={"page": 1},
            ),
            docsearch.index_to_docstore_id[2]: Document(
                id=docsearch.index_to_docstore_id[2],
                page_content="baz",
                metadata={"page": 2},
            ),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    output = docsearch.similarity_search("foo", k=1, filter={"page": 1})
    # make sure it returns the result that matches the filter.
    # Not the one who's text matches better.
    assert output == [
        Document(id=output[0].id, page_content="bar", metadata={"page": 1})
    ]
    assert output == docsearch.similarity_search(
        "foo", k=1, filter=lambda di: di["page"] == 1
    )


@pytest.mark.requires("faiss")
async def test_faiss_async_with_metadatas_and_filter() -> None:
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = await FAISS.afrom_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    expected_docstore = InMemoryDocstore(
        {
            docsearch.index_to_docstore_id[0]: Document(
                id=docsearch.index_to_docstore_id[0],
                page_content="foo",
                metadata={"page": 0},
            ),
            docsearch.index_to_docstore_id[1]: Document(
                id=docsearch.index_to_docstore_id[1],
                page_content="bar",
                metadata={"page": 1},
            ),
            docsearch.index_to_docstore_id[2]: Document(
                id=docsearch.index_to_docstore_id[2],
                page_content="baz",
                metadata={"page": 2},
            ),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    output = await docsearch.asimilarity_search("foo", k=1, filter={"page": 1})
    # make sure it returns the result that matches the filter.
    # Not the one who's text matches better.
    assert output == [
        Document(id=output[0].id, page_content="bar", metadata={"page": 1})
    ]
    assert output == await docsearch.asimilarity_search(
        "foo", k=1, filter=lambda di: di["page"] == 1
    )


@pytest.mark.requires("faiss")
def test_faiss_with_metadatas_and_list_filter() -> None:
    texts = ["foo", "bar", "baz", "foo", "qux"]
    metadatas = [{"page": i} if i <= 3 else {"page": 3} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    expected_docstore = InMemoryDocstore(
        {
            docsearch.index_to_docstore_id[0]: Document(
                id=docsearch.index_to_docstore_id[0],
                page_content="foo",
                metadata={"page": 0},
            ),
            docsearch.index_to_docstore_id[1]: Document(
                id=docsearch.index_to_docstore_id[1],
                page_content="bar",
                metadata={"page": 1},
            ),
            docsearch.index_to_docstore_id[2]: Document(
                id=docsearch.index_to_docstore_id[2],
                page_content="baz",
                metadata={"page": 2},
            ),
            docsearch.index_to_docstore_id[3]: Document(
                id=docsearch.index_to_docstore_id[3],
                page_content="foo",
                metadata={"page": 3},
            ),
            docsearch.index_to_docstore_id[4]: Document(
                id=docsearch.index_to_docstore_id[4],
                page_content="qux",
                metadata={"page": 3},
            ),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    output = docsearch.similarity_search("foor", k=1, filter={"page": [0, 1, 2]})
    assert output == [
        Document(id=output[0].id, page_content="foo", metadata={"page": 0})
    ]
    assert output == docsearch.similarity_search(
        "foor", k=1, filter=lambda di: di["page"] in [0, 1, 2]
    )


@pytest.mark.requires("faiss")
async def test_faiss_async_with_metadatas_and_list_filter() -> None:
    texts = ["foo", "bar", "baz", "foo", "qux"]
    metadatas = [{"page": i} if i <= 3 else {"page": 3} for i in range(len(texts))]
    docsearch = await FAISS.afrom_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    expected_docstore = InMemoryDocstore(
        {
            docsearch.index_to_docstore_id[0]: Document(
                id=docsearch.index_to_docstore_id[0],
                page_content="foo",
                metadata={"page": 0},
            ),
            docsearch.index_to_docstore_id[1]: Document(
                id=docsearch.index_to_docstore_id[1],
                page_content="bar",
                metadata={"page": 1},
            ),
            docsearch.index_to_docstore_id[2]: Document(
                id=docsearch.index_to_docstore_id[2],
                page_content="baz",
                metadata={"page": 2},
            ),
            docsearch.index_to_docstore_id[3]: Document(
                id=docsearch.index_to_docstore_id[3],
                page_content="foo",
                metadata={"page": 3},
            ),
            docsearch.index_to_docstore_id[4]: Document(
                id=docsearch.index_to_docstore_id[4],
                page_content="qux",
                metadata={"page": 3},
            ),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    output = await docsearch.asimilarity_search("foor", k=1, filter={"page": [0, 1, 2]})
    assert output == [
        Document(id=output[0].id, page_content="foo", metadata={"page": 0})
    ]
    assert output == await docsearch.asimilarity_search(
        "foor", k=1, filter=lambda di: di["page"] in [0, 1, 2]
    )


@pytest.mark.requires("faiss")
def test_faiss_search_not_found() -> None:
    """Test what happens when document is not found."""
    texts = ["foo", "bar", "baz"]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings())
    # Get rid of the docstore to purposefully induce errors.
    docsearch.docstore = InMemoryDocstore({})
    with pytest.raises(ValueError):
        docsearch.similarity_search("foo")


@pytest.mark.requires("faiss")
async def test_faiss_async_search_not_found() -> None:
    """Test what happens when document is not found."""
    texts = ["foo", "bar", "baz"]
    docsearch = await FAISS.afrom_texts(texts, FakeEmbeddings())
    # Get rid of the docstore to purposefully induce errors.
    docsearch.docstore = InMemoryDocstore({})
    with pytest.raises(ValueError):
        await docsearch.asimilarity_search("foo")


@pytest.mark.requires("faiss")
def test_faiss_add_texts() -> None:
    """Test end to end adding of texts."""
    # Create initial doc store.
    texts = ["foo", "bar", "baz"]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings())
    # Test adding a similar document as before.
    docsearch.add_texts(["foo"])
    output = docsearch.similarity_search("foo", k=2)
    assert output == [
        Document(id=output[0].id, page_content="foo"),
        Document(id=output[1].id, page_content="foo"),
    ]


@pytest.mark.requires("faiss")
async def test_faiss_async_add_texts() -> None:
    """Test end to end adding of texts."""
    # Create initial doc store.
    texts = ["foo", "bar", "baz"]
    docsearch = await FAISS.afrom_texts(texts, FakeEmbeddings())
    # Test adding a similar document as before.
    await docsearch.aadd_texts(["foo"])
    output = await docsearch.asimilarity_search("foo", k=2)
    assert output == [
        Document(id=output[0].id, page_content="foo"),
        Document(id=output[1].id, page_content="foo"),
    ]


@pytest.mark.requires("faiss")
def test_faiss_add_texts_not_supported() -> None:
    """Test adding of texts to a docstore that doesn't support it."""
    docsearch = FAISS(FakeEmbeddings(), None, FakeDocstore(), {})
    with pytest.raises(ValueError):
        docsearch.add_texts(["foo"])


@pytest.mark.requires("faiss")
async def test_faiss_async_add_texts_not_supported() -> None:
    """Test adding of texts to a docstore that doesn't support it."""
    docsearch = FAISS(FakeEmbeddings(), None, FakeDocstore(), {})
    with pytest.raises(ValueError):
        await docsearch.aadd_texts(["foo"])


@pytest.mark.requires("faiss")
def test_faiss_local_save_load() -> None:
    """Test end to end serialization."""
    texts = ["foo", "bar", "baz"]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings())
    temp_timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    with tempfile.TemporaryDirectory(suffix="_" + temp_timestamp + "/") as temp_folder:
        docsearch.save_local(temp_folder)
        new_docsearch = FAISS.load_local(
            temp_folder, FakeEmbeddings(), allow_dangerous_deserialization=True
        )
    assert new_docsearch.index is not None


@pytest.mark.requires("faiss")
async def test_faiss_async_local_save_load() -> None:
    """Test end to end serialization."""
    texts = ["foo", "bar", "baz"]
    docsearch = await FAISS.afrom_texts(texts, FakeEmbeddings())
    temp_timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    with tempfile.TemporaryDirectory(suffix="_" + temp_timestamp + "/") as temp_folder:
        docsearch.save_local(temp_folder)
        new_docsearch = FAISS.load_local(
            temp_folder, FakeEmbeddings(), allow_dangerous_deserialization=True
        )
    assert new_docsearch.index is not None


@pytest.mark.requires("faiss")
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
    assert output == Document(id=output.id, page_content="foo")
    assert score == 1.0


@pytest.mark.requires("faiss")
async def test_faiss_async_similarity_search_with_relevance_scores() -> None:
    """Test the similarity search with normalized similarities."""
    texts = ["foo", "bar", "baz"]
    docsearch = await FAISS.afrom_texts(
        texts,
        FakeEmbeddings(),
        relevance_score_fn=lambda score: 1.0 - score / math.sqrt(2),
    )
    outputs = await docsearch.asimilarity_search_with_relevance_scores("foo", k=1)
    output, score = outputs[0]
    assert output == Document(id=output.id, page_content="foo")
    assert score == 1.0


@pytest.mark.requires("faiss")
def test_faiss_similarity_search_with_relevance_scores_with_threshold() -> None:
    """Test the similarity search with normalized similarities with score threshold."""
    texts = ["foo", "bar", "baz"]
    docsearch = FAISS.from_texts(
        texts,
        FakeEmbeddings(),
        relevance_score_fn=lambda score: 1.0 - score / math.sqrt(2),
    )
    outputs = docsearch.similarity_search_with_relevance_scores(
        "foo", k=2, score_threshold=0.5
    )
    assert len(outputs) == 1
    output, score = outputs[0]
    assert output == Document(id=output.id, page_content="foo")
    assert score == 1.0


@pytest.mark.requires("faiss")
async def test_faiss_asimilarity_search_with_relevance_scores_with_threshold() -> None:
    """Test the similarity search with normalized similarities with score threshold."""
    texts = ["foo", "bar", "baz"]
    docsearch = await FAISS.afrom_texts(
        texts,
        FakeEmbeddings(),
        relevance_score_fn=lambda score: 1.0 - score / math.sqrt(2),
    )
    outputs = await docsearch.asimilarity_search_with_relevance_scores(
        "foo", k=2, score_threshold=0.5
    )
    assert len(outputs) == 1
    output, score = outputs[0]
    assert output == Document(id=output.id, page_content="foo")
    assert score == 1.0


@pytest.mark.requires("faiss")
def test_faiss_invalid_normalize_fn() -> None:
    """Test the similarity search with normalized similarities."""
    texts = ["foo", "bar", "baz"]
    docsearch = FAISS.from_texts(
        texts, FakeEmbeddings(), relevance_score_fn=lambda _: 2.0
    )
    with pytest.warns(Warning, match="scores must be between"):
        docsearch.similarity_search_with_relevance_scores("foo", k=1)


@pytest.mark.requires("faiss")
async def test_faiss_async_invalid_normalize_fn() -> None:
    """Test the similarity search with normalized similarities."""
    texts = ["foo", "bar", "baz"]
    docsearch = await FAISS.afrom_texts(
        texts, FakeEmbeddings(), relevance_score_fn=lambda _: 2.0
    )
    with pytest.warns(Warning, match="scores must be between"):
        await docsearch.asimilarity_search_with_relevance_scores("foo", k=1)


@pytest.mark.requires("faiss")
def test_missing_normalize_score_fn() -> None:
    """Test doesn't perform similarity search without a valid distance strategy."""
    texts = ["foo", "bar", "baz"]
    faiss_instance = FAISS.from_texts(texts, FakeEmbeddings(), distance_strategy="fake")
    with pytest.raises(ValueError):
        faiss_instance.similarity_search_with_relevance_scores("foo", k=2)


@pytest.mark.skip(reason="old relevance score feature")
@pytest.mark.requires("faiss")
def test_ip_score() -> None:
    embedding = FakeEmbeddings()
    vector = embedding.embed_query("hi")
    assert vector == [1] * 9 + [0], f"FakeEmbeddings() has changed, produced {vector}"

    db = FAISS.from_texts(
        ["sundays coming so i drive my car"],
        embedding=FakeEmbeddings(),
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    scores = db.similarity_search_with_relevance_scores("sundays", k=1)
    assert len(scores) == 1, "only one vector should be in db"
    _, score = scores[0]
    assert (
        score == 1
    ), f"expected inner product of equivalent vectors to be 1, not {score}"


@pytest.mark.requires("faiss")
async def test_async_missing_normalize_score_fn() -> None:
    """Test doesn't perform similarity search without a valid distance strategy."""
    texts = ["foo", "bar", "baz"]
    faiss_instance = await FAISS.afrom_texts(
        texts, FakeEmbeddings(), distance_strategy="fake"
    )
    with pytest.raises(ValueError):
        await faiss_instance.asimilarity_search_with_relevance_scores("foo", k=2)


@pytest.mark.requires("faiss")
def test_delete() -> None:
    """Test the similarity search with normalized similarities."""
    ids = ["a", "b", "c"]
    docsearch = FAISS.from_texts(["foo", "bar", "baz"], FakeEmbeddings(), ids=ids)
    docsearch.delete(ids[1:2])

    result = docsearch.similarity_search("bar", k=2)
    assert sorted([d.page_content for d in result]) == ["baz", "foo"]
    assert docsearch.index_to_docstore_id == {0: ids[0], 1: ids[2]}


@pytest.mark.requires("faiss")
async def test_async_delete() -> None:
    """Test the similarity search with normalized similarities."""
    ids = ["a", "b", "c"]
    docsearch = await FAISS.afrom_texts(
        ["foo", "bar", "baz"], FakeEmbeddings(), ids=ids
    )
    docsearch.delete(ids[1:2])

    result = await docsearch.asimilarity_search("bar", k=2)
    assert sorted([d.page_content for d in result]) == ["baz", "foo"]
    assert docsearch.index_to_docstore_id == {0: ids[0], 1: ids[2]}


@pytest.mark.requires("faiss")
def test_faiss_with_duplicate_ids() -> None:
    """Test whether FAISS raises an exception for duplicate ids."""
    texts = ["foo", "bar", "baz"]
    duplicate_ids = ["id1", "id1", "id2"]

    with pytest.raises(ValueError) as exc_info:
        FAISS.from_texts(texts, FakeEmbeddings(), ids=duplicate_ids)

    assert "Duplicate ids found in the ids list." in str(exc_info.value)


@pytest.mark.requires("faiss")
def test_faiss_document_ids() -> None:
    """Test whether FAISS assigns the correct document ids."""
    ids = ["id1", "id2", "id3"]
    texts = ["foo", "bar", "baz"]

    vstore = FAISS.from_texts(texts, FakeEmbeddings(), ids=ids)
    for id_, text in zip(ids, texts):
        doc = vstore.docstore.search(id_)
        assert isinstance(doc, Document)
        assert doc.id == id_
        assert doc.page_content == text


@pytest.mark.requires("faiss")
def test_faiss_get_by_ids() -> None:
    """Test FAISS `get_by_ids` method."""
    ids = ["id1", "id2", "id3"]
    texts = ["foo", "bar", "baz"]

    vstore = FAISS.from_texts(texts, FakeEmbeddings(), ids=ids)
    docs = vstore.get_by_ids(ids)
    assert len(docs) == 3
    assert {doc.id for doc in docs} == set(ids)

    for id_ in ids:
        res = vstore.get_by_ids([id_])
        assert len(res) == 1
        assert res[0].id == id_
