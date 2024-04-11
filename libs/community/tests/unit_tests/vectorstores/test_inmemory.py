from langchain_core.documents import Document

from langchain_community.vectorstores.inmemory import InMemoryVectorStore
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
)


async def test_inmemory() -> None:
    """Test end to end construction and search."""
    store = await InMemoryVectorStore.afrom_texts(
        ["foo", "bar", "baz"], ConsistentFakeEmbeddings()
    )
    output = await store.asimilarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    output = await store.asimilarity_search("bar", k=2)
    assert output == [Document(page_content="bar"), Document(page_content="baz")]

    output2 = await store.asimilarity_search_with_score("bar", k=2)
    assert output2[0][1] > output2[1][1]


async def test_inmemory_mmr() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    docsearch = await InMemoryVectorStore.afrom_texts(texts, ConsistentFakeEmbeddings())
    # make sure we can k > docstore size
    output = await docsearch.amax_marginal_relevance_search(
        "foo", k=10, lambda_mult=0.1
    )
    assert len(output) == len(texts)
    assert output[0] == Document(page_content="foo")
    assert output[1] == Document(page_content="foy")
