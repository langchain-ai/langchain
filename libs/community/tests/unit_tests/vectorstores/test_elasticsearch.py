"""Test Elasticsearch functionality."""

import pytest

from langchain_community.vectorstores.elasticsearch import (
    ApproxRetrievalStrategy,
    ElasticsearchStore,
)
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


@pytest.mark.requires("elasticsearch")
def test_elasticsearch_hybrid_scores_guard() -> None:
    """Ensure an error is raised when search with score in hybrid mode
    because in this case Elasticsearch does not return any score.
    """
    from elasticsearch import Elasticsearch

    query_string = "foo"
    embeddings = FakeEmbeddings()

    store = ElasticsearchStore(
        index_name="dummy_index",
        es_connection=Elasticsearch(hosts=["http://dummy-host:9200"]),
        embedding=embeddings,
        strategy=ApproxRetrievalStrategy(hybrid=True),
    )
    with pytest.raises(ValueError):
        store.similarity_search_with_score(query_string)

    embedded_query = embeddings.embed_query(query_string)
    with pytest.raises(ValueError):
        store.similarity_search_by_vector_with_relevance_scores(embedded_query)
