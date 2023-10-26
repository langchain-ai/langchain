"""Test ElasticSearch functionality."""
import logging
import os
import re
import uuid
from typing import Any, Dict, Generator, List, Union

import pytest

from langchain.docstore.document import Document
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
    FakeEmbeddings,
)

logging.basicConfig(level=logging.DEBUG)

"""
cd tests/integration_tests/vectorstores/docker-compose
docker-compose -f elasticsearch.yml up

By default runs against local docker instance of Elasticsearch.
To run against Elastic Cloud, set the following environment variables:
- ES_CLOUD_ID
- ES_USERNAME
- ES_PASSWORD

Some of the tests require the following models to be deployed in the ML Node:
- elser (can be downloaded and deployed through Kibana and trained models UI)
- sentence-transformers__all-minilm-l6-v2 (can be deployed 
  through API, loaded via eland)

These tests that require the models to be deployed are skipped by default. 
Enable them by adding the model name to the modelsDeployed list below.
"""

modelsDeployed: List[str] = [
    # "elser",
    # "sentence-transformers__all-minilm-l6-v2",
]


class TestElasticsearch:
    @classmethod
    def setup_class(cls) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

    @pytest.fixture(scope="class", autouse=True)
    def elasticsearch_connection(self) -> Union[dict, Generator[dict, None, None]]:
        # Running this integration test with Elastic Cloud
        # Required for in-stack inference testing (ELSER + model_id)
        from elasticsearch import Elasticsearch

        es_url = os.environ.get("ES_URL", "http://localhost:9200")
        cloud_id = os.environ.get("ES_CLOUD_ID")
        es_username = os.environ.get("ES_USERNAME", "elastic")
        es_password = os.environ.get("ES_PASSWORD", "changeme")

        if cloud_id:
            es = Elasticsearch(
                cloud_id=cloud_id,
                basic_auth=(es_username, es_password),
            )
            yield {
                "es_cloud_id": cloud_id,
                "es_user": es_username,
                "es_password": es_password,
            }

        else:
            # Running this integration test with local docker instance
            es = Elasticsearch(hosts=es_url)
            yield {"es_url": es_url}

        # Clear all indexes
        index_names = es.indices.get(index="_all").keys()
        for index_name in index_names:
            if index_name.startswith("test_"):
                es.indices.delete(index=index_name)
        es.indices.refresh(index="_all")

        # clear all test pipelines
        try:
            response = es.ingest.get_pipeline(id="test_*,*_sparse_embedding")

            for pipeline_id, _ in response.items():
                try:
                    es.ingest.delete_pipeline(id=pipeline_id)
                    print(f"Deleted pipeline: {pipeline_id}")
                except Exception as e:
                    print(f"Pipeline error: {e}")
        except Exception:
            pass

    @pytest.fixture(scope="function")
    def es_client(self) -> Any:
        # Running this integration test with Elastic Cloud
        # Required for in-stack inference testing (ELSER + model_id)
        from elastic_transport import Transport
        from elasticsearch import Elasticsearch

        class CustomTransport(Transport):
            requests = []

            def perform_request(self, *args, **kwargs):  # type: ignore
                self.requests.append(kwargs)
                return super().perform_request(*args, **kwargs)

        es_url = os.environ.get("ES_URL", "http://localhost:9200")
        cloud_id = os.environ.get("ES_CLOUD_ID")
        es_username = os.environ.get("ES_USERNAME", "elastic")
        es_password = os.environ.get("ES_PASSWORD", "changeme")

        if cloud_id:
            es = Elasticsearch(
                cloud_id=cloud_id,
                basic_auth=(es_username, es_password),
                transport_class=CustomTransport,
            )
            return es
        else:
            # Running this integration test with local docker instance
            es = Elasticsearch(hosts=es_url, transport_class=CustomTransport)
            return es

    @pytest.fixture(scope="function")
    def index_name(self) -> str:
        """Return the index name."""
        return f"test_{uuid.uuid4().hex}"

    def test_similarity_search_without_metadata(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test end to end construction and search without metadata."""

        def assert_query(query_body: dict, query: str) -> dict:
            assert query_body == {
                "knn": {
                    "field": "vector",
                    "filter": [],
                    "k": 1,
                    "num_candidates": 50,
                    "query_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                }
            }
            return query_body

        texts = ["foo", "bar", "baz"]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            **elasticsearch_connection,
            index_name=index_name,
        )
        output = docsearch.similarity_search("foo", k=1, custom_query=assert_query)
        assert output == [Document(page_content="foo")]

    @pytest.mark.asyncio
    async def test_similarity_search_without_metadat_async(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test end to end construction and search without metadata."""
        texts = ["foo", "bar", "baz"]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            **elasticsearch_connection,
            index_name=index_name,
        )
        output = await docsearch.asimilarity_search("foo", k=1)
        assert output == [Document(page_content="foo")]

    def test_add_embeddings(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """
        Test add_embeddings, which accepts pre-built embeddings instead of
         using inference for the texts.
        This allows you to separate the embeddings text and the page_content
         for better proximity between user's question and embedded text.
        For example, your embedding text can be a question, whereas page_content
         is the answer.
        """
        embeddings = ConsistentFakeEmbeddings()
        text_input = ["foo1", "foo2", "foo3"]
        metadatas = [{"page": i} for i in range(len(text_input))]

        """In real use case, embedding_input can be questions for each text"""
        embedding_input = ["foo2", "foo3", "foo1"]
        embedding_vectors = embeddings.embed_documents(embedding_input)

        docsearch = ElasticsearchStore._create_cls_from_kwargs(
            embeddings,
            **elasticsearch_connection,
            index_name=index_name,
        )
        docsearch.add_embeddings(list(zip(text_input, embedding_vectors)), metadatas)
        output = docsearch.similarity_search("foo1", k=1)
        assert output == [Document(page_content="foo3", metadata={"page": 2})]

    def test_similarity_search_with_metadata(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            ConsistentFakeEmbeddings(),
            metadatas=metadatas,
            **elasticsearch_connection,
            index_name=index_name,
        )

        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo", metadata={"page": 0})]

        output = docsearch.similarity_search("bar", k=1)
        assert output == [Document(page_content="bar", metadata={"page": 1})]

    def test_similarity_search_with_filter(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "foo", "foo"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            metadatas=metadatas,
            **elasticsearch_connection,
            index_name=index_name,
        )

        def assert_query(query_body: dict, query: str) -> dict:
            assert query_body == {
                "knn": {
                    "field": "vector",
                    "filter": [{"term": {"metadata.page": "1"}}],
                    "k": 3,
                    "num_candidates": 50,
                    "query_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                }
            }
            return query_body

        output = docsearch.similarity_search(
            query="foo",
            k=3,
            filter=[{"term": {"metadata.page": "1"}}],
            custom_query=assert_query,
        )
        assert output == [Document(page_content="foo", metadata={"page": 1})]

    def test_similarity_search_exact_search(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            **elasticsearch_connection,
            index_name=index_name,
            strategy=ElasticsearchStore.ExactRetrievalStrategy(),
        )

        expected_query = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",  # noqa: E501
                        "params": {
                            "query_vector": [
                                1.0,
                                1.0,
                                1.0,
                                1.0,
                                1.0,
                                1.0,
                                1.0,
                                1.0,
                                1.0,
                                0.0,
                            ]
                        },
                    },
                }
            }
        }

        def assert_query(query_body: dict, query: str) -> dict:
            assert query_body == expected_query
            return query_body

        output = docsearch.similarity_search("foo", k=1, custom_query=assert_query)
        assert output == [Document(page_content="foo")]

    def test_similarity_search_exact_search_with_filter(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            **elasticsearch_connection,
            index_name=index_name,
            metadatas=metadatas,
            strategy=ElasticsearchStore.ExactRetrievalStrategy(),
        )

        def assert_query(query_body: dict, query: str) -> dict:
            expected_query = {
                "query": {
                    "script_score": {
                        "query": {"bool": {"filter": [{"term": {"metadata.page": 0}}]}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",  # noqa: E501
                            "params": {
                                "query_vector": [
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    0.0,
                                ]
                            },
                        },
                    }
                }
            }
            assert query_body == expected_query
            return query_body

        output = docsearch.similarity_search(
            "foo",
            k=1,
            custom_query=assert_query,
            filter=[{"term": {"metadata.page": 0}}],
        )
        assert output == [Document(page_content="foo", metadata={"page": 0})]

    def test_similarity_search_exact_search_distance_dot_product(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            **elasticsearch_connection,
            index_name=index_name,
            strategy=ElasticsearchStore.ExactRetrievalStrategy(),
            distance_strategy="DOT_PRODUCT",
        )

        def assert_query(query_body: dict, query: str) -> dict:
            assert query_body == {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": """
            double value = dotProduct(params.query_vector, 'vector');
            return sigmoid(1, Math.E, -value); 
            """,
                            "params": {
                                "query_vector": [
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    0.0,
                                ]
                            },
                        },
                    }
                }
            }
            return query_body

        output = docsearch.similarity_search("foo", k=1, custom_query=assert_query)
        assert output == [Document(page_content="foo")]

    def test_similarity_search_exact_search_unknown_distance_strategy(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test end to end construction and search with unknown distance strategy."""

        with pytest.raises(KeyError):
            texts = ["foo", "bar", "baz"]
            ElasticsearchStore.from_texts(
                texts,
                FakeEmbeddings(),
                **elasticsearch_connection,
                index_name=index_name,
                strategy=ElasticsearchStore.ExactRetrievalStrategy(),
                distance_strategy="NOT_A_STRATEGY",
            )

    def test_max_marginal_relevance_search(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test max marginal relevance search."""
        texts = ["foo", "bar", "baz"]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            **elasticsearch_connection,
            index_name=index_name,
            strategy=ElasticsearchStore.ExactRetrievalStrategy(),
        )

        mmr_output = docsearch.max_marginal_relevance_search(texts[0], k=3, fetch_k=3)
        sim_output = docsearch.similarity_search(texts[0], k=3)
        assert mmr_output == sim_output

        mmr_output = docsearch.max_marginal_relevance_search(texts[0], k=2, fetch_k=3)
        assert len(mmr_output) == 2
        assert mmr_output[0].page_content == texts[0]
        assert mmr_output[1].page_content == texts[1]

        mmr_output = docsearch.max_marginal_relevance_search(
            texts[0], k=2, fetch_k=3, lambda_mult=0.1  # more diversity
        )
        assert len(mmr_output) == 2
        assert mmr_output[0].page_content == texts[0]
        assert mmr_output[1].page_content == texts[2]

        # if fetch_k < k, then the output will be less than k
        mmr_output = docsearch.max_marginal_relevance_search(texts[0], k=3, fetch_k=2)
        assert len(mmr_output) == 2

    def test_similarity_search_approx_with_hybrid_search(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            **elasticsearch_connection,
            index_name=index_name,
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=True),
        )

        def assert_query(query_body: dict, query: str) -> dict:
            assert query_body == {
                "knn": {
                    "field": "vector",
                    "filter": [],
                    "k": 1,
                    "num_candidates": 50,
                    "query_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                },
                "query": {
                    "bool": {
                        "filter": [],
                        "must": [{"match": {"text": {"query": "foo"}}}],
                    }
                },
                "rank": {"rrf": {}},
            }
            return query_body

        output = docsearch.similarity_search("foo", k=1, custom_query=assert_query)
        assert output == [Document(page_content="foo")]

    def test_similarity_search_approx_with_custom_query_fn(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """test that custom query function is called
        with the query string and query body"""

        def my_custom_query(query_body: dict, query: str) -> dict:
            assert query == "foo"
            assert query_body == {
                "knn": {
                    "field": "vector",
                    "filter": [],
                    "k": 1,
                    "num_candidates": 50,
                    "query_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                }
            }
            return {"query": {"match": {"text": {"query": "bar"}}}}

        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        docsearch = ElasticsearchStore.from_texts(
            texts, FakeEmbeddings(), **elasticsearch_connection, index_name=index_name
        )
        output = docsearch.similarity_search("foo", k=1, custom_query=my_custom_query)
        assert output == [Document(page_content="bar")]

    @pytest.mark.skipif(
        "sentence-transformers__all-minilm-l6-v2" not in modelsDeployed,
        reason="Sentence Transformers model not deployed in ML Node, skipping test",
    )
    def test_similarity_search_with_approx_infer_instack(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """test end to end with approx retrieval strategy and inference in-stack"""
        docsearch = ElasticsearchStore(
            index_name=index_name,
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(
                query_model_id="sentence-transformers__all-minilm-l6-v2"
            ),
            query_field="text_field",
            vector_query_field="vector_query_field.predicted_value",
            **elasticsearch_connection,
        )

        # setting up the pipeline for inference
        docsearch.client.ingest.put_pipeline(
            id="test_pipeline",
            processors=[
                {
                    "inference": {
                        "model_id": "sentence-transformers__all-minilm-l6-v2",
                        "field_map": {"query_field": "text_field"},
                        "target_field": "vector_query_field",
                    }
                }
            ],
        )

        # creating a new index with the pipeline,
        # not relying on langchain to create the index
        docsearch.client.indices.create(
            index=index_name,
            mappings={
                "properties": {
                    "text_field": {"type": "text"},
                    "vector_query_field": {
                        "properties": {
                            "predicted_value": {
                                "type": "dense_vector",
                                "dims": 384,
                                "index": True,
                                "similarity": "l2_norm",
                            }
                        }
                    },
                }
            },
            settings={"index": {"default_pipeline": "test_pipeline"}},
        )

        # adding documents to the index
        texts = ["foo", "bar", "baz"]

        for i, text in enumerate(texts):
            docsearch.client.create(
                index=index_name,
                id=str(i),
                document={"text_field": text, "metadata": {}},
            )

        docsearch.client.indices.refresh(index=index_name)

        def assert_query(query_body: dict, query: str) -> dict:
            assert query_body == {
                "knn": {
                    "filter": [],
                    "field": "vector_query_field.predicted_value",
                    "k": 1,
                    "num_candidates": 50,
                    "query_vector_builder": {
                        "text_embedding": {
                            "model_id": "sentence-transformers__all-minilm-l6-v2",
                            "model_text": "foo",
                        }
                    },
                }
            }
            return query_body

        output = docsearch.similarity_search("foo", k=1, custom_query=assert_query)
        assert output == [Document(page_content="foo")]

        output = docsearch.similarity_search("bar", k=1)
        assert output == [Document(page_content="bar")]

    @pytest.mark.skipif(
        "elser" not in modelsDeployed,
        reason="ELSER not deployed in ML Node, skipping test",
    )
    def test_similarity_search_with_sparse_infer_instack(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """test end to end with sparse retrieval strategy and inference in-stack"""
        texts = ["foo", "bar", "baz"]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            **elasticsearch_connection,
            index_name=index_name,
            strategy=ElasticsearchStore.SparseVectorRetrievalStrategy(),
        )
        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo")]

    def test_elasticsearch_with_relevance_score(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test to make sure the relevance score is scaled to 0-1."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        embeddings = FakeEmbeddings()

        docsearch = ElasticsearchStore.from_texts(
            index_name=index_name,
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            **elasticsearch_connection,
        )

        embedded_query = embeddings.embed_query("foo")
        output = docsearch.similarity_search_by_vector_with_relevance_scores(
            embedding=embedded_query, k=1
        )
        assert output == [(Document(page_content="foo", metadata={"page": "0"}), 1.0)]

    def test_elasticsearch_delete_ids(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test delete methods from vector store."""
        texts = ["foo", "bar", "baz", "gni"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = ElasticsearchStore(
            embedding=ConsistentFakeEmbeddings(),
            **elasticsearch_connection,
            index_name=index_name,
        )

        ids = docsearch.add_texts(texts, metadatas)
        output = docsearch.similarity_search("foo", k=10)
        assert len(output) == 4

        docsearch.delete(ids[1:3])
        output = docsearch.similarity_search("foo", k=10)
        assert len(output) == 2

        docsearch.delete(["not-existing"])
        output = docsearch.similarity_search("foo", k=10)
        assert len(output) == 2

        docsearch.delete([ids[0]])
        output = docsearch.similarity_search("foo", k=10)
        assert len(output) == 1

        docsearch.delete([ids[3]])
        output = docsearch.similarity_search("gni", k=10)
        assert len(output) == 0

    def test_elasticsearch_indexing_exception_error(
        self,
        elasticsearch_connection: dict,
        index_name: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test bulk exception logging is giving better hints."""
        from elasticsearch.helpers import BulkIndexError

        docsearch = ElasticsearchStore(
            embedding=ConsistentFakeEmbeddings(),
            **elasticsearch_connection,
            index_name=index_name,
        )

        docsearch.client.indices.create(
            index=index_name,
            mappings={"properties": {}},
            settings={"index": {"default_pipeline": "not-existing-pipeline"}},
        )

        texts = ["foo"]

        with pytest.raises(BulkIndexError):
            docsearch.add_texts(texts)

        error_reason = "pipeline with id [not-existing-pipeline] does not exist"
        log_message = f"First error reason: {error_reason}"

        assert log_message in caplog.text

    def test_elasticsearch_with_user_agent(
        self, es_client: Any, index_name: str
    ) -> None:
        """Test to make sure the user-agent is set correctly."""

        texts = ["foo", "bob", "baz"]
        ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            es_connection=es_client,
            index_name=index_name,
        )

        user_agent = es_client.transport.requests[0]["headers"]["User-Agent"]
        pattern = r"^langchain-py-vs/\d+\.\d+\.\d+$"
        match = re.match(pattern, user_agent)

        assert (
            match is not None
        ), f"The string '{user_agent}' does not match the expected pattern."

    def test_elasticsearch_with_internal_user_agent(
        self, elasticsearch_connection: Dict, index_name: str
    ) -> None:
        """Test to make sure the user-agent is set correctly."""

        texts = ["foo"]
        store = ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            **elasticsearch_connection,
            index_name=index_name,
        )

        user_agent = store.client._headers["User-Agent"]
        pattern = r"^langchain-py-vs/\d+\.\d+\.\d+$"
        match = re.match(pattern, user_agent)

        assert (
            match is not None
        ), f"The string '{user_agent}' does not match the expected pattern."

    def test_bulk_args(self, es_client: Any, index_name: str) -> None:
        """Test to make sure the user-agent is set correctly."""

        texts = ["foo", "bob", "baz"]
        ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            es_connection=es_client,
            index_name=index_name,
            bulk_kwargs={"chunk_size": 1},
        )

        # 1 for index exist, 1 for index create, 3 for index docs
        assert len(es_client.transport.requests) == 5  # type: ignore
