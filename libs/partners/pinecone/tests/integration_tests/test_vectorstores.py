import os
import time
import uuid
from typing import List

import numpy as np
import pinecone  # type: ignore
import pytest  # type: ignore[import-not-found]
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings  # type: ignore[import-not-found]
from pinecone import PodSpec
from pytest_mock import MockerFixture  # type: ignore[import-not-found]

from langchain_pinecone import PineconeVectorStore

INDEX_NAME = "langchain-test-index"  # name of the index
NAMESPACE_NAME = "langchain-test-namespace"  # name of the namespace
DIMENSION = 1536  # dimension of the embeddings

DEFAULT_SLEEP = 20


class TestPinecone:
    index: "pinecone.Index"

    @classmethod
    def setup_class(cls) -> None:
        import pinecone

        client = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index_list = client.list_indexes()
        for i in index_list:
            if i["name"] == INDEX_NAME:
                client.delete_index(INDEX_NAME)
                break
        if len(index_list) > 0:
            time.sleep(DEFAULT_SLEEP)  # prevent race with creation
        client.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=PodSpec(environment="gcp-starter"),
        )

        cls.index = client.Index(INDEX_NAME)

        # insure the index is empty
        index_stats = cls.index.describe_index_stats()
        assert index_stats["dimension"] == DIMENSION
        if index_stats["namespaces"].get(NAMESPACE_NAME) is not None:
            assert index_stats["namespaces"][NAMESPACE_NAME]["vector_count"] == 0

    @classmethod
    def teardown_class(cls) -> None:
        index_stats = cls.index.describe_index_stats()
        for _namespace_name in index_stats["namespaces"].keys():
            cls.index.delete(delete_all=True, namespace=_namespace_name)

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        # delete all the vectors in the index
        print("called")  # noqa: T201
        try:
            self.index.delete(delete_all=True, namespace=NAMESPACE_NAME)
            time.sleep(DEFAULT_SLEEP)  # prevent race condition with previous step
        except Exception:
            # if namespace not found
            pass

    @pytest.fixture
    def embedding_openai(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings()

    @pytest.fixture
    def texts(self) -> List[str]:
        return ["foo", "bar", "baz"]

    def test_from_texts(
        self, texts: List[str], embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test end to end construction and search."""
        unique_id = uuid.uuid4().hex
        needs = f"foobuu {unique_id} booo"
        texts.insert(0, needs)

        docsearch = PineconeVectorStore.from_texts(
            texts=texts,
            embedding=embedding_openai,
            index_name=INDEX_NAME,
            namespace=NAMESPACE_NAME,
        )
        time.sleep(DEFAULT_SLEEP)  # prevent race condition
        output = docsearch.similarity_search(unique_id, k=1, namespace=NAMESPACE_NAME)
        assert output == [Document(page_content=needs)]

    def test_from_texts_with_metadatas(
        self, texts: List[str], embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test end to end construction and search."""

        unique_id = uuid.uuid4().hex
        needs = f"foobuu {unique_id} booo"
        texts = [needs] + texts

        metadatas = [{"page": i} for i in range(len(texts))]

        namespace = f"{NAMESPACE_NAME}-md"
        docsearch = PineconeVectorStore.from_texts(
            texts,
            embedding_openai,
            index_name=INDEX_NAME,
            metadatas=metadatas,
            namespace=namespace,
        )
        time.sleep(DEFAULT_SLEEP)  # prevent race condition
        output = docsearch.similarity_search(needs, k=1, namespace=namespace)

        # TODO: why metadata={"page": 0.0}) instead of {"page": 0}?
        assert output == [Document(page_content=needs, metadata={"page": 0.0})]

    def test_from_texts_with_scores(self, embedding_openai: OpenAIEmbeddings) -> None:
        """Test end to end construction and search with scores and IDs."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        print("metadatas", metadatas)  # noqa: T201
        docsearch = PineconeVectorStore.from_texts(
            texts,
            embedding_openai,
            index_name=INDEX_NAME,
            metadatas=metadatas,
            namespace=NAMESPACE_NAME,
        )
        print(texts)  # noqa: T201
        time.sleep(DEFAULT_SLEEP)  # prevent race condition
        output = docsearch.similarity_search_with_score(
            "foo", k=3, namespace=NAMESPACE_NAME
        )
        docs = [o[0] for o in output]
        scores = [o[1] for o in output]
        sorted_documents = sorted(docs, key=lambda x: x.metadata["page"])
        print(sorted_documents)  # noqa: T201

        # TODO: why metadata={"page": 0.0}) instead of {"page": 0}, etc???
        assert sorted_documents == [
            Document(page_content="foo", metadata={"page": 0.0}),
            Document(page_content="bar", metadata={"page": 1.0}),
            Document(page_content="baz", metadata={"page": 2.0}),
        ]
        assert scores[0] > scores[1] > scores[2]

    def test_from_existing_index_with_namespaces(
        self, embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test that namespaces are properly handled."""
        # Create two indexes with the same name but different namespaces
        texts_1 = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts_1))]
        PineconeVectorStore.from_texts(
            texts_1,
            embedding_openai,
            index_name=INDEX_NAME,
            metadatas=metadatas,
            namespace=f"{INDEX_NAME}-1",
        )

        texts_2 = ["foo2", "bar2", "baz2"]
        metadatas = [{"page": i} for i in range(len(texts_2))]

        PineconeVectorStore.from_texts(
            texts_2,
            embedding_openai,
            index_name=INDEX_NAME,
            metadatas=metadatas,
            namespace=f"{INDEX_NAME}-2",
        )

        time.sleep(DEFAULT_SLEEP)  # prevent race condition

        # Search with namespace
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embedding_openai,
            namespace=f"{INDEX_NAME}-1",
        )
        output = docsearch.similarity_search("foo", k=20, namespace=f"{INDEX_NAME}-1")
        # check that we don't get results from the other namespace
        page_contents = sorted(set([o.page_content for o in output]))
        assert all(content in ["foo", "bar", "baz"] for content in page_contents)
        assert all(content not in ["foo2", "bar2", "baz2"] for content in page_contents)

    def test_add_documents_with_ids(
        self, texts: List[str], embedding_openai: OpenAIEmbeddings
    ) -> None:
        ids = [uuid.uuid4().hex for _ in range(len(texts))]
        PineconeVectorStore.from_texts(
            texts=texts,
            ids=ids,
            embedding=embedding_openai,
            index_name=INDEX_NAME,
            namespace=NAMESPACE_NAME,
        )
        time.sleep(DEFAULT_SLEEP)  # prevent race condition
        index_stats = self.index.describe_index_stats()
        assert index_stats["namespaces"][NAMESPACE_NAME]["vector_count"] == len(texts)

        ids_1 = [uuid.uuid4().hex for _ in range(len(texts))]
        PineconeVectorStore.from_texts(
            texts=[t + "-1" for t in texts],
            ids=ids_1,
            embedding=embedding_openai,
            index_name=INDEX_NAME,
            namespace=NAMESPACE_NAME,
        )
        time.sleep(DEFAULT_SLEEP)  # prevent race condition
        index_stats = self.index.describe_index_stats()
        assert (
            index_stats["namespaces"][NAMESPACE_NAME]["vector_count"] == len(texts) * 2
        )
        # only focused on this namespace now
        # assert index_stats["total_vector_count"] == len(texts) * 2

    @pytest.mark.xfail(reason="relevance score just over 1")
    def test_relevance_score_bound(self, embedding_openai: OpenAIEmbeddings) -> None:
        """Ensures all relevance scores are between 0 and 1."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = PineconeVectorStore.from_texts(
            texts,
            embedding_openai,
            index_name=INDEX_NAME,
            metadatas=metadatas,
        )
        # wait for the index to be ready
        time.sleep(DEFAULT_SLEEP)
        output = docsearch.similarity_search_with_relevance_scores("foo", k=3)
        print(output)  # noqa: T201
        assert all(
            (1 >= score or np.isclose(score, 1)) and score >= 0 for _, score in output
        )

    @pytest.mark.skipif(reason="slow to run for benchmark")
    @pytest.mark.parametrize(
        "pool_threads,batch_size,embeddings_chunk_size,data_multiplier",
        [
            (
                1,
                32,
                32,
                1000,
            ),  # simulate single threaded with embeddings_chunk_size = batch_size = 32
            (
                1,
                32,
                1000,
                1000,
            ),  # simulate single threaded with embeddings_chunk_size = 1000
            (
                4,
                32,
                1000,
                1000,
            ),  # simulate 4 threaded with embeddings_chunk_size = 1000
            (20, 64, 5000, 1000),
        ],  # simulate 20 threaded with embeddings_chunk_size = 5000
    )
    def test_from_texts_with_metadatas_benchmark(
        self,
        pool_threads: int,
        batch_size: int,
        embeddings_chunk_size: int,
        data_multiplier: int,
        documents: List[Document],
        embedding_openai: OpenAIEmbeddings,
    ) -> None:
        """Test end to end construction and search."""

        texts = [document.page_content for document in documents] * data_multiplier
        uuids = [uuid.uuid4().hex for _ in range(len(texts))]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = PineconeVectorStore.from_texts(
            texts,
            embedding_openai,
            ids=uuids,
            metadatas=metadatas,
            index_name=INDEX_NAME,
            namespace=NAMESPACE_NAME,
            pool_threads=pool_threads,
            batch_size=batch_size,
            embeddings_chunk_size=embeddings_chunk_size,
        )

        query = "What did the president say about Ketanji Brown Jackson"
        _ = docsearch.similarity_search(query, k=1, namespace=NAMESPACE_NAME)

    @pytest.fixture
    def mock_pool_not_supported(self, mocker: MockerFixture) -> None:
        """
        This is the error thrown when multiprocessing is not supported.
        See https://github.com/langchain-ai/langchain/issues/11168
        """
        mocker.patch(
            "multiprocessing.synchronize.SemLock.__init__",
            side_effect=OSError(
                "FileNotFoundError: [Errno 2] No such file or directory"
            ),
        )

    @pytest.mark.usefixtures("mock_pool_not_supported")
    def test_that_async_freq_uses_multiprocessing(
        self, texts: List[str], embedding_openai: OpenAIEmbeddings
    ) -> None:
        with pytest.raises(OSError):
            PineconeVectorStore.from_texts(
                texts=texts,
                embedding=embedding_openai,
                index_name=INDEX_NAME,
                namespace=NAMESPACE_NAME,
                async_req=True,
            )

    @pytest.mark.usefixtures("mock_pool_not_supported")
    def test_that_async_freq_false_enabled_singlethreading(
        self, texts: List[str], embedding_openai: OpenAIEmbeddings
    ) -> None:
        PineconeVectorStore.from_texts(
            texts=texts,
            embedding=embedding_openai,
            index_name=INDEX_NAME,
            namespace=NAMESPACE_NAME,
            async_req=False,
        )
