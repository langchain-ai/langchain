from __future__ import annotations

import importlib
import os
import uuid
from typing import TYPE_CHECKING, Generator, List

import pytest

from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.pinecone_hybrid_search import (
    PineconeHybridSearchRetriever,
    PineconeIndexUpsert,
)

if TYPE_CHECKING:
    from pinecone_text.sparse.bm25_encoder import BM25Encoder

index_name = "langchain-pinecone-hybrid-search"  # name of the index
dimension = 1536  # dimension of the embeddings


def reset_pinecone() -> None:
    assert os.environ.get("PINECONE_API_KEY") is not None
    assert os.environ.get("PINECONE_ENVIRONMENT") is not None

    import pinecone

    importlib.reload(pinecone)

    pinecone.init(
        api_key=os.environ.get("PINECONE_API_KEY"),
        environment=os.environ.get("PINECONE_ENVIRONMENT"),
    )


@pytest.fixture(scope="function")
def texts() -> Generator[List[str], None, None]:
    # Load the documents from a file located in the fixtures directory
    documents = TextLoader(
        os.path.join(
            os.path.dirname(__file__), "../vectorstores/fixtures", "sharks.txt"
        )
    ).load()

    yield [doc.page_content for doc in documents]


@pytest.fixture(scope="module")
def embedding_openai() -> OpenAIEmbeddings:
    return OpenAIEmbeddings()


@pytest.fixture(scope="function")
def bm25_encoder() -> BM25Encoder:
    from pinecone_text.sparse.bm25_encoder import BM25Encoder

    return BM25Encoder().default()


class TestPinecone:
    @classmethod
    def setup_class(cls) -> None:
        import pinecone

        if index_name in pinecone.list_indexes():
            pinecone.delete_index(index_name)
        pinecone.create_index(
            name=index_name,
            dimension=dimension,
            metric="dotproduct",
            pod_type="s1",
            metadata_config={"indexed": []},
        )

    @classmethod
    def teardown_class(cls) -> None:
        reset_pinecone()

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        reset_pinecone()

    @pytest.mark.vcr()
    def test_add_text_threaded(
        self,
        texts: List[str],
        embedding_openai: OpenAIEmbeddings,
        bm25_encoder: BM25Encoder,
    ) -> None:
        """Test end to end construction and search."""
        index = PineconeIndexUpsert.get_index_upsert(index_name, pool_threads=2)
        needs = "foobuu booo"
        texts.insert(0, needs)
        retrieval = PineconeHybridSearchRetriever(
            index_upsert=index,
            top_k=1,
            embeddings=embedding_openai,
            sparse_encoder=bm25_encoder,
        )
        retrieval.add_texts(texts, batch_size=32, chunk_size=1000)
        res = retrieval.get_relevant_documents(needs)
        assert res == [Document(page_content=needs)]

    @pytest.mark.vcr()
    def test_add_text_synced(
        self,
        texts: List[str],
        embedding_openai: OpenAIEmbeddings,
        bm25_encoder: BM25Encoder,
    ) -> None:
        """Test end to end construction and search."""
        index = PineconeIndexUpsert.get_index_upsert(index_name)
        needs = "foobuu booo"
        texts.insert(0, needs)
        retrieval = PineconeHybridSearchRetriever(
            index_upsert=index,
            top_k=1,
            embeddings=embedding_openai,
            sparse_encoder=bm25_encoder,
        )
        retrieval.add_texts(texts)
        res = retrieval.get_relevant_documents(needs)
        assert res == [Document(page_content=needs)]

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
        texts: List[str],
        embedding_openai: OpenAIEmbeddings,
        bm25_encoder: BM25Encoder,
    ) -> None:
        """Test end to end construction and search."""

        index = PineconeIndexUpsert.get_index_upsert(
            index_name, pool_threads=pool_threads
        )

        texts *= data_multiplier
        uuids = [uuid.uuid4().hex for _ in range(len(texts))]
        metadatas = [{"page": i} for i in range(len(texts))]

        retrieval = PineconeHybridSearchRetriever(
            index_upsert=index,
            top_k=1,
            embeddings=embedding_openai,
            sparse_encoder=bm25_encoder,
        )

        retrieval.add_texts(
            texts,
            uuids,
            metadatas,
            batch_size=batch_size,
            chunk_size=embeddings_chunk_size,
        )
