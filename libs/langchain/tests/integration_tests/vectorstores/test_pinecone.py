import importlib
import os
import time
import uuid
from typing import List

import numpy as np
import pinecone
import pytest

from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone

index_name = "langchain-test-index"  # name of the index
namespace_name = "langchain-test-namespace"  # name of the namespace
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


class TestPinecone:
    index: pinecone.Index

    @classmethod
    def setup_class(cls) -> None:
        reset_pinecone()

        cls.index = pinecone.Index(index_name)

        if index_name in pinecone.list_indexes():
            index_stats = cls.index.describe_index_stats()
            if index_stats["dimension"] == dimension:
                # delete all the vectors in the index if the dimension is the same
                # from all namespaces
                index_stats = cls.index.describe_index_stats()
                for _namespace_name in index_stats["namespaces"].keys():
                    cls.index.delete(delete_all=True, namespace=_namespace_name)

            else:
                pinecone.delete_index(index_name)
                pinecone.create_index(name=index_name, dimension=dimension)
        else:
            pinecone.create_index(name=index_name, dimension=dimension)

        # insure the index is empty
        index_stats = cls.index.describe_index_stats()
        assert index_stats["dimension"] == dimension
        if index_stats["namespaces"].get(namespace_name) is not None:
            assert index_stats["namespaces"][namespace_name]["vector_count"] == 0

    @classmethod
    def teardown_class(cls) -> None:
        index_stats = cls.index.describe_index_stats()
        for _namespace_name in index_stats["namespaces"].keys():
            cls.index.delete(delete_all=True, namespace=_namespace_name)

        reset_pinecone()

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        # delete all the vectors in the index
        index_stats = self.index.describe_index_stats()
        for _namespace_name in index_stats["namespaces"].keys():
            self.index.delete(delete_all=True, namespace=_namespace_name)

        reset_pinecone()

    @pytest.mark.vcr()
    def test_from_texts(
        self, texts: List[str], embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test end to end construction and search."""
        unique_id = uuid.uuid4().hex
        needs = f"foobuu {unique_id} booo"
        texts.insert(0, needs)

        docsearch = Pinecone.from_texts(
            texts=texts,
            embedding=embedding_openai,
            index_name=index_name,
            namespace=namespace_name,
        )
        output = docsearch.similarity_search(unique_id, k=1, namespace=namespace_name)
        assert output == [Document(page_content=needs)]

    @pytest.mark.vcr()
    def test_from_texts_with_metadatas(
        self, texts: List[str], embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test end to end construction and search."""

        unique_id = uuid.uuid4().hex
        needs = f"foobuu {unique_id} booo"
        texts.insert(0, needs)

        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = Pinecone.from_texts(
            texts,
            embedding_openai,
            index_name=index_name,
            metadatas=metadatas,
            namespace=namespace_name,
        )
        output = docsearch.similarity_search(needs, k=1, namespace=namespace_name)

        # TODO: why metadata={"page": 0.0}) instead of {"page": 0}?
        assert output == [Document(page_content=needs, metadata={"page": 0.0})]

    @pytest.mark.vcr()
    def test_from_texts_with_scores(self, embedding_openai: OpenAIEmbeddings) -> None:
        """Test end to end construction and search with scores and IDs."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = Pinecone.from_texts(
            texts,
            embedding_openai,
            index_name=index_name,
            metadatas=metadatas,
            namespace=namespace_name,
        )
        output = docsearch.similarity_search_with_score(
            "foo", k=3, namespace=namespace_name
        )
        docs = [o[0] for o in output]
        scores = [o[1] for o in output]
        sorted_documents = sorted(docs, key=lambda x: x.metadata["page"])

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
        Pinecone.from_texts(
            texts_1,
            embedding_openai,
            index_name=index_name,
            metadatas=metadatas,
            namespace=f"{index_name}-1",
        )

        texts_2 = ["foo2", "bar2", "baz2"]
        metadatas = [{"page": i} for i in range(len(texts_2))]

        Pinecone.from_texts(
            texts_2,
            embedding_openai,
            index_name=index_name,
            metadatas=metadatas,
            namespace=f"{index_name}-2",
        )

        # Search with namespace
        docsearch = Pinecone.from_existing_index(
            index_name=index_name,
            embedding=embedding_openai,
            namespace=f"{index_name}-1",
        )
        output = docsearch.similarity_search("foo", k=20, namespace=f"{index_name}-1")
        # check that we don't get results from the other namespace
        page_contents = sorted(set([o.page_content for o in output]))
        assert all(content in ["foo", "bar", "baz"] for content in page_contents)
        assert all(content not in ["foo2", "bar2", "baz2"] for content in page_contents)

    def test_add_documents_with_ids(
        self, texts: List[str], embedding_openai: OpenAIEmbeddings
    ) -> None:
        ids = [uuid.uuid4().hex for _ in range(len(texts))]
        Pinecone.from_texts(
            texts=texts,
            ids=ids,
            embedding=embedding_openai,
            index_name=index_name,
            namespace=index_name,
        )
        index_stats = self.index.describe_index_stats()
        assert index_stats["namespaces"][index_name]["vector_count"] == len(texts)

        ids_1 = [uuid.uuid4().hex for _ in range(len(texts))]
        Pinecone.from_texts(
            texts=texts,
            ids=ids_1,
            embedding=embedding_openai,
            index_name=index_name,
            namespace=index_name,
        )
        index_stats = self.index.describe_index_stats()
        assert index_stats["namespaces"][index_name]["vector_count"] == len(texts) * 2
        assert index_stats["total_vector_count"] == len(texts) * 2

    @pytest.mark.vcr()
    def test_relevance_score_bound(self, embedding_openai: OpenAIEmbeddings) -> None:
        """Ensures all relevance scores are between 0 and 1."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = Pinecone.from_texts(
            texts,
            embedding_openai,
            index_name=index_name,
            metadatas=metadatas,
        )
        # wait for the index to be ready
        time.sleep(20)
        output = docsearch.similarity_search_with_relevance_scores("foo", k=3)
        assert all(
            (1 >= score or np.isclose(score, 1)) and score >= 0 for _, score in output
        )
