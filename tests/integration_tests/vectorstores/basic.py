import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, List, Type

import pytest

from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores import VectorStore


# TODO: This is a DRAFT of the test class that will be used to test all vector stores
# TODO: NOT complete yet
class AbstractVectorStoreTest(ABC):
    vector_store_class: Type[VectorStore]

    @classmethod
    @abstractmethod
    def setup_class(cls) -> None:
        """
        Prepare the test environment to connect to the vector store, create
        necessary namespaces, etc.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def teardown_class(cls) -> None:
        """
        Clean up the test environment after running the tests, including
        disconnecting from the vector store and deleting any namespaces created
        during testing.
        """
        raise NotImplementedError()

    @pytest.fixture(autouse=True)
    @abstractmethod
    def setup(self) -> None:
        """
        A fixture that is fired before each test in the class to prepare the
        environment for the test, including setting up any necessary data.
        """
        raise NotImplementedError()

    @abstractmethod
    def test_from_texts(self, texts: List[str], embedding: Embeddings) -> None:
        """
        Test that the vector store can be populated from a list of texts.
        """
        raise NotImplementedError()

    @abstractmethod
    def test_from_texts_with_meta(
            self, texts: List[str], embedding: Embeddings
    ) -> None:
        """
        Test that the vector store can be populated from a list of texts and
        associated metadata.
        """
        raise NotImplementedError()

    @abstractmethod
    def test_from_texts_with_ids(
            self, texts: List[str], ids: List[str], embedding: Embeddings
    ) -> None:
        """
        Test that an existing document in the vector store is updated with new content.
        """
        raise NotImplementedError()

    @abstractmethod
    def test_from_documents(
            self, documents: List[Document], embedding: Embeddings
    ) -> None:
        """
        Test that the vector store can be populated from a list of documents.
        """
        raise NotImplementedError()

    @abstractmethod
    def test_from_documents_with_meta(
            self, documents: List[Document], embedding: Embeddings
    ) -> None:
        """
        Test that the vector store can be populated from a list of documents and
        associated metadata.
        """
        raise NotImplementedError()

    @abstractmethod
    def test_add_documents_with_ids(
            self, documents: List[Document], embedding: Embeddings
    ) -> None:
        """
        Test that documents can be added to the vector store with IDs and
        that the updated document can be retrieved correctly.
        """
        raise NotImplementedError()

    @abstractmethod
    def test_update_documents_with_ids(
            self, documents: List[Document], embedding: Embeddings
    ) -> None:
        """
        Test that existing documents can be updated in the vector store with IDs and
        that the updated document can be retrieved correctly.
        Make sure that the document will not be duplicated.
        """
        raise NotImplementedError()

    @abstractmethod
    def test_delete_document_with_id(
            self, document: Document, embedding: Embeddings
    ) -> None:
        """
        Test that document can be deleted from the vector store with ID.
        """
        raise NotImplementedError()

    @abstractmethod
    def test_similarity_search(
            self, query: str, expected_results: List[Document]
    ) -> None:
        """
        Test that the vector store can perform a similarity search and return the
        expected results.
        """
        raise NotImplementedError()

    @abstractmethod
    def test_similarity_search_with_meta(
            self, query: str, expected_results: List[Document]
    ) -> None:
        """
        Test that the vector store can perform a similarity search with metadata and
        return the expected results.
        """
        raise NotImplementedError()

    @abstractmethod
    def test_similarity_search_with_filters(
            self, query: str, expected_results: List[Document]
    ) -> None:
        """
        Test that the vector store can perform a similarity search with filters and
        return the expected results.
        Note: Not all vector stores are supported with this API.
        """
        raise NotImplementedError()


class AbstractVectorStoreTestRemote(AbstractVectorStoreTest, ABC):
    """Test vector stores that are remote."""

    pass


class AbstractVectorStoreTestLocal(AbstractVectorStoreTest, ABC):
    """Test vector stores can be either local with a persistent database on a
    filesystem or stored in memory. Every database on a filesystem should be stored
    in a temporary directory and cleaned up after each test."""

    @pytest.fixture()
    def db_store_dir(self) -> Generator[Path, None, None]:
        """A fixture that returns the path for the temporary directory where the
        database is stored."""

        # use a temporary directory for the database
        temp_dir = tempfile.mkdtemp()

        # not sure if this is necessary to use try/finally
        try:
            yield Path(temp_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class BaseVectorStoreTest(AbstractVectorStoreTest, ABC):
    """Doing all hard works here"""
    vector_store_class: Type[VectorStore]


class TestChromaRemote(AbstractVectorStoreTestRemote, BaseVectorStoreTest, ABC):
    vector_store_class = Chroma

    def setup_class(self) -> None:
        """Prepare the test environment to connect to the vector store"""


class TestChromaLocal(AbstractVectorStoreTestLocal, BaseVectorStoreTest, ABC):
    vector_store_class = Chroma

    def setup_class(self, db_store_dir) -> None:
        """Prepare the test environment to prepare local DB for the vector store"""
