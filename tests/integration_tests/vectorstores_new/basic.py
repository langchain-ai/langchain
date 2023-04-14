# tests/integration_tests/vectorstores_new/basic.py
import logging
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, List, Type

import pytest

from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores import VectorStore

# TODO: Move the logging setup and set the logging level in the fixture setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(logging.DEBUG)


class AbstractVectorStoreStaticTest(ABC):
    """
    Abstract base class defining the interface for vector store tests.

    This class is used as a template for creating test classes that verify the
    functionality of a `VectorStore`. It defines an abstract interface that
    specifies the methods that a vector store must support, and provides
    default implementations for setup and teardown methods that are run
    before and after the tests are run.

    Attributes:
        vector_store_class: The concrete class that implements the `VectorStore`
            interface.
    """

    vector_store_class: Type[VectorStore]

    @classmethod
    @abstractmethod
    def setup_class(cls) -> None:
        """
        Prepare the test environment to connect to the vector store, create
        necessary namespaces, etc.

        This method is called once before any tests are run to set up the test
        environment. It should create any resources needed to connect to the
        vector store, such as a temporary database file or a connection to a
        remote server.

        Raises:
            NotImplementedError: This method must be implemented by the
                concrete test class.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def teardown_class(cls) -> None:
        """
        Clean up the test environment after running the tests, including
        disconnecting from the vector store and deleting any artifacts created
        during testing.

        This method is called once after all tests have been run to clean up
        any resources that were created during testing. It should close any
        connections to the vector store and delete any temporary files or
        directories that were created.

        Raises:
            NotImplementedError: This method must be implemented by the
                concrete test class.
        """
        raise NotImplementedError()

    @abstractmethod
    def setup_method(self) -> None:
        """
        This method is called before each individual test is run to set up
        the environment for the test. It should create any data or resources
        that are needed for the test.

        Raises:
            NotImplementedError: This method must be implemented by the
                concrete test class.
        """
        raise NotImplementedError()

    @abstractmethod
    def teardown_method(self) -> None:
        """
        This method is called after each individual test is run to clean up
        any resources that were created during the test, such as documents
        that were added to the vector store.

        Raises:
            NotImplementedError: This method must be implemented by the
                concrete test class.
        """
        raise NotImplementedError()

    @abstractmethod
    def test_from_texts(self, texts: List[str], embedding: Embeddings) -> None:
        """
        Test that the vector store can be populated from a list of texts.
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
    def test_similarity_search(self, query: str) -> None:
        """
        Test that the vector store can perform a similarity search and return the
        expected results.
        """
        raise NotImplementedError()


class VectorStoreStaticTestMixin(AbstractVectorStoreStaticTest, ABC):
    """
    Mixin class containing shared test methods for vector stores.
    """

    def test_from_texts(self, texts: List[str], embedding: Embeddings) -> None:
        logger.debug("test_from_texts")

    def test_from_documents(
        self, documents: List[Document], embedding: Embeddings
    ) -> None:
        logger.debug("test_from_documents")

    def test_similarity_search(self, query: str) -> None:
        logger.debug("test_similarity_search")


class BaseVectorStoreStaticTestRemote(VectorStoreStaticTestMixin, ABC):
    """
    Base class for remote vector store tests, which inherit the test cases
    defined in VectorStoreTestMixin.
    """


class BaseVectorStoreStaticTestLocal(VectorStoreStaticTestMixin, ABC):
    """
    Base class for local vector store tests, which inherit the test cases
    defined in VectorStoreTestMixin.
    """

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
