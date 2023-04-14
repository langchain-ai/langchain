# tests/integration_tests/vectorstores_new/basic.py
import logging
from abc import ABC, abstractmethod
from typing import List, Type, Union

import pytest

from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores import VectorStore

# TODO: Move the logging setup and set the logging level in the fixture setup
# Set up logging configuration
logging.basicConfig(level=logging.DEBUG)


class AbstractVectorStoreStaticTest(ABC):
    """
    Define an abstract base class that provides default implementations for
    setup and teardown methods that are run before and after the tests are run

    Attributes:
        vector_store_class: The concrete class that implements the `VectorStore`
            interface.
    """

    vector_store_class: Type[VectorStore]
    docsearch: Union[VectorStore, None] = None
    logger: logging.Logger

    @pytest.fixture(autouse=True)
    def _setup_logger(self) -> None:
        # Set up logger for each test class
        self.logger = logging.getLogger(self.__class__.__name__)
        # TODO: Move the logging setup and set the logging level in the fixture setup
        # Set up logging level
        logging.getLogger(__name__).setLevel(logging.DEBUG)

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


class VectorStoreStaticTestMixin(AbstractVectorStoreStaticTest, ABC):
    """
    Mixin class containing shared test methods for vector stores.
    """

    @pytest.mark.vcr()
    def similarity_search(self, query: str, k: int = 1) -> List[Document]:
        """
        Perform a similarity search on the vector store for the given query.

        Args:
            query (str): The query string to search for.
            k (int): The number of results to return (default: 1).

        Returns:
            A list of Document objects representing the search results.
        """
        self.logger.debug("test_similarity_search")
        assert self.docsearch is not None

        output = self.docsearch.similarity_search(query, k=1)
        assert len(output) >= 1
        return output

    @pytest.mark.vcr()
    def test_from_texts(
        self, texts: List[str], embedding: Embeddings, query: str
    ) -> None:
        """
        Test creating a VectorStore from a list of texts.
        """
        self.logger.debug("test_from_texts")

        self.docsearch = self.vector_store_class.from_texts(
            texts,
            embedding=embedding,
        )

        self.similarity_search(query=query)

    @pytest.mark.vcr()
    def test_from_documents(
        self, documents: List[Document], embedding: Embeddings, query: str
    ) -> None:
        """
        Test creating a VectorStore from a list of Documents.
        """
        self.logger.debug("test_from_documents")

        self.docsearch = self.vector_store_class.from_documents(
            documents,
            embedding=embedding,
        )

        self.similarity_search(query=query)


class BaseVectorStoreStaticTest(VectorStoreStaticTestMixin, ABC):
    """
    Base class for remote vector store tests, which inherit the test cases
    defined in VectorStoreTestMixin.
    """
